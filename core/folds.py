"""Create stratified train/test and validation partitions grouped by composition."""

import pandas as pd
from tqdm.auto import tqdm

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as e:
    raise ImportError(
        "StratifiedGroupKFold is required. Please upgrade scikit-learn to a version "
        "that includes StratifiedGroupKFold (>= 1.1 recommended)."
    ) from e


def make_domain_splits(
    df: pd.DataFrame,
    label_col: str = "centrosymmetry",
    group_col: str = "group_id",
    outer_n_splits: int = 5,
    inner_n_splits: int = 5,
    test_outer_fold: int = 0,
    min_class_samples: int = 20,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Create nested splits for a single domain using StratifiedGroupKFold.

    Output columns:
        - outer_fold: 0..outer_n_splits-1
        - is_test: 1 if outer_fold == test_outer_fold else 0
        - fold: 0..inner_n_splits-1 for train/val (is_test == 0), -1 for test

    Notes:
        - Group leakage is prevented by group_col.
        - Stratification is performed on label_col.
        - A single outer fold is designated as the test set.
    """

    df = df.copy()

    # Identify invalid rows for splitting
    bad_mask = df[group_col].isna() | df[label_col].isna()
    df_bad = df[bad_mask].copy()

    df = df[~bad_mask].copy()

    if verbose and len(df_bad) > 0:
        print(
            f"[make_domain_splits] Marked {len(df_bad)} rows as splits_bad "
            f"(NaN in {group_col} or {label_col})"
        )

    # Ensure groups are a single comparable dtype
    df[group_col] = df[group_col].astype(str)

    # Basic validations
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")

    y = df[label_col]
    if y.isna().any():
        raise ValueError("NaNs detected in label column.")

    min_class_all = y.value_counts().min()
    if min_class_all < min_class_samples:
        raise ValueError(
            f"Minority class has only {min_class_all} samples (< {min_class_samples})."
        )

    n_groups = df[group_col].nunique()
    if n_groups < outer_n_splits:
        raise ValueError(
            f"Not enough groups for outer split: groups={n_groups}, outer_n_splits={outer_n_splits}."
        )

    # Outer split (test selection)
    outer_cv = StratifiedGroupKFold(
        n_splits=outer_n_splits,
        shuffle=True,
        random_state=random_state,
    )

    df["outer_fold"] = -1
    df["is_test"] = 0
    df["fold"] = -1

    if verbose:
        print(f"[make_domain_splits] Outer split: n_splits={outer_n_splits}, test_fold={test_outer_fold}")

    for fold_id, (_, test_idx) in enumerate(outer_cv.split(df, y, groups=df[group_col])):
        df.loc[df.index[test_idx], "outer_fold"] = fold_id

    df.loc[df["outer_fold"] == test_outer_fold, "is_test"] = 1

    # Inner split (only on train/val)
    df_tv = df[df["is_test"] == 0].copy()
    y_tv = df_tv[label_col]
    groups_tv = df_tv[group_col]

    min_class_tv = y_tv.value_counts().min()
    if min_class_tv < min_class_samples:
        raise ValueError(
            f"After removing test set, minority class has only {min_class_tv} samples (< {min_class_samples})."
        )

    n_groups_tv = df_tv[group_col].nunique()
    if n_groups_tv < inner_n_splits:
        raise ValueError(
            f"Not enough groups for inner split: groups={n_groups_tv}, inner_n_splits={inner_n_splits}."
        )

    inner_cv = StratifiedGroupKFold(
        n_splits=inner_n_splits,
        shuffle=True,
        random_state=random_state,
    )

    if verbose:
        print(f"[make_domain_splits] Inner split: n_splits={inner_n_splits}")

    for fold_id, (_, val_idx) in tqdm(
        enumerate(inner_cv.split(df_tv, y_tv, groups=groups_tv)),
        total=inner_n_splits,
        desc="Inner folds",
        disable=not verbose,
    ):
        val_global_idx = df_tv.index[val_idx]
        df.loc[val_global_idx, "fold"] = fold_id

    return df, df_bad


def save_domain_splits(
    df_ok: pd.DataFrame,
    output_path: str,
    domain: str,
    df_bad: pd.DataFrame | None = None,
    bad_output_path: str | None = None,
    label_col: str = "centrosymmetry",
    group_col: str = "group_id",
    verbose: bool = True,
):
    df_ok.to_csv(output_path, index=False)

    if df_bad is not None and bad_output_path is not None:
        df_bad.to_csv(bad_output_path, index=False)

    if verbose:
        print(
            f"FAST pass summary: domain={domain} OK={len(df_ok)} "
            f"BAD={0 if df_bad is None else len(df_bad)}"
        )
        print("Label distribution:", df_ok[label_col].value_counts().to_dict())
        if "fold" in df_ok.columns:
            print(
                "Fold distribution (train/val only):",
                df_ok[df_ok["is_test"] == 0]["fold"]
                .value_counts()
                .sort_index()
                .to_dict(),
            )
        print("Saved OK:", output_path)
        if df_bad is not None and bad_output_path is not None:
            print("Saved BAD:", bad_output_path)
