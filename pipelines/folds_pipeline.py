"""Create composition-group splits for each chemical domain."""


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

from core.folds import make_domain_splits, save_domain_splits


FeatureStep = str  # e.g. "magpie", "simple", "frac"


@dataclass(frozen=True)
class FoldsPipelineConfig:
    """
    Orchestrator-only configuration.

    This pipeline generates CV folds for one or multiple feature variants.
    """
    # Dataset columns
    chem_class_col: str = "chem_class"          # domain: N/O/M
    label_col: str = "centrosymmetry"           # label: C/N (as used in notebooks)
    group_col: str = "group_id"                 # group key for group-aware splitting
    sg_col: str = "sg_number_spglib"            # Minimum class-size filter.

    # Filters
    domains: Tuple[str, str, str] = ("N", "O", "M")
    allowed_labels: Tuple[str, str] = ("C", "N")
    require_sg_notna: bool = True

    # Split params
    min_class_samples: int = 20
    outer_n_splits: int = 5
    inner_n_splits: int = 5
    seed: int = 42
    test_outer_fold: int = 0

    # Behavior
    verbose: bool = True
    write_summary: bool = True


def _select_io_from_cfg(cfg, step: FeatureStep) -> Tuple[str, str, str]:
    """
    Map a feature step name to (input_features_csv, folds_template, folds_bad_template).

    Supported steps:
      - "magpie" / "default": cfg.DATASET_FEATURES + cfg.DATASET_FOLDS_TEMPLATE
      - "simple":            cfg.DATASET_FEATURES_SIMPLE + cfg.DATASET_FOLDS_TEMPLATE_S
      - "frac":              cfg.DATASET_FEATURES_FRAC + cfg.DATASET_FOLDS_TEMPLATE_F
    """
    s = (step or "default").strip().lower()

    if s in ("default", "magpie"):
        input_features = cfg.DATASET_FEATURES
        folds_template = cfg.DATASET_FOLDS_TEMPLATE
        folds_bad_template = cfg.DATASET_FOLDS_BAD_TEMPLATE
    elif s == "simple":
        input_features = cfg.DATASET_FEATURES_SIMPLE
        folds_template = cfg.DATASET_FOLDS_TEMPLATE_S
        folds_bad_template = cfg.DATASET_FOLDS_BAD_TEMPLATE_S
    elif s in ("frac", "fractional", "f"):
        input_features = cfg.DATASET_FEATURES_FRAC
        folds_template = cfg.DATASET_FOLDS_TEMPLATE_F
        folds_bad_template = cfg.DATASET_FOLDS_BAD_TEMPLATE_F
    else:
        raise ValueError(f"Unknown folds feature step '{step}'. Use: magpie | simple | frac")

    return input_features, folds_template, folds_bad_template


def _normalize_steps(steps: Optional[Union[FeatureStep, Iterable[FeatureStep]]]) -> List[FeatureStep]:
    """
    Normalize input into a list of step strings.

    Accepts:
      - None -> ["magpie"]
      - "frac" -> ["frac"]
      - ("magpie","simple","frac") -> ["magpie","simple","frac"]
      - ["simple","frac"] -> ["simple","frac"]
    """
    if steps is None:
        return ["magpie"]
    if isinstance(steps, str):
        return [steps]
    return list(steps)

def summarize_domain(df: pd.DataFrame, domain: str, *, label_col: str, group_col: str) -> None:
    """
    Print a compact audit summary for a single domain split dataframe.

    Expected columns (produced by make_domain_splits):
      - label_col (e.g. "centrosymmetry")
      - "is_test" (0/1)
      - optionally "fold" for train/val
      - group_col (e.g. "group_id")
    """
    print(f"\n=== SUMMARY FOR DOMAIN {domain} ===")
    print(f"Total samples: {len(df)}")

    if label_col in df.columns:
        print("\nLabel counts:")
        print(df[label_col].value_counts(dropna=False))

    if "is_test" in df.columns:
        print("\nTrain/Test counts:")
        print(df["is_test"].value_counts(dropna=False).rename({0: "train_val", 1: "test"}))

        n_train_val = int((df["is_test"] == 0).sum())
        n_test = int((df["is_test"] == 1).sum())
        print(f"Train/Val samples: {n_train_val}")
        print(f"Test samples:      {n_test}")

        if "fold" in df.columns:
            print("\nFold distribution (train/val only):")
            print(df[df["is_test"] == 0]["fold"].value_counts().sort_index())

    if group_col in df.columns:
        print("\nUnique groups (overall):", df[group_col].nunique())
        if "is_test" in df.columns:
            print("Groups in TEST:", df[df["is_test"] == 1][group_col].nunique())

    if "is_test" in df.columns and label_col in df.columns:
        print("\nClass balance in TRAIN_VAL:")
        print(df[df["is_test"] == 0][label_col].value_counts(dropna=False))

        print("\nClass balance in TEST:")
        print(df[df["is_test"] == 1][label_col].value_counts(dropna=False))


class FoldsPipeline:
    """
    Folds orchestrator.

    Generates folds per domain for one or multiple feature variants.
    Uses existing core functions: make_domain_splits(...) and save_domain_splits(...).
    """

    def __init__(self, cfg, config: Optional[FoldsPipelineConfig] = None):
        self.cfg = cfg
        self.c = config or FoldsPipelineConfig()

    # public API
    def run(self, *, steps=None, input_csv: Optional[str] = None):
        """
        Run folds generation for one or multiple feature steps.

        Parameters
        ----------
        steps : None | str | iterable[str]
            Feature steps to process, e.g.:
              - None -> default ("magpie")
              - "frac"
              - ("magpie", "simple", "frac")
        input_csv : str or None
            Optional features file for this call. Applies only when a single
            step is requested. Other calls use the configured feature-step mapping.
        """
        from core.notebook_logger import NotebookLogger
        import os

        steps_list = _normalize_steps(steps)
        if input_csv is not None and len(steps_list) != 1:
            raise ValueError("input_csv overrides the features file for a single step.")

        logger = NotebookLogger(
            project_dir=self.cfg.PROJECT_DIR,
            prefix=self.cfg.PROJECT_NAME,
        )

        logger.start(header_lines=[
            f"Project        : {self.cfg.PROJECT_NAME}",
            'Pipeline       : FoldsPipeline',
            f"Steps          : {steps_list}",
            f"CWD            : {os.getcwd()}",
            f"Domains        : {self.c.domains}",
            f"Group column   : {self.c.group_col}",
            f"Label column   : {self.c.label_col}",
            f"Outer folds    : {self.c.outer_n_splits}",
            f"Inner folds    : {self.c.inner_n_splits}",
            f"Seed           : {self.c.seed}",
        ])

        results = {}

        try:
            for step in steps_list:
                input_features, folds_template, folds_bad_template = _select_io_from_cfg(
                    self.cfg, step
                )
                if input_csv is not None:
                    input_features = input_csv

                if self.c.verbose:
                    print(f"\n=== STEP: {step} ===")
                    print("Input features:", input_features)
                    print("Folds template:", folds_template)
                    print("Folds BAD template:", folds_bad_template)
                self._ensure_exists(
                    input_features,
                    what=f"features dataset (step='{step}')"
                )

                df = pd.read_csv(input_features)

                # Preflight checks
                required = [self.c.chem_class_col, self.c.label_col, self.c.group_col]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(
                        f"Missing required columns in features dataset "
                        f"(step='{step}'): {missing}"
                    )

                # Apply the minimum class-size filter.
                if self.c.require_sg_notna and (self.c.sg_col in df.columns):
                    df = df[df[self.c.sg_col].notna()].copy()

                step_out = {}

                for domain in self.c.domains:
                    dfD = df[df[self.c.chem_class_col] == domain].copy()
                    dfD = dfD[dfD[self.c.label_col].isin(self.c.allowed_labels)].copy()

                    if len(dfD) == 0:
                        if self.c.verbose:
                            print(f"[WARN] Step={step}, domain={domain}: empty, skipping.")
                        continue

                    df_ok, df_bad = make_domain_splits(
                        dfD,
                        label_col=self.c.label_col,
                        group_col=self.c.group_col,
                        min_class_samples=self.c.min_class_samples,
                        outer_n_splits=self.c.outer_n_splits,
                        inner_n_splits=self.c.inner_n_splits,
                        test_outer_fold=self.c.test_outer_fold,
                        random_state=self.c.seed,
                        verbose=self.c.verbose,
                    )

                    out_ok = folds_template.format(domain=domain)
                    out_bad = folds_bad_template.format(domain=domain)

                    save_domain_splits(
                        df_ok=df_ok,
                        df_bad=df_bad,
                        output_path=out_ok,
                        bad_output_path=out_bad,
                        domain=domain,
                        label_col=self.c.label_col,
                        group_col=self.c.group_col,
                        verbose=self.c.verbose,
                    )

                    step_out[domain] = df_ok
                    if self.c.write_summary:
                        summarize_domain(
                            df_ok,
                            domain=domain,
                            label_col=self.c.label_col,
                            group_col=self.c.group_col,
                        )

                results[step] = step_out

        finally:
            logger.stop()

        return results


    # helpers
    def _ensure_exists(self, path: str, what: str):
        """
        Ensure that a required artifact exists on disk.

        Parameters
        ----------
        path : str
            Path to the required artifact.
        what : str
            Human-readable description of the artifact.
        """
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact for {what}: {p}")
        return p
