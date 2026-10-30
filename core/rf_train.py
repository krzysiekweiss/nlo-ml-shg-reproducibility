"""Fit and evaluate Random Forest classifiers on fixed composition-group partitions."""


from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd

from . import ml_utils
from .task_handlers import get_task_handler
from targets.preset import TargetSpec

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def build_rf(task: str, params: Optional[Dict[str, Any]] = None):
    """
    Build a RandomForest estimator for classification or regression.
    """
    defaults = dict(
        n_estimators=800,
        random_state=42,
        n_jobs=4,
        max_depth=None,
        min_samples_leaf=1,
    )

    if task == "classification":
        defaults["class_weight"] = "balanced"
        Estimator = RandomForestClassifier
    elif task == "regression":
        Estimator = RandomForestRegressor
    else:
        raise ValueError(f"Unsupported task: {task}")

    if params:
        defaults.update(params)

    return Estimator(**defaults)


def train_rf_cv(
    df_folds: pd.DataFrame,
    domain: str,
    run_id: str,
    target_spec: TargetSpec,
    fold_col: str,                         # Explicit column name (required since both 'fold' and 'outer_fold' exist)
    rf_params: Optional[Dict[str, Any]] = None,
    id_col: str = "id",
    elements_col: str = "elements_cif",
    group_col: str = "group_id",
    feature_prefix: str = "MagpieData",
    threshold: float = 0.5,
    leakage_check: bool = True,
    use_only_non_test: bool = True,
    is_test_col: str = "is_test",
    progress: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Train only from df_folds (contains features + folds).
    No filesystem, no project_config.
    """
    df = df_folds.copy()

    # Optional domain filter (belt & suspenders)
    if "chem_class" in df.columns:
        df = df[df["chem_class"].astype(str) == str(domain)].copy()

    # Target
    df, y_col = ml_utils.apply_target_spec(df, target_spec)

    # Features
    feature_cols = sorted([c for c in df.columns if c.startswith(feature_prefix)])
    if not feature_cols:
        raise ValueError(f"No feature columns with prefix '{feature_prefix}' found in df_folds.")
    print(f"Domain {domain}: {len(df)} samples, {len(feature_cols)} features")
    # Numeric validation
    rep = ml_utils.validate_numeric_features(df, feature_cols)
    if rep["non_numeric_all_nan"]:
        raise ValueError(
            f"All-NaN after numeric coercion for columns: {rep['non_numeric_all_nan'][:15]} "
            f"(total={len(rep['non_numeric_all_nan'])})"
        )

    # Estimator
    task = target_spec.task
    estimator = build_rf(task, rf_params)

    # Get task handler for threshold logic
    handler = get_task_handler(task)

    # CV
    oof_df, metrics_df, final_pipe = ml_utils.run_predefined_cv(
        df=df,
        estimator=estimator,
        feature_cols=feature_cols,
        y_col=y_col,
        fold_col=fold_col,
        id_col=id_col,
        elements_col=elements_col,
        group_col=group_col,
        threshold=threshold,
        leakage_check=leakage_check,
        use_only_non_test=use_only_non_test,
        is_test_col=is_test_col,
        task=task,
        progress=progress,
    )

    # Bundle
    bundle = ml_utils.ModelBundle(
        pipeline=final_pipe,
        feature_cols=feature_cols,
        target_spec=target_spec,
        domain=domain,
        run_id=run_id,
        fold_col=fold_col,
        id_col=id_col,
        elements_col=elements_col,
        group_col=group_col,
        params=rf_params,
        task=task,
        threshold=threshold if handler.requires_threshold else None,
        notes={"nan_fraction": rep["nan_fraction"], "use_only_non_test": use_only_non_test},
    )


    return {
        "df_used": df,              # after domain filter + target filter; before is_test filtering (CV does it)
        "oof_df": oof_df,
        "metrics_df": metrics_df,
        "bundle": bundle,
        "feature_cols": feature_cols,
        "y_col": y_col,
        "fold_col": fold_col,
        "validation_report": rep,
    }

def train_rf_train_test(
    df_folds: pd.DataFrame,
    domain: str,
    run_id: str,
    target_spec: TargetSpec,
    rf_params: Optional[Dict[str, Any]] = None,
    id_col: str = "id",
    elements_col: str = "elements_cif",
    group_col: str = "group_id",
    is_test_col: str = "is_test",
    feature_prefix: str = "MagpieData",
    threshold: float = 0.5,
    progress: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Train RF on TRAIN_VAL (is_test==0) and evaluate once on TEST (is_test==1).
    No folds, no CV.

    Returns dict with:
      - df_train
      - df_test
      - test_metrics
      - test_pred_df
      - bundle
      - feature_cols
      - y_col
    """

    df = df_folds.copy()

    # Optional domain filter
    if "chem_class" in df.columns:
        df = df[df["chem_class"].astype(str) == str(domain)].copy()

    if is_test_col not in df.columns:
        raise KeyError(f"is_test_col='{is_test_col}' not found in df_folds.")

    is_test = df[is_test_col].astype(str).str.lower().isin(["1", "true", "yes"])

    df_train = df[~is_test].copy()
    df_test  = df[is_test].copy()

    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError("TRAIN or TEST split is empty.")

    # Target
    df_train, y_col = ml_utils.apply_target_spec(df_train, target_spec)
    df_test,  _     = ml_utils.apply_target_spec(df_test,  target_spec)

    # Features
    feature_cols = sorted([c for c in df_train.columns if c.startswith(feature_prefix)])
    if not feature_cols:
        raise ValueError(f"No feature columns with prefix '{feature_prefix}' found.")

    # Numeric validation (train only)
    rep = ml_utils.validate_numeric_features(df_train, feature_cols)
    if rep["non_numeric_all_nan"]:
        raise ValueError(
            f"All-NaN after numeric coercion for columns: {rep['non_numeric_all_nan'][:15]} "
            f"(total={len(rep['non_numeric_all_nan'])})"
        )

    # Model
    task = target_spec.task
    estimator = build_rf(task, rf_params)

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])

    X_tr = ml_utils.coerce_numeric(df_train, feature_cols)
    X_te = ml_utils.coerce_numeric(df_test, feature_cols)

    # Get task-specific handler
    handler = get_task_handler(task)

    # Prepare targets
    y_tr = handler.prepare_y(df_train[y_col])
    y_te = handler.prepare_y(df_test[y_col])

    # Train
    pipe = ml_utils.fit_or_resume(pipe, X_tr, y_tr, progress=progress, params=rf_params)

    # Test predictions
    predictions = handler.predict(pipe, X_te)
    test_metrics = handler.compute_metrics(y_te, predictions, threshold=threshold)
    if progress and progress.get("checkpoint_path"):
        print(
            f"[FIT done] experiment={progress.get('experiment_id')} stage=final "
            f"status=ok auc={test_metrics.get('auc')} checkpoint_saved={progress.get('checkpoint_path')}"
        )

    # Test predictions DataFrame
    test_pred_df = handler.create_pred_df(
        df=df_test,
        y_true=y_te,
        predictions=predictions,
        id_col=id_col,
        elements_col=elements_col,
        group_col=group_col,
        y_col=y_col,
        threshold=threshold,
    )


    # Bundle

    bundle = ml_utils.ModelBundle(
        pipeline=pipe,
        feature_cols=feature_cols,
        target_spec=target_spec,
        domain=domain,
        run_id=run_id,
        fold_col="NONE",
        id_col=id_col,
        elements_col=elements_col,
        group_col=group_col,
        params=rf_params,
        task=task,
        threshold=threshold if handler.requires_threshold else None,
        notes={
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "threshold": threshold if handler.requires_threshold else None,
        },
    )

    return {
        "df_train": df_train,
        "df_test": df_test,
        "test_metrics": test_metrics,
        "test_pred_df": test_pred_df,
        "bundle": bundle,
        "feature_cols": feature_cols,
        "y_col": y_col,

    }
