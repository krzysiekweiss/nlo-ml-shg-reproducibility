"""Prepare targets and features, compute metrics and manage model artifacts."""

from __future__ import annotations
import os
import re
import json
import pickle
import time
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from scipy.stats import spearmanr
import joblib
from tqdm import tqdm
from .task_handlers import get_task_handler


# Target handling (target_spec)


def apply_target_spec(df: pd.DataFrame, spec: Any) -> Tuple[pd.DataFrame, str]:
    """
    Materialize the target column according to the provided spec.

    Supported inputs:
    - dictionary specification, for example:
        {
          "source_col": str,
          "mapping": dict | None,            # {"C":0,"N":1}
          "drop_values": list | None,        # ["U"]
          "out_col": str,                    # "y"
          "task": "classification"|"regression"  # optional
        }
    - TargetSpec-like object with attributes:
        source_col, out_col, task, mapping, drop_values (optional)

    Returns
    -------
    (df_filtered, y_col_name)
    """
    # Resolve spec fields for both dict and object styles
    if isinstance(spec, dict):
        src = spec["source_col"]
        out = spec.get("out_col", "y")
        mapping = spec.get("mapping", None)
        drop_vals = spec.get("drop_values", None)
        task = spec.get("task", "classification")
    else:
        src = getattr(spec, "source_col")
        out = getattr(spec, "out_col", "y")
        mapping = getattr(spec, "mapping", None)
        drop_vals = getattr(spec, "drop_values", None)
        task = getattr(spec, "task", "classification")

    task = str(task).lower().strip()
    if task not in {"classification", "regression"}:
        raise ValueError("spec.task must be either 'classification' or 'regression'")

    if src not in df.columns:
        raise KeyError(f"Target source column '{src}' not found in df columns.")

    d = df.copy()

    # Optional filtering of specific raw target values (mostly useful for classification)
    if drop_vals is not None:
        drop_vals_norm = set(str(v).strip() for v in drop_vals)
        raw = d[src].astype(str).str.strip()
        d = d[~raw.isin(drop_vals_norm)].copy()

    if task == "classification":
        raw = d[src].astype(str).str.strip()

        if mapping is None:
            raise ValueError("Classification target requires a 'mapping' in spec.")

        d[out] = raw.map(mapping)
        d = d[d[out].notna()].copy()
        d[out] = d[out].astype(int)
        return d, out

    # Regression
    d[out] = pd.to_numeric(d[src], errors="coerce")
    d = d[d[out].notna()].copy()
    d[out] = d[out].astype(float)
    return d, out


# Feature selection / numeric validation


def get_feature_cols_by_prefix(df: pd.DataFrame, prefix: str = "MagpieData") -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns starting with '{prefix}' found.")
    return sorted(cols)

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def validate_numeric_features(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    non_numeric_all_nan = []
    nan_fraction = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        nan_fraction[c] = float(s.isna().mean())
        if s.notna().sum() == 0:
            non_numeric_all_nan.append(c)
    return {"non_numeric_all_nan": non_numeric_all_nan, "nan_fraction": nan_fraction}


# Leakage check (optional; folds should already be OK)


def assert_no_group_leakage(df: pd.DataFrame, fold_col: str, group_col: str = "group_id") -> None:
    if group_col not in df.columns:
        return
    folds = sorted(df[fold_col].dropna().unique().tolist())
    for f in folds:
        tr_groups = set(df.loc[df[fold_col] != f, group_col].dropna().unique())
        va_groups = set(df.loc[df[fold_col] == f, group_col].dropna().unique())
        inter = tr_groups.intersection(va_groups)
        if inter:
            raise AssertionError(f"Group leakage detected in fold={f}. Example: {list(inter)[:10]}")


# Metrics


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, Any] = {"n": int(len(y_true))}
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
        out["bacc"] = float(balanced_accuracy_score(y_true, y_pred))
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    else:
        out["auc"] = np.nan
        out["auprc"] = np.nan
        out["bacc"] = np.nan
        out["mcc"] = np.nan

    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return out

def compute_regression_metrics(y_true, y_pred) -> dict:
    """
    Compute regression metrics for model evaluation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    rho = spearmanr(y_true, y_pred).correlation
    spearman = float(rho) if rho is not None else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman}


# CV on predefined folds + optional is_test filter


def atomic_pickle(path: str | os.PathLike, payload: Any) -> None:
    """Atomically replace ``path`` with a pickle of ``payload``."""
    destination = os.fspath(path)
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = destination + ".tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def atomic_text(path: str | os.PathLike, text: str) -> None:
    """Atomically replace a text file."""
    destination = os.fspath(path)
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = destination + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(payload: Dict[str, Any]) -> str:
    """Stable hash of a JSON-compatible configuration."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_checkpoint(path: str | os.PathLike, fingerprint: str, *, resume: bool, force: bool = False) -> Optional[dict]:
    """Load a checkpoint, or return None when resume is off and no file exists.

    ``force`` ignores an existing checkpoint so the caller can overwrite it.
    A fingerprint mismatch never continues an old state.
    """
    destination = os.fspath(path)
    if not os.path.exists(destination):
        return None
    if force:
        print(f"[FORCE] Existing checkpoint will be overwritten: {destination}")
        return None
    if not resume:
        raise RuntimeError(
            f"Checkpoint already exists at {destination}. "
            "Refusing to ignore it while resume is false."
        )
    with open(destination, "rb") as handle:
        state = pickle.load(handle)
    saved = state.get("fingerprint")
    if saved != fingerprint:
        raise RuntimeError(
            "Checkpoint configuration does not match the current experiment. "
            f"Saved fingerprint {saved}, current fingerprint {fingerprint}. "
            f"File: {destination}. It was not modified."
        )
    return state


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fit_or_resume(pipe, x_train, y_train, *, progress: Optional[dict], params: Optional[dict]):
    """Fit once, or reload a completed final fit from an atomic checkpoint."""
    experiment_id = None if not progress else progress.get("experiment_id")
    checkpoint_path = None if not progress else progress.get("checkpoint_path")
    fingerprint = None if not progress else progress.get("fingerprint")
    if checkpoint_path and fingerprint:
        loaded = load_checkpoint(
            checkpoint_path, fingerprint,
            resume=bool(progress.get("resume", True)),
            force=bool(progress.get("force", False)),
        )
        if loaded and loaded.get("kind") == "final" and loaded.get("finished") and loaded.get("pipe") is not None:
            print(
                f"[FIT done] experiment={experiment_id} stage=final status=skipped_completed "
                f"auc={loaded.get('auc')} checkpoint={checkpoint_path}"
            )
            return loaded["pipe"]
    print(f"[FIT start] experiment={experiment_id} stage=final params={params}")
    started = utc_now()
    started_counter = time.perf_counter()
    pipe.fit(x_train, y_train)
    elapsed = time.perf_counter() - started_counter
    if checkpoint_path and fingerprint:
        atomic_pickle(checkpoint_path, {
            "kind": "final",
            "fingerprint": fingerprint,
            "pipe": pipe,
            "finished": True,
            "started": started,
            "elapsed_seconds": elapsed,
        })
    print(
        f"[FIT done] experiment={experiment_id} stage=final status=ok "
        f"started={started} elapsed_seconds={elapsed:.3f} checkpoint_saved={checkpoint_path}"
    )
    return pipe


def run_predefined_cv(
    df: pd.DataFrame,
    estimator,
    feature_cols: List[str],
    y_col: str,
    fold_col: str,
    id_col: str = "id",
    elements_col: str = "elements_cif",
    group_col: str = "group_id",
    threshold: float = 0.5,
    leakage_check: bool = True,
    imputer_strategy: str = "median",
    use_only_non_test: bool = True,
    is_test_col: str = "is_test",
    task: str = "classification",
    progress: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    """
    Run CV on predefined folds with optional filtering of is_test rows.

    Parameters
    ----------
    task : str
        Either "classification" or "regression".

    Returns
    -------
    oof_df : pd.DataFrame
        Out-of-fold predictions for CV rows.
    metrics_df : pd.DataFrame
        Per-fold metrics plus an "ALL" row computed on OOF predictions.
    final_pipe : Pipeline
        Pipeline fit on the same rows used for CV (or all rows if use_only_non_test=False).
    """
    d = df.copy()

    if use_only_non_test and is_test_col in d.columns:
        # Accept 0/1 or True/False strings
        mask = ~d[is_test_col].astype(str).str.lower().isin(["1", "true", "yes"])
        d = d[mask].copy()

    if fold_col not in d.columns:
        raise KeyError(f"fold_col='{fold_col}' not found in dataframe.")

    if leakage_check:
        assert_no_group_leakage(d, fold_col=fold_col, group_col=group_col)

    task = str(task).lower().strip()
    if task not in {"classification", "regression"}:
        raise ValueError("task must be either 'classification' or 'regression'")

    # Get task-specific handler
    handler = get_task_handler(task)

    base_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("model", estimator),
    ])

    folds = sorted(d[fold_col].dropna().unique().tolist())
    oof_rows: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    saved_folds: Dict[str, Dict[str, Any]] = {}
    checkpoint_path = None
    fingerprint = None
    experiment_id = None
    model_params = None
    if progress:
        checkpoint_path = progress.get("checkpoint_path")
        fingerprint = progress.get("fingerprint")
        experiment_id = progress.get("experiment_id")
        model_params = progress.get("params")
        if checkpoint_path and fingerprint:
            loaded = load_checkpoint(
                checkpoint_path, fingerprint,
                resume=bool(progress.get("resume", True)),
                force=bool(progress.get("force", False)),
            )
            if loaded and loaded.get("kind") == "cv":
                saved_folds = dict(loaded.get("folds") or {})
                print(
                    f"[RESUME] experiment={experiment_id} stage=cv "
                    f"completed_folds={sorted(saved_folds)} checkpoint={checkpoint_path}"
                )

    def _save_cv(refit_done: bool, final_pipe_obj) -> None:
        if not checkpoint_path:
            return
        atomic_pickle(checkpoint_path, {
            "kind": "cv",
            "fingerprint": fingerprint,
            "folds": saved_folds,
            "refit_done": refit_done,
            "final_pipe": final_pipe_obj,
            "finished": refit_done,
        })

    for f in tqdm(folds, desc=f"CV ({fold_col})", total=len(folds)):
        fold_key = str(f)
        if fold_key in saved_folds:
            cached = saved_folds[fold_key]
            oof_rows.extend(cached["oof_rows"])
            fold_metrics.append(cached["metrics"])
            print(
                f"[FIT done] experiment={experiment_id} stage=cv fold={f} "
                f"status=skipped_completed auc={cached['metrics'].get('auc')} "
                f"checkpoint={checkpoint_path}"
            )
            continue

        tr = d[d[fold_col] != f].copy()
        va = d[d[fold_col] == f].copy()

        X_tr = coerce_numeric(tr, feature_cols)
        X_va = coerce_numeric(va, feature_cols)

        # Prepare targets using handler
        y_tr = handler.prepare_y(tr[y_col])
        y_va = handler.prepare_y(va[y_col])

        print(
            f"[FIT start] experiment={experiment_id} stage=cv fold={f} "
            f"trial={len(saved_folds) + 1}/{len(folds)} params={model_params}"
        )
        started = utc_now()
        started_counter = time.perf_counter()
        pipe = clone(base_pipe)
        pipe.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - started_counter

        # Get predictions using handler
        predictions = handler.predict(pipe, X_va)

        # Create OOF rows using handler
        fold_oof_start = len(oof_rows)
        for i, idx in enumerate(va.index):
            row = {
                id_col: va.at[idx, id_col] if id_col in va.columns else None,
                "filepath": va.at[idx, "filepath"] if "filepath" in va.columns else None,
                "chem_class": va.at[idx, "chem_class"] if "chem_class" in va.columns else None,
                fold_col: f,
                y_col: handler.prepare_y(va.loc[[idx], y_col])[0],
                elements_col: va.at[idx, elements_col] if elements_col in va.columns else None,
                group_col: va.at[idx, group_col] if group_col in va.columns else None,
            }
            # Add task-specific prediction columns
            if "y_prob" in predictions:
                row["y_prob"] = float(predictions["y_prob"][i])
                row["y_pred"] = int(predictions["y_pred"][i])
            else:
                row["y_pred"] = float(predictions["y_pred"][i])
            oof_rows.append(row)

        # Compute metrics using handler
        m = handler.compute_metrics(y_va, predictions, threshold=threshold)
        m["fold"] = f
        fold_metrics.append(m)
        saved_folds[fold_key] = {
            "oof_rows": oof_rows[fold_oof_start:],
            "metrics": dict(m),
            "started": started,
            "elapsed_seconds": elapsed,
            "status": "ok",
        }
        _save_cv(False, None)
        finished_scores = [item["metrics"].get("auc") for item in saved_folds.values()]
        finished_scores = [score for score in finished_scores if score is not None and score == score]
        best_auc = max(finished_scores) if finished_scores else None
        print(
            f"[FIT done] experiment={experiment_id} stage=cv fold={f} status=ok "
            f"auc={m.get('auc')} started={started} elapsed_seconds={elapsed:.3f} "
            f"best_auc={best_auc} checkpoint_saved={checkpoint_path}"
        )

    oof_df = pd.DataFrame(oof_rows)
    metrics_df = pd.DataFrame(fold_metrics)

    # Compute global metrics
    if "y_prob" in oof_df.columns:
        global_m = compute_metrics(oof_df[y_col].values, oof_df["y_prob"].values, threshold=threshold)
    else:
        global_m = compute_regression_metrics(oof_df[y_col].values, oof_df["y_pred"].values)

    global_m["fold"] = "ALL"
    metrics_df = pd.concat([metrics_df, pd.DataFrame([global_m])], ignore_index=True)

    # Fit final pipeline on the same data used for CV.
    loaded_refit = None
    if checkpoint_path and fingerprint:
        loaded_state = load_checkpoint(
            checkpoint_path, fingerprint,
            resume=bool(progress.get("resume", True)),
            force=bool(progress.get("force", False)),
        ) if progress else None
        if loaded_state and loaded_state.get("refit_done") and loaded_state.get("final_pipe") is not None:
            loaded_refit = loaded_state["final_pipe"]
    if loaded_refit is not None:
        final_pipe = loaded_refit
        print(
            f"[FIT done] experiment={experiment_id} stage=cv_refit status=skipped_completed "
            f"checkpoint={checkpoint_path}"
        )
    else:
        X_all = coerce_numeric(d, feature_cols)
        y_all = handler.prepare_y(d[y_col])
        print(
            f"[FIT start] experiment={experiment_id} stage=cv_refit params={model_params}"
        )
        refit_started = utc_now()
        refit_counter = time.perf_counter()
        final_pipe = clone(base_pipe)
        final_pipe.fit(X_all, y_all)
        refit_elapsed = time.perf_counter() - refit_counter
        _save_cv(True, final_pipe)
        print(
            f"[FIT done] experiment={experiment_id} stage=cv_refit status=ok "
            f"started={refit_started} elapsed_seconds={refit_elapsed:.3f} "
            f"checkpoint_saved={checkpoint_path}"
        )

    return oof_df, metrics_df, final_pipe


# Bundle + IO helpers (still no config dependency)


@dataclass
class ModelBundle:
    """
    Container for a trained model pipeline and its metadata.
    """
    pipeline: Any
    feature_cols: List[str]
    target_spec: Dict[str, Any]
    domain: str
    run_id: str
    fold_col: str
    id_col: str
    elements_col: str
    group_col: str
    params: Optional[Dict[str, Any]] = None
    task: str = "classification"
    threshold: Optional[float] = None
    notes: Optional[Dict[str, Any]] = None

def save_bundle(path: str, bundle: ModelBundle) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)

def load_bundle(path: str) -> ModelBundle:
    return joblib.load(path)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def write_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# Prediction on new data


def predict_df(df_new: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    missing = [c for c in bundle.feature_cols if c not in df_new.columns]
    if missing:
        raise KeyError(f"New data missing {len(missing)} feature columns. Example: {missing[:10]}")

    X = coerce_numeric(df_new, bundle.feature_cols)

    # Get task-specific handler
    handler = get_task_handler(bundle.task)

    # Get predictions
    predictions = handler.predict(bundle.pipeline, X)

    # Create prediction DataFrame using handler
    # Note: we don't have y_true here, so we create a dummy array
    dummy_y = np.zeros(len(df_new))
    threshold = bundle.threshold if bundle.threshold is not None else 0.5

    out = handler.create_pred_df(
        df=df_new,
        y_true=dummy_y,
        predictions=predictions,
        id_col=bundle.id_col,
        elements_col=bundle.elements_col,
        group_col=bundle.group_col,
        y_col="y",  # dummy column name
        threshold=threshold,
    )

    # Remove the dummy y column if it was added
    if "y" in out.columns:
        out = out.drop(columns=["y"])

    return out


# Reporting: per element


_SPLIT_RE = re.compile(r"[,\s;]+")

def explode_elements(df: pd.DataFrame, elements_col: str = "elements_cif") -> pd.DataFrame:
    tmp = df.copy()
    tmp[elements_col] = tmp[elements_col].fillna("").astype(str)
    tmp["_elements_list"] = tmp[elements_col].apply(lambda s: [e for e in _SPLIT_RE.split(s.strip()) if e])
    long = tmp.explode("_elements_list", ignore_index=True).rename(columns={"_elements_list": "element"})
    return long[(long["element"].notna()) & (long["element"] != "")]

def report_by_element(
    df_pred: pd.DataFrame,
    elements_col: str = "elements_cif",
    y_true_col: Optional[str] = None,
    y_prob_col: str = "y_prob",
    min_n: int = 30,
    threshold: float = 0.5,
) -> pd.DataFrame:
    long = explode_elements(df_pred, elements_col=elements_col)

    rows: List[Dict[str, Any]] = []
    for el, g in long.groupby("element"):
        n = len(g)
        if n < min_n:
            continue

        row: Dict[str, Any] = {"element": el, "n": int(n)}
        if y_true_col is not None and y_true_col in g.columns:
            m = compute_metrics(g[y_true_col].values, g[y_prob_col].values, threshold=threshold)
            row.update({k: m[k] for k in ["auc", "auprc", "bacc", "mcc", "acc", "f1"]})
        else:
            row.update({
                "y_prob_mean": float(g[y_prob_col].mean()),
                "y_prob_median": float(g[y_prob_col].median()),
                "y_prob_p90": float(g[y_prob_col].quantile(0.90)),
            })

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("n", ascending=False).reset_index(drop=True)
    return out
