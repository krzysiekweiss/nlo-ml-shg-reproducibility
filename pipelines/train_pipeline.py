"""Execute cross-validation and final model fitting with selected parameters."""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd

from core.rf_train import train_rf_cv, train_rf_train_test
from core.lgbm_train import train_lgbm_cv, train_lgbm_train_test
from core.task_handlers import get_task_handler
from .run_artifacts import save_run_artifacts

from targets.preset import make_target_spec


FeatureStep = str     # e.g. "magpie", "simple", "frac"
ModelName = str       # e.g. "rf", "lgbm"
ModeName = str        # "cv" or "final"


@dataclass(frozen=True)
class TrainPipelineConfig:
    """
    Training orchestrator configuration.
    """
    # Data columns
    fold_col: str = "fold"
    outer_fold_col: str = "outer_fold"   # optional if present; not required by RF core
    is_test_col: str = "is_test"
    group_col: str = "group_id"
    label_col: str = "centrosymmetry"
    magpie_prefix: str = "MagpieData"

    # Domains / filters
    domains: Tuple[str, str, str] = ("N", "O", "M")

    # Behavior
    verbose: bool = True
    verbose_batches: bool = False   # per-run detailed prints
    verbose_summary: bool = True    # end-of-run summary table
    # Output
    save_artifacts: bool = True


def _normalize_list(x: Optional[Union[str, Iterable[str]]], default: List[str]) -> List[str]:
    if x is None:
        return default
    if isinstance(x, str):
        return [x]
    return list(x)


def _select_folds_template_from_cfg(cfg, step: FeatureStep) -> str:
    """
    Return folds template path for a given feature step.
    Template must contain '{domain}' placeholder, e.g. ..._{domain}.csv
    """
    s = (step or "magpie").strip().lower()

    if s in ("magpie", "default"):
        return cfg.DATASET_FOLDS_TEMPLATE
    if s == "simple":
        return cfg.DATASET_FOLDS_TEMPLATE_S
    if s in ("frac", "fractional", "f"):
        return cfg.DATASET_FOLDS_TEMPLATE_F

    raise ValueError(f"Unknown feature step '{step}'. Use: magpie | simple | frac")


def _ensure_exists(path: str, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required artifact for {what}: {p}")
    return p


def _make_run_id(cfg, *, mode: ModeName, model: ModelName, step: FeatureStep, domain: str, user_run: Optional[str]) -> str:
    """
    Build a unique run_id string. If user_run is provided, it is used as prefix.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{cfg.PROJECT_NAME}__{mode}__{model}__{step}__{domain}__{ts}"
    return f"{user_run}__{base}" if user_run else base

def print_cv_report(
    metrics_df,
    *,
    model: str,
    domain: str,
    rename_map: dict | None = None,
):
    """
    Print and summarize CV metrics in a notebook-friendly way.
    """
    df_cv = metrics_df.copy()

    # Exclude aggregate row if present
    df_folds = df_cv[df_cv["fold"] != "ALL"].copy() if "fold" in df_cv.columns else df_cv

    if rename_map:
        df_disp = df_folds.rename(columns=rename_map)
    else:
        df_disp = df_folds

    # Display-like table (print fallback)
    print("\n=== CV per-fold metrics (display) ===")
    print(df_disp.to_string(index=False))

    print("\n=== CV raw metrics ===")
    print(df_cv.to_string(index=False))

    metric_cols = [c for c in ("auc", "auprc", "acc", "f1", "AUC", "AP", "ACC", "F1_N") if c in df_disp.columns]

    if metric_cols:
        print(f"\n=== CV summary ({model.upper()}, domain {domain}) ===")
        print(
            df_disp[metric_cols]
            .agg(["mean", "std"])
            .round(4)
            .to_string()
        )

def print_final_report(
    test_metrics: dict,
    *,
    model: str,
    domain: str,
):
    """
    Print a train/test (FINAL) evaluation report.
    """
    df_tt = (
        pd.DataFrame([test_metrics])
        .rename(columns={
            "n": "N",
            "auc": "AUC",
            "auprc": "AP",
            "acc": "ACC",
            "f1": "F1_N",
            "bacc": "BACC",
            "mcc": "MCC",
        })
    )

    print(f"\n=== FINAL TEST METRICS ({model.upper()}, domain {domain}) ===")
    print(df_tt.to_string(index=False))

def _extract_summary_row(run_id: str, step: str, domain: str, model: str, mode: str, out: dict) -> dict:
    """
    Build a single summary row for the global run summary table.
    """
    row = {
        "run_id": run_id,
        "step": step,
        "domain": domain,
        "model": model,
        "mode": mode,
    }

    # Time
    if "elapsed_sec" in out:
        row["time_sec"] = float(out["elapsed_sec"])

    # CV: take mean over folds (excluding ALL if present)
    if mode == "cv" and "metrics_df" in out:
        dfm = out["metrics_df"].copy()
        if "fold" in dfm.columns:
            dfm = dfm[dfm["fold"] != "ALL"].copy()

        # Prefer canonical metric column names if present
        for k in ("auc", "auprc", "acc", "f1", "bacc", "mcc"):
            if k in dfm.columns:
                row[k] = float(dfm[k].mean())

        # Also store std if available
        for k in ("auc", "auprc", "acc", "f1", "bacc", "mcc"):
            if k in dfm.columns:
                row[f"{k}_std"] = float(dfm[k].std())

    # FINAL: take test metrics dict
    if mode == "final" and "test_metrics" in out:
        tm = out["test_metrics"]
        for k in ("n", "auc", "auprc", "acc", "f1", "bacc", "mcc"):
            if k in tm:
                row[k] = float(tm[k]) if k != "n" else int(tm[k])

    return row


def print_global_summary(df: pd.DataFrame) -> None:
    """
    Print a compact global summary and a simple ranking.
    """
    if df.empty:
        print("\n=== GLOBAL SUMMARY ===")
        print("No results collected.")
        return

    print("\n=== GLOBAL SUMMARY (all runs) ===")
    cols = [c for c in ["mode", "model", "step", "domain", "auc", "auprc", "acc", "f1", "bacc", "mcc", "time_sec"] if c in df.columns]
    print(df[cols + ["run_id"]].sort_values(cols[:1] if cols else ["run_id"]).to_string(index=False))

    # Ranking by AUC if present
    if "auc" in df.columns:
        print("\n=== RANKING by AUC (desc) ===")
        rank_cols = [c for c in ["auc", "auprc", "acc", "f1", "time_sec"] if c in df.columns]
        print(
            df.sort_values("auc", ascending=False)[["mode", "model", "step", "domain"] + rank_cols + ["run_id"]]
            .head(30)
            .to_string(index=False)
        )

import json

def _load_params_registry(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Params registry not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _registry_lookup(reg: dict, *, model: str, step: str, domain: str, mode: str) -> Optional[dict]:
    key = {"model": model, "step": step, "domain": domain, "mode": mode}
    for e in reg.get("entries", []):
        if e.get("key") == key:
            return e.get("params")
    return None

def _merge_params(defaults: dict, reg_params: Optional[dict], override: Optional[dict]) -> dict:
    out = dict(defaults or {})
    if reg_params:
        out.update(reg_params)
    if override:
        out.update(override)
    return out

def _sanitize_rf_params(p: dict) -> dict:
    """
    Cast Hyperopt-style numeric params to sklearn-compatible types.
    """
    out = dict(p or {})
    for k in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"):
        if k in out and out[k] is not None:
            out[k] = int(out[k])
    return out

def _sanitize_lgbm_params(p: dict) -> dict:
    """Cast Hyperopt-style numeric params to LightGBM-compatible types."""
    out = dict(p or {})
    for k in ("num_leaves", "max_depth", "n_estimators", "min_child_samples"):
        if k in out and out[k] is not None:
            out[k] = int(out[k])
    # Ensure warnings stay off unless user overrides
    out.setdefault("verbosity", -1)
    return out


class TrainPipeline:
    """
    Orchestrates training runs across:
      - multiple feature steps
      - multiple domains
      - multiple models (rf, lgbm, ...)
    """

    def __init__(self, cfg, config: Optional[TrainPipelineConfig] = None):
        self.cfg = cfg
        self.c = config or TrainPipelineConfig()

    # public API
    def run(
        self,
        *,
        models: Optional[Union[ModelName, Iterable[ModelName]]] = None,
        steps: Optional[Union[FeatureStep, Iterable[FeatureStep]]] = None,
        domains: Optional[Union[str, Iterable[str]]] = None,
        mode: ModeName = "cv",  # "cv" or "final"
        run: Optional[str] = None,  # user-provided run prefix/id
        target_name: Optional[str] = "centrosymmetry_CN",
        rf_params: Optional[Dict] = None,
        lgbm_params: Optional[Dict] = None,
        params_registry_path: Optional[str] = None,
        use_registry: bool = True,
        series: Optional[str] = None,
        fingerprint_payload: Optional[dict] = None,
        resume: bool = True,
        force: bool = False,
        log_context: Optional[dict] = None,
    ) -> Dict[str, Dict]:
        """
        Run training for the given combinations.

        Returns dict keyed by run_id with training outputs (metrics, bundle, etc.).
        """
        from core.notebook_logger import NotebookLogger
        from core.log_timing import LogTiming
        from core.feature_utils import detect_feature_prefix
        import os

        model_list = _normalize_list(models, default=["rf"])
        step_list = _normalize_list(steps, default=["magpie"])
        domain_list = _normalize_list(domains, default=list(self.c.domains))

        registry = None
        if use_registry and params_registry_path:
            registry = _load_params_registry(params_registry_path)


        target_spec = make_target_spec(target_name)

        # Get task handler to determine metric display names
        handler = get_task_handler(target_spec.task)
        # Choose metric display names based on the task type
        if handler.task_name == "classification":
            metric_rename_map = {"auc": "AUC", "auprc": "AP", "acc": "ACC", "f1": "F1_N"}
        else:
            metric_rename_map = {"mae": "MAE", "rmse": "RMSE", "r2": "R2", "spearman": "SPEARMAN"}


        checkpoint_series = series or run
        logger = NotebookLogger(
            project_dir=self.cfg.PROJECT_DIR,
            prefix=f"{checkpoint_series}_{self.cfg.PROJECT_NAME}_{mode}" if checkpoint_series else self.cfg.PROJECT_NAME,
        )

        logger.start(header_lines=[
            f"Project        : {self.cfg.PROJECT_NAME}",
            'Pipeline       : TrainPipeline',
            f"CWD            : {os.getcwd()}",
            f"Mode           : {mode}",
            f"Models         : {model_list}",
            f"Steps          : {step_list}",
            f"Domains        : {domain_list}",
            f"Target spec    : {target_spec}",
            f"Series         : {checkpoint_series}",
            f"Resume         : {resume}",
            f"Force          : {force}",
        ])
        if log_context is not None:
            print("Configuration:")
            print(json.dumps(log_context, indent=2, sort_keys=True, default=str))
        print(
            "[MODEL] RF training builder defaults are class_weight='balanced' and n_jobs=4. "
            "LGBM training builder default is class_weight='balanced'. "
            "Tuning does not set class_weight. These defaults are unchanged."
        )

        results: Dict[str, Dict] = {}
        summary_rows = []

        try:
            import traceback
            from core.ml_utils import file_sha256, json_fingerprint, load_checkpoint
            for step in step_list:
                folds_template = _select_folds_template_from_cfg(self.cfg, step)

                for domain in domain_list:
                    folds_path = folds_template.format(domain=domain)
                    _ensure_exists(folds_path, what=f"folds CSV (step='{step}', domain='{domain}')")

                    df_folds = pd.read_csv(folds_path)
                    feature_prefix = detect_feature_prefix(df_folds, step)

                    for model in model_list:
                        run_id = _make_run_id(
                            self.cfg, mode=mode, model=model, step=step, domain=domain, user_run=run
                        )

                        if self.c.verbose:
                            print(f"\n=== RUN {run_id} ===")
                            print("Folds:", folds_path)
                        timing_label = f"train_{model}_{mode}__step={step}__domain={domain}"
                        registry = None
                        if use_registry and params_registry_path:
                            registry = _load_params_registry(params_registry_path)

                        # Registry lookup (tuning params are keyed by mode="cv" even when training final)
                        reg_params = None
                        if registry is not None:
                            reg_params = _registry_lookup(
                                registry,
                                model=model,
                                step=step,
                                domain=domain,
                                mode="cv",
                            )


                        if model == "rf":
                            rf_defaults = {}
                            rf_params_eff = _sanitize_rf_params(_merge_params(rf_defaults, reg_params, rf_params))

                            if reg_params and rf_params:
                                params_source = "registry+override"
                            elif reg_params:
                                params_source = "registry"
                            elif rf_params:
                                params_source = "manual_override"
                            else:
                                params_source = "default"

                        elif model == "lgbm":
                            lgbm_defaults = {"verbosity": -1}  # Set the LightGBM logging level.
                            lgbm_params_eff = _sanitize_lgbm_params(_merge_params(lgbm_defaults, reg_params, lgbm_params))

                            if reg_params and lgbm_params:
                                params_source = "registry+override"
                            elif reg_params:
                                params_source = "registry"
                            elif lgbm_params:
                                params_source = "manual_override"
                            else:
                                params_source = "default"

                        if use_registry and reg_params is None:
                            print(
                                f"[WARN] No tuned parameters for {model} {step} {domain} "
                                f"in {params_registry_path}. Training uses builder defaults."
                            )

                        data_sha = file_sha256(folds_path)
                        print("Data sha256:", data_sha)
                        progress = None
                        if checkpoint_series:
                            payload = dict(fingerprint_payload or {})
                            active_params = rf_params_eff if model == "rf" else lgbm_params_eff
                            payload.update({
                                "series": checkpoint_series,
                                "project": self.cfg.PROJECT_NAME,
                                "domain": domain,
                                "model": model,
                                "step": step,
                                "stage": mode,
                                "data_sha256": data_sha,
                                "params": active_params,
                                "target_name": target_name,
                            })
                            checkpoint_path = (
                                Path(self.cfg.PROJECT_DIR) / "tuning" / "checkpoints"
                                / f"{checkpoint_series}__{self.cfg.PROJECT_NAME}__{mode}__{model}__{step}__{domain}.pkl"
                            )
                            progress = {
                                "experiment_id": f"{checkpoint_series}__{self.cfg.PROJECT_NAME}__{mode}__{model}__{step}__{domain}",
                                "checkpoint_path": str(checkpoint_path),
                                "fingerprint": json_fingerprint(payload),
                                "resume": resume,
                                "force": force,
                                "params": active_params,
                            }
                            print("Fingerprint:", progress["fingerprint"])
                            print("[MODEL] params passed into the training builder:", active_params)
                            if not force:
                                loaded = load_checkpoint(
                                    progress["checkpoint_path"],
                                    progress["fingerprint"],
                                    resume=True,
                                )
                                if loaded and loaded.get("finished"):
                                    print(
                                        f"[SKIP] {model} {step} {domain} {mode}: "
                                        f"result for run {checkpoint_series} already exists."
                                    )
                                    print("Checkpoint:", progress["checkpoint_path"])
                                    continue

                        with LogTiming(timing_label, enabled=getattr(self.cfg, "LOG_TIME", False)):
                            if model == "rf":
                                if mode == "cv":
                                    out = train_rf_cv(
                                        df_folds=df_folds,
                                        domain=domain,
                                        run_id=run_id,
                                        target_spec=target_spec,
                                        fold_col=self.c.fold_col,
                                        rf_params=rf_params_eff,
                                        group_col=self.c.group_col,
                                        feature_prefix=feature_prefix,
                                        is_test_col=self.c.is_test_col,
                                        progress=progress,
                                    )
                                elif mode == "final":
                                    out = train_rf_train_test(
                                        df_folds=df_folds,
                                        domain=domain,
                                        run_id=run_id,
                                        target_spec=target_spec,
                                        rf_params=rf_params_eff,
                                        group_col=self.c.group_col,
                                        feature_prefix=feature_prefix,
                                        is_test_col=self.c.is_test_col,
                                        progress=progress,
                                    )
                                else:
                                    raise ValueError("mode must be 'cv' or 'final'")

                            elif model == "lgbm":
                                if mode == "cv":
                                    out = train_lgbm_cv(
                                        df_folds=df_folds,
                                        domain=domain,
                                        run_id=run_id,
                                        target_spec=target_spec,
                                        fold_col=self.c.fold_col,
                                        lgbm_params=lgbm_params_eff,
                                        group_col=self.c.group_col,
                                        feature_prefix=feature_prefix,
                                        is_test_col=self.c.is_test_col,
                                        progress=progress,
                                    )
                                elif mode == "final":
                                    out = train_lgbm_train_test(
                                        df_folds=df_folds,
                                        domain=domain,
                                        run_id=run_id,
                                        target_spec=target_spec,
                                        lgbm_params=lgbm_params_eff,
                                        group_col=self.c.group_col,
                                        feature_prefix=feature_prefix,
                                        is_test_col=self.c.is_test_col,
                                        progress=progress,
                                    )

                            else:
                                raise ValueError(f"Unknown model '{model}'. Use: rf | lgbm")

                        results[run_id] = out
                        if model == "rf":
                            out["params_used"] = rf_params_eff
                        else:
                            out["params_used"] = lgbm_params_eff

                        out["params_source"] = params_source
                        out["tuning_key"] = {"model": model, "step": step, "domain": domain, "mode": "cv"}

                        if self.c.save_artifacts:
                            paths = save_run_artifacts(
                                cfg=self.cfg,
                                run_id=run_id,
                                mode=mode,
                                model=model,
                                step=step,
                                domain=domain,
                                out=out,
                            )
                            out["artifact_paths"] = paths


                        if self.c.verbose_batches:
                            if mode == "cv" and "metrics_df" in out:
                                print_cv_report(
                                    out["metrics_df"],
                                    model=model,
                                    domain=domain,
                                    rename_map=metric_rename_map,
                                )
                            if mode == "final" and "test_metrics" in out:
                                print_final_report(
                                    out["test_metrics"],
                                    model=model,
                                    domain=domain,
                                )

                        summary_rows.append(_extract_summary_row(run_id, step, domain, model, mode, out))
                        # Optional: persist artifacts here (metrics CSV, bundle pickle, predictions)
                        # Write artifacts under the configured project directory.

            summary_df = pd.DataFrame(summary_rows)

            if self.c.verbose_summary:
                print_global_summary(summary_df)

        except BaseException:
            traceback.print_exc()
            raise
        finally:
            logger.stop()

        return results
