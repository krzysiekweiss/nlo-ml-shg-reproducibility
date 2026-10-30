"""Run and checkpoint domain-specific hyperparameter searches."""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Union
import json
import traceback

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from core.log_timing import LogTiming
import core.ml_utils as mlu


FeatureStep = str
ModelName = str
ModeName = str


@dataclass(frozen=True)
class HyperTuningPipelineConfig:
    """
    Hypertuning orchestrator config.
    """
    # Columns
    fold_col: str = "fold"          # Column identifying the predefined validation fold.
    is_test_col: str = "is_test"
    domain_col: str = "chem_class"

    # Domains
    domains: Tuple[str, str, str] = ("N", "O", "M")

    # Tuning defaults
    selected_fold: int = 0
    max_evals: int = 50
    metric: str = "auc"             # "auc" | "auprc" | "acc"
    random_state: int = 42

    # Progress
    show_progress: bool = True      # job-level tqdm + trial-level tqdm (if implemented in tuner)

    # Output
    save_trials: bool = True
    save_registry: bool = True
    registry_filename: str = "best_params.json"


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


def _make_tune_run_id(cfg, *, user_run: Optional[str]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{cfg.PROJECT_NAME}__TUNE__{ts}"
    return f"{user_run}__{base}" if user_run else base


def _registry_path(cfg, filename: str) -> Path:
    # Keep registry next to results, or under project dir tuning/
    tuning_dir = Path(cfg.PROJECT_DIR) / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    return tuning_dir / filename


def _load_registry(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "project": None,
        "primary_metric": None,
        "entries": [],
    }


def _upsert_registry_entry(reg: dict, *, key: dict, params: dict, tuning_meta: dict) -> None:
    """
    Upsert by key equality.
    """
    entries = reg.setdefault("entries", [])
    for e in entries:
        if e.get("key") == key:
            e["params"] = params
            e["tuning"] = tuning_meta
            return
    entries.append({"key": key, "params": params, "tuning": tuning_meta})


def _registry_params(params: dict) -> dict:
    """Keep searched parameters. Fixed estimator settings stay in the training builders."""
    skipped = {"n_jobs", "verbosity", "class_weight", "random_state"}
    return {key: value for key, value in params.items() if key not in skipped}


def _atomic_json(path: Path, payload: dict) -> None:
    from core.ml_utils import atomic_text
    atomic_text(path, json.dumps(payload, indent=2, default=str))


class HyperTuningPipeline:
    """
    Orchestrates hypertuning across (model x step x domain), saves:
      - trials CSV per job
      - best_params.json registry
    """

    def __init__(self, cfg, config: Optional[HyperTuningPipelineConfig] = None):
        self.cfg = cfg
        self.c = config or HyperTuningPipelineConfig()

    def run(
        self,
        *,
        models: Optional[Union[ModelName, Iterable[ModelName]]] = None,
        steps: Optional[Union[FeatureStep, Iterable[FeatureStep]]] = None,
        domains: Optional[Union[str, Iterable[str]]] = None,
        mode: ModeName = "cv",  # tuning key, typically "cv"
        run: Optional[str] = None,  # user prefix for tuning run id
        target_spec: Optional[dict] = None,
        search_space: Optional[dict] = None,
        fixed_params: Optional[dict] = None,
        n_startup_jobs: Optional[int] = None,
        series: Optional[str] = None,
        fingerprint_payload: Optional[dict] = None,
        resume: bool = True,
        force: bool = False,
        log_context: Optional[dict] = None,
    ) -> dict:
        from core.notebook_logger import NotebookLogger
        from core.feature_utils import detect_feature_prefix

        import os

        model_list = _normalize_list(models, default=["rf"])
        step_list = _normalize_list(steps, default=["magpie"])
        domain_list = _normalize_list(domains, default=list(self.c.domains))

        # Use the centrosymmetry target mapping when no specification is supplied.

        if target_spec is None:
            target_spec = {
                "source_col": "centrosymmetry",
                "out_col": "y",
                "mapping": {"C": 0, "N": 1},
                "allow_values": ["C", "N"],
            }

        if series:
            tune_run_id = f"{series}__{self.cfg.PROJECT_NAME}__TUNE"
        elif run:
            tune_run_id = f"{run}__{self.cfg.PROJECT_NAME}__TUNE"
        else:
            tune_run_id = _make_tune_run_id(self.cfg, user_run=None)

        run_label = series or run
        log_prefix = f"{run_label}_{self.cfg.PROJECT_NAME}_tune" if run_label else self.cfg.PROJECT_NAME
        logger = NotebookLogger(project_dir=self.cfg.PROJECT_DIR, prefix=log_prefix)
        logger.start(header_lines=[
            f"Project        : {self.cfg.PROJECT_NAME}",
            'Pipeline       : HyperTuningPipeline',
            f"Tune run id    : {tune_run_id}",
            f"CWD            : {os.getcwd()}",
            f"Mode key       : {mode}",
            f"Models         : {model_list}",
            f"Steps          : {step_list}",
            f"Domains        : {domain_list}",
            f"Metric         : {self.c.metric}",
            f"Max evals      : {self.c.max_evals}",
            f"Selected fold  : {self.c.selected_fold}",
            f"Run            : {run_label}",
            f"Startup trials : {n_startup_jobs}",
            f"Resume         : {resume}",
            f"Force          : {force}",
        ])
        if log_context is not None:
            print("Configuration:")
            print(json.dumps(log_context, indent=2, sort_keys=True, default=str))

        results: dict = {
            "tune_run_id": tune_run_id,
            "jobs": [],
        }

        registry_filename = self.c.registry_filename
        if registry_filename == "best_params.json" and run_label:
            registry_filename = f"best_params_{run_label}.json"
        reg_path = _registry_path(self.cfg, registry_filename)
        registry = _load_registry(reg_path)
        registry["project"] = self.cfg.PROJECT_NAME
        registry["primary_metric"] = self.c.metric
        registry_dirty = False

        try:
            jobs = [(m, s, d) for m in model_list for s in step_list for d in domain_list]
            job_iter = tqdm(jobs, desc="Hypertuning jobs", disable=not self.c.show_progress)

            for model, step, domain in job_iter:
                folds_template = _select_folds_template_from_cfg(self.cfg, step)
                folds_path = folds_template.format(domain=domain)
                _ensure_exists(folds_path, what=f"folds CSV for tuning (step='{step}', domain='{domain}')")

                from core.ml_utils import file_sha256, json_fingerprint
                from core.hyperopt_tuning import LGBM_PAPER_V2_SPACE, RF_PAPER_V2_SPACE
                data_sha = file_sha256(folds_path)
                space = search_space if search_space is not None else (
                    RF_PAPER_V2_SPACE if model == "rf" else LGBM_PAPER_V2_SPACE
                )
                if fixed_params is not None:
                    fixed = fixed_params
                elif model == "lgbm":
                    fixed = {"n_jobs": -1, "random_state": self.c.random_state, "verbosity": -1}
                else:
                    fixed = {"n_jobs": -1, "random_state": self.c.random_state}
                startup = 20 if n_startup_jobs is None else int(n_startup_jobs)
                job_id = f"{tune_run_id}__{mode}__{model}__{step}__{domain}"
                print(f"\n=== TUNING JOB {job_id} ===")
                print("Folds:", folds_path)
                print("Data sha256:", data_sha)
                existing = _lookup_saved_tuning(
                    project_dir=self.cfg.PROJECT_DIR,
                    logs_dir=self.cfg.LOGS_DIR,
                    registry=registry,
                    model=model,
                    step=step,
                    domain=domain,
                    mode=mode,
                    metric=self.c.metric,
                    max_evals=self.c.max_evals,
                    random_state=self.c.random_state,
                    selected_fold=self.c.selected_fold,
                    startup=startup,
                    space=space,
                    data_sha256=data_sha,
                )
                if existing["status"] == "complete" and not force:
                    print(f"[SKIP] {model} {step} {domain}: result for run {run_label} already exists.")
                    print("Best params:", existing["params"])
                    print("Trials:", existing["trials_path"])
                    results["jobs"].append({
                        "job_id": existing["job_id"] or job_id,
                        "model": model,
                        "step": step,
                        "domain": domain,
                        "mode": mode,
                        "status": "skipped",
                        "best_params": existing["params"],
                        "paths": {"trials": existing["trials_path"]},
                    })
                    continue
                if existing["status"] == "conflict" and not force:
                    raise RuntimeError(
                        f"Run {run_label} already has a tuning result for {model}/{step}/{domain}, "
                        f"but it does not match the current settings ({', '.join(existing['problems'])}). "
                        "Pass force=True to overwrite it."
                    )
                if force and existing["status"] in {"complete", "conflict"}:
                    print(
                        f"[FORCE] Overwriting the existing {model} {step} {domain} result "
                        f"for run {run_label}. Previous best params: {existing['params']}"
                    )

                df_folds = pd.read_csv(folds_path)
                feature_prefix = detect_feature_prefix(df_folds, step)
                print("Feature prefix:", feature_prefix)
                payload = dict(fingerprint_payload or {})
                payload.update({
                    "run": run_label,
                    "domain": domain,
                    "model": model,
                    "step": step,
                    "search_space": space,
                    "fixed_params": fixed,
                    "max_evals": self.c.max_evals,
                    "n_startup_jobs": startup,
                    "metric": self.c.metric,
                    "seed": self.c.random_state,
                    "selected_fold": self.c.selected_fold,
                    "target_spec": target_spec,
                    "data_sha256": data_sha,
                })
                fingerprint = json_fingerprint(payload)
                print("Fingerprint:", fingerprint)
                checkpoint_dir = Path(self.cfg.PROJECT_DIR) / "tuning" / "checkpoints"
                checkpoint_path = checkpoint_dir / f"{tune_run_id}__{model}__{step}__{domain}.pkl"
                if force and checkpoint_path.exists():
                    print(f"[FORCE] Overwriting checkpoint {checkpoint_path}")
                checkpoint = {
                    "experiment_id": f"{tune_run_id}__{model}__{step}__{domain}",
                    "path": str(checkpoint_path),
                    "progress_csv": str(checkpoint_path.with_suffix(".progress.json")),
                    "fingerprint": fingerprint,
                    "resume": resume,
                    "force": force,
                }

                # dispatch to model-specific tuner
                with LogTiming(f"tune_{model}_{step}_{domain}", enabled=getattr(self.cfg, "LOG_TIME", False)):
                    if model == "rf":
                        from core.hyperopt_tuning import rf_hyperopt_fold0
                        out = rf_hyperopt_fold0(
                            df_folds=df_folds,
                            domain=domain,
                            target_spec=target_spec,
                            feature_prefix=feature_prefix,
                            fold_col=self.c.fold_col,
                            is_test_col=self.c.is_test_col,
                            domain_col=self.c.domain_col,
                            selected_fold=self.c.selected_fold,
                            metric=self.c.metric,
                            max_evals=self.c.max_evals,
                            random_state=self.c.random_state,
                            show_progress=self.c.show_progress,
                            search_space=space,
                            fixed_params=fixed,
                            n_startup_jobs=startup,
                            checkpoint=checkpoint,
                        )
                    elif model == "lgbm":
                        from core.hyperopt_tuning import lgbm_hyperopt_fold0
                        out = lgbm_hyperopt_fold0(
                            df_folds=df_folds,
                            domain=domain,
                            target_spec=target_spec,
                            feature_prefix=feature_prefix,
                            fold_col=self.c.fold_col,
                            is_test_col=self.c.is_test_col,
                            domain_col=self.c.domain_col,
                            selected_fold=self.c.selected_fold,
                            metric=self.c.metric,
                            max_evals=self.c.max_evals,
                            random_state=self.c.random_state,
                            show_progress=self.c.show_progress,
                            search_space=space,
                            fixed_params=fixed,
                            n_startup_jobs=startup,
                            checkpoint=checkpoint,
                        )
                    else:
                        raise ValueError(f"Unknown model '{model}'. Use: rf | lgbm")

                # persist trials
                tuning_dir = Path(self.cfg.PROJECT_DIR) / "tuning" / tune_run_id
                tuning_dir.mkdir(parents=True, exist_ok=True)

                paths = {}
                if self.c.save_trials and "trials_df" in out:
                    trials_path = tuning_dir / f"trials__{mode}__{model}__{step}__{domain}.csv"
                    print(f"[WRITE] Trials -> {trials_path}")
                    mlu.write_csv(out["trials_df"], str(trials_path))
                    paths["trials"] = str(trials_path)

                # update registry
                if self.c.save_registry and out.get("complete") and "best_params" in out:
                    key = {"model": model, "step": step, "domain": domain, "mode": mode}
                    tuning_meta = {
                        "strategy": "single_fold",
                        "val_fold": self.c.selected_fold,
                        "metric": self.c.metric,
                        "max_evals": self.c.max_evals,
                        "seed": self.c.random_state,
                        "feature_prefix": feature_prefix,
                        "tune_run_id": tune_run_id,
                        "job_id": job_id,
                        "n_startup_jobs": startup,
                        "search_space": space,
                        "data_sha256": data_sha,
                    }
                    _upsert_registry_entry(
                        registry, key=key, params=_registry_params(out["best_params"]), tuning_meta=tuning_meta,
                    )
                    registry_dirty = True

                print("Best params:", out.get("best_params"))
                results["jobs"].append({
                    "job_id": job_id,
                    "model": model,
                    "step": step,
                    "domain": domain,
                    "mode": mode,
                    "status": "tuned",
                    "feature_prefix": feature_prefix,
                    "best_params": out.get("best_params"),
                    "paths": paths,
                })

            results["registry_path"] = str(reg_path)
            if self.c.save_registry and registry_dirty:
                print(f"\n[WRITE] Registry -> {reg_path}")
                _atomic_json(reg_path, registry)
            else:
                print(f"\n[SKIP] Registry unchanged: {reg_path}")

        except BaseException:
            traceback.print_exc()
            raise
        finally:
            logger.stop()

        return results


def _same_number(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_f = float(left)
    right_f = float(right)
    if np.isnan(left_f) or np.isnan(right_f):
        return False
    # A trials CSV does not preserve the last digits of a float.
    return bool(np.isclose(left_f, right_f, rtol=1e-9, atol=1e-12))


def _decode_saved_max_features(value):
    from core.hyperopt_tuning import RF_PAPER_V2_MAX_FEATURES
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        return None if value in {"", "None", "null"} else value
    return RF_PAPER_V2_MAX_FEATURES[int(value)]


def _params_match(saved: dict, registry: dict) -> bool:
    if _decode_saved_max_features(saved.get("max_features")) != registry.get("max_features"):
        if "max_features" in saved or "max_features" in registry:
            return False
    for name, expected in registry.items():
        if name == "max_features":
            continue
        if not _same_number(saved.get(name), expected):
            return False
    return True


def _registry_entry(registry: dict, key: dict) -> Optional[dict]:
    for entry in registry.get("entries", []):
        if entry.get("key") == key:
            return entry
    return None


def _trials_csv(project_dir: str, tune_run_id: str, mode: str, model: str, step: str, domain: str) -> Path:
    return (
        Path(project_dir) / "tuning" / tune_run_id
        / f"trials__{mode}__{model}__{step}__{domain}.csv"
    )


def _ledger_sha256(logs_dir: str, *, model: str, step: str, domain: str, tune_run_id: str) -> Optional[str]:
    logs = Path(logs_dir)
    if not logs.is_dir():
        return None
    for path in sorted(logs.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for job in payload.get("jobs") or []:
            if job.get("model") != model or job.get("step") != step or job.get("domain") != domain:
                continue
            recorded = (job.get("tuning_result") or {}).get("tune_run_id")
            if recorded != tune_run_id:
                continue
            digest = (job.get("input") or {}).get("sha256")
            if digest:
                return digest
    return None


def _lookup_saved_tuning(
    *,
    project_dir: str,
    logs_dir: str,
    registry: dict,
    model: str,
    step: str,
    domain: str,
    mode: str,
    metric: str,
    max_evals: int,
    random_state: int,
    selected_fold: int,
    startup: int,
    space: dict,
    data_sha256: str,
) -> dict:
    """Classify a saved tuning result for this run's registry.

    ``complete`` can be skipped. ``conflict`` is an existing result whose
    settings or data hash differ. ``incomplete`` has no finished trial table,
    so the checkpoint resume path can continue it. A different run uses another
    registry file and therefore looks absent.
    """
    from core.hyperopt_tuning import LGBM_PAPER_V2_SPACE, RF_PAPER_V2_SPACE

    key = {"model": model, "step": step, "domain": domain, "mode": mode}
    entry = _registry_entry(registry, key)
    empty = {"status": "absent", "params": None, "trials_path": None, "job_id": None, "problems": []}
    if entry is None or not entry.get("params"):
        return empty
    tuning = entry.get("tuning") or {}
    problems = []
    if tuning.get("metric") != metric:
        problems.append("metric")
    if tuning.get("max_evals") != max_evals:
        problems.append("max_evals")
    if tuning.get("seed") != random_state:
        problems.append("seed")
    if tuning.get("val_fold") != selected_fold:
        problems.append("val_fold")
    stored_startup = tuning.get("n_startup_jobs")
    if stored_startup is None:
        if startup != 20:
            problems.append("n_startup_jobs")
    elif int(stored_startup) != startup:
        problems.append("n_startup_jobs")
    default_space = RF_PAPER_V2_SPACE if model == "rf" else LGBM_PAPER_V2_SPACE
    stored_space = tuning.get("search_space")
    if stored_space is None:
        if space != default_space:
            problems.append("search_space")
    elif stored_space != space:
        problems.append("search_space")

    tune_run_id = tuning.get("tune_run_id")
    trials_path = None
    finished_table = False
    if not tune_run_id:
        problems.append("tune_run_id")
    else:
        trials_path = _trials_csv(project_dir, tune_run_id, mode, model, step, domain)
        if not trials_path.is_file():
            problems.append("trials")
        else:
            trials = pd.read_csv(trials_path)
            expected_rows = int(tuning.get("max_evals") or max_evals)
            if len(trials) != expected_rows:
                problems.append("trial_count")
            elif not _params_match(trials.loc[trials["loss"].idxmin()].to_dict(), entry["params"]):
                problems.append("best_params")
            else:
                finished_table = True
    recorded_hash = tuning.get("data_sha256")
    if recorded_hash is None and tune_run_id:
        recorded_hash = _ledger_sha256(
            logs_dir, model=model, step=step, domain=domain, tune_run_id=tune_run_id,
        )
    if recorded_hash is not None and recorded_hash != data_sha256:
        problems.append("data_sha256")

    blocking = [item for item in problems if item not in {"trials", "trial_count"}]
    if finished_table and not blocking:
        status = "complete"
    elif blocking:
        status = "conflict"
    else:
        status = "incomplete"
    return {
        "status": status,
        "params": entry.get("params"),
        "trials_path": None if trials_path is None else str(trials_path),
        "job_id": tuning.get("job_id"),
        "problems": problems,
    }


def _split_counts(path: Path, *, domain: str, selected_fold: int) -> dict:
    frame = pd.read_csv(path, usecols=["is_test", "fold", "chem_class", "centrosymmetry"])
    frame = frame[frame["chem_class"].astype(str) == str(domain)]
    trainable = frame[frame["is_test"] == 0]
    train = trainable[trainable["fold"] != selected_fold]
    validation = trainable[trainable["fold"] == selected_fold]
    return {
        "n_train": int(len(train)),
        "n_val": int(len(validation)),
        "train_class_counts": train["centrosymmetry"].value_counts().sort_index().to_dict(),
        "val_class_counts": validation["centrosymmetry"].value_counts().sort_index().to_dict(),
    }


def accept_completed_tuning(cfg, spec: dict) -> dict:
    """Accept a finished PAPER_V2 RF job without refitting or rewriting its files.

    A trials CSV alone is not enough. The ledger must show a finished job, the
    registry entry must match that run, and the current fold file hash must match.
    """
    from core.hyperopt_tuning import RF_PAPER_V2_SPACE
    from core.ml_utils import file_sha256

    adopt = spec["adopt_completed"]
    failures = []
    if spec.get("search_space") != RF_PAPER_V2_SPACE:
        failures.append("search_space does not match the completed PAPER_V2 RF space")
    if spec.get("max_evals") != 50 or spec.get("n_startup_jobs") != 20:
        failures.append("completed PAPER_V2 jobs used 50 trials and the hyperopt default of 20 startup trials")
    if spec.get("metric") != "auc" or spec.get("seed") != 42 or spec.get("selected_fold") != 0:
        failures.append("metric, seed or validation fold does not match the completed run")
    if spec.get("stage_b") not in (None, {}):
        failures.append("the completed run has no stage B")
    if spec.get("series") != "PAPER_V2":
        failures.append("completed results belong to series PAPER_V2")
    fixed = spec.get("fixed_params") or {}
    if fixed.get("n_jobs") != -1 or fixed.get("random_state") != 42 or fixed.get("class_weight") is not None:
        failures.append("fixed tuning parameters do not match the completed RF run")

    ledger_path = Path(cfg.LOGS_DIR) / adopt["ledger_name"]
    registry_path = Path(cfg.PROJECT_DIR) / "tuning" / spec["registry_filename"]
    trials_path = (
        Path(cfg.PROJECT_DIR) / "tuning" / adopt["tune_run_id"] / adopt["trials_name"]
    )
    folds_path = Path(cfg.DATASETS_DIR) / f"dataset_splits_folds_{spec['domain']}.csv"
    for label, path in ("ledger", ledger_path), ("registry", registry_path), ("trials", trials_path), ("folds", folds_path):
        if not path.is_file():
            failures.append(f"missing {label}: {path}")
    if failures:
        raise RuntimeError("Completed tuning was not accepted:\n- " + "\n- ".join(failures))

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    jobs = [
        job for job in ledger.get("jobs", [])
        if job.get("model") == spec["model"] and job.get("step") == spec["featurizer"] and job.get("domain") == spec["domain"]
    ]
    if len(jobs) != 1 or not jobs[0].get("elapsed_seconds"):
        failures.append("ledger does not contain exactly one finished job for this experiment")
    else:
        recorded_hash = jobs[0].get("input", {}).get("sha256")
        current_hash = file_sha256(folds_path)
        if recorded_hash != current_hash:
            failures.append("fold file hash differs from the completed job")
        recorded_run = jobs[0].get("tuning_result", {}).get("tune_run_id")
        if recorded_run != adopt["tune_run_id"]:
            failures.append("ledger tune_run_id differs from the configured completed run")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    key = {"model": spec["model"], "step": spec["featurizer"], "domain": spec["domain"], "mode": "cv"}
    entry = next((item for item in registry.get("entries", []) if item.get("key") == key), None)
    if entry is None:
        failures.append(f"registry has no entry for {key}")
    else:
        tuning = entry.get("tuning") or {}
        if tuning.get("tune_run_id") != adopt["tune_run_id"]:
            failures.append("registry tune_run_id differs from the configured completed run")
        if tuning.get("max_evals") != 50 or tuning.get("metric") != "auc" or tuning.get("seed") != 42 or tuning.get("val_fold") != 0:
            failures.append("registry tuning metadata does not match the completed protocol")
        trials = pd.read_csv(trials_path)
        if len(trials) != 50:
            failures.append(f"trials file has {len(trials)} rows, not 50")
        elif entry.get("params"):
            best = trials.loc[trials["loss"].idxmin()]
            if not _params_match(best.to_dict(), entry["params"]):
                failures.append("best trial does not match the registry parameters")

    if failures:
        raise RuntimeError(
            "Completed tuning was not accepted and will not be recomputed automatically:\n- "
            + "\n- ".join(failures)
        )

    counts = _split_counts(folds_path, domain=spec["domain"], selected_fold=spec["selected_fold"])
    print("Accepted existing completed tuning. No fit was started and no artifact was rewritten.")
    print("The trials CSV is a finished result table. It is not a TPE checkpoint: Trials and RNG state are absent.")
    print("Ledger:", ledger_path)
    print("Registry:", registry_path)
    print("Trials:", trials_path)
    print("Data sha256:", file_sha256(folds_path))
    print("Split counts:", counts)
    print("Best params:", entry["params"])
    return {
        "adopted": True,
        "complete": True,
        "tune_run_id": adopt["tune_run_id"],
        "best_params": entry["params"],
        "trials_path": str(trials_path),
        "registry_path": str(registry_path),
        "ledger_path": str(ledger_path),
        "counts": counts,
    }


def run_configured(spec: dict, stage: str) -> dict:
    """Run one experiment configuration. ``stage`` is ``tune``, ``cv`` or ``final``."""
    from importlib.metadata import PackageNotFoundError, version
    import traceback
    from project_config import make_cfg
    from private_paths import COD_DIR, MP_DIR

    stage = str(stage).strip().lower()
    if stage not in {"tune", "cv", "final"}:
        raise ValueError("stage must be tune, cv or final")
    roots = {"COD_CANON_V2": COD_DIR, "MP_ALL": MP_DIR}
    if spec["project"] not in roots:
        raise ValueError(f"Unknown project {spec['project']}.")
    cfg = make_cfg(project_name=spec["project"], root_cif=roots[spec["project"]], log_time=True)
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "hyperopt"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    context = {"spec": spec, "stage": stage, "package_versions": versions}

    if stage == "tune" and spec.get("adopt_completed"):
        from core.notebook_logger import NotebookLogger
        logger = NotebookLogger(
            project_dir=cfg.PROJECT_DIR,
            prefix=f"{spec['series']}_{spec['project']}_tune",
        )
        logger.start(header_lines=[
            f"Series: {spec['series']}; project: {spec['project']}; stage: tune",
            "Adopt completed tuning if the ledger, registry, trials and data hash agree.",
        ])
        print("Configuration:")
        print(json.dumps(context, indent=2, sort_keys=True, default=str))
        try:
            return accept_completed_tuning(cfg, spec)
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            logger.stop()

    fingerprint_payload = {
        "series": spec["series"],
        "project": spec["project"],
        "featurizer": spec["featurizer"],
        "search_space": spec["search_space"],
        "fixed_params": spec["fixed_params"],
        "max_evals": spec["max_evals"],
        "n_startup_jobs": spec["n_startup_jobs"],
        "metric": spec["metric"],
        "seed": spec["seed"],
        "selected_fold": spec["selected_fold"],
        "target_spec": spec["target_spec"],
        "resume": spec["resume"],
    }
    if stage == "tune":
        tune_cfg = HyperTuningPipelineConfig(
            fold_col="fold",
            is_test_col="is_test",
            domain_col="chem_class",
            selected_fold=spec["selected_fold"],
            max_evals=spec["max_evals"],
            metric=spec["metric"],
            random_state=spec["seed"],
            show_progress=True,
            save_trials=True,
            save_registry=True,
            registry_filename=spec["registry_filename"],
        )
        return HyperTuningPipeline(cfg, tune_cfg).run(
            models=(spec["model"],),
            steps=(spec["featurizer"],),
            domains=(spec["domain"],),
            mode="cv",
            run=f"TUNE_{spec['series']}",
            target_spec=spec["target_spec"],
            search_space=spec["search_space"],
            fixed_params=spec["fixed_params"],
            n_startup_jobs=spec["n_startup_jobs"],
            series=spec["series"],
            fingerprint_payload=fingerprint_payload,
            resume=bool(spec["resume"]),
            log_context=context,
        )

    from pipelines.train_pipeline import TrainPipeline, TrainPipelineConfig
    train_cfg = TrainPipelineConfig(
        fold_col="fold",
        is_test_col="is_test",
        group_col="group_id",
        verbose=True,
        verbose_batches=True,
        verbose_summary=True,
        save_artifacts=True,
    )
    registry_path = str(Path(cfg.PROJECT_DIR) / "tuning" / spec["registry_filename"])
    return TrainPipeline(cfg, train_cfg).run(
        models=(spec["model"],),
        steps=(spec["featurizer"],),
        domains=(spec["domain"],),
        mode=stage,
        run=f"{spec['model']}_RUN_{spec['series']}",
        target_name="centrosymmetry_CN",
        params_registry_path=registry_path,
        use_registry=True,
        series=spec["series"],
        fingerprint_payload=fingerprint_payload,
        resume=bool(spec["resume"]),
        log_context=context,
    )
