"""Optimize classifier parameters with resumable Hyperopt TPE searches."""


from __future__ import annotations

import json
from functools import partial
from typing import Dict, Optional

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.base import JOB_STATE_NEW
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

from core import ml_utils


# Historical search spaces. Callers that pass search_space replace these.
RF_PAPER_V2_SPACE = {
    "n_estimators": {"quniform": [100, 1500, 50]},
    "max_depth": {"quniform": [5, 60, 1]},
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": {"quniform": [2, 10, 1]},
    "min_samples_leaf": {"quniform": [1, 5, 1]},
}
LGBM_PAPER_V2_SPACE = {
    "n_estimators": {"quniform": [200, 2000, 50]},
    "learning_rate": {"loguniform": [0.01, 0.2]},
    "num_leaves": {"quniform": [15, 255, 4]},
    "max_depth": {"quniform": [3, 16, 1]},
    "min_child_samples": {"quniform": [5, 50, 1]},
    "subsample": {"uniform": [0.6, 1.0]},
    "colsample_bytree": {"uniform": [0.6, 1.0]},
    "reg_alpha": {"loguniform": [1e-8, 10.0]},
    "reg_lambda": {"loguniform": [1e-8, 10.0]},
}
RF_PAPER_V2_MAX_FEATURES = ["sqrt", "log2", None]


def _score(y_true: np.ndarray, y_prob: np.ndarray, metric: str) -> float:
    """Compute the requested classification score from labels and probabilities."""
    if metric == "auc":
        return roc_auc_score(y_true, y_prob)
    if metric == "auprc":
        return average_precision_score(y_true, y_prob)
    if metric == "acc":
        y_hat = (y_prob >= 0.5).astype(int)
        return accuracy_score(y_true, y_hat)
    raise ValueError("metric must be one of: 'auc', 'auprc', 'acc'")


def _detect_feature_cols(df: pd.DataFrame, feature_prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(feature_prefix)]
    if not cols:
        raise ValueError(f"No feature columns found for prefix='{feature_prefix}'.")
    return sorted(cols)


def build_hyperopt_space(search_space: dict) -> dict:
    """Turn a notebook search-space dict into a hyperopt space.

    Lists become ``hp.choice``. A dict with one of ``quniform``, ``uniform`` or
    ``loguniform`` becomes the matching hyperopt distribution. Fixed scalars
    belong in ``fixed_params``, not here.
    """
    space = {}
    for name, value in search_space.items():
        if isinstance(value, list):
            space[name] = hp.choice(name, value)
        elif isinstance(value, dict) and set(value) == {"quniform"}:
            low, high, step = value["quniform"]
            space[name] = hp.quniform(name, low, high, step)
        elif isinstance(value, dict) and set(value) == {"uniform"}:
            low, high = value["uniform"]
            space[name] = hp.uniform(name, low, high)
        elif isinstance(value, dict) and set(value) == {"loguniform"}:
            low, high = value["loguniform"]
            space[name] = hp.loguniform(name, np.log(low), np.log(high))
        else:
            raise ValueError(
                f"Unsupported search dimension {name}={value!r}. "
                "Use a list or quniform/uniform/loguniform. Put fixed values in fixed_params."
            )
    return space


def _cast_model_param(name: str, value):
    integer_names = {
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "num_leaves", "min_child_samples",
    }
    if name in integer_names and value is not None:
        return int(value)
    return value


def _model_params(sampled: dict, fixed_params: Optional[dict]) -> dict:
    """Parameters actually passed to the estimator, with readable categories."""
    params = {}
    for name, value in dict(fixed_params or {}).items():
        if name == "class_weight" and value is None:
            continue
        params[name] = _cast_model_param(name, value)
    for name, value in sampled.items():
        params[name] = _cast_model_param(name, value)
    return params


def _prepare_xy(
    df_folds: pd.DataFrame,
    *,
    domain: str,
    target_spec: dict,
    feature_prefix: str,
    fold_col: str,
    is_test_col: str,
    domain_col: str,
    selected_fold: int,
):
    df = df_folds.copy()
    if is_test_col in df.columns:
        df = df[df[is_test_col] == 0].copy()
    if domain_col in df.columns:
        df = df[df[domain_col].astype(str) == str(domain)].copy()
    df, y_col = ml_utils.apply_target_spec(df, target_spec)
    feature_cols = _detect_feature_cols(df, feature_prefix)
    x_values = df[feature_cols]
    y_values = df[y_col].astype(int).values
    if fold_col not in df.columns:
        raise KeyError(f"Missing fold_col='{fold_col}' in df_folds.")
    is_val = (df[fold_col].values == selected_fold)
    if is_val.sum() == 0:
        raise ValueError(f"No samples for selected_fold={selected_fold} in domain={domain}.")
    counts = {
        "n_train": int((~is_val).sum()),
        "n_val": int(is_val.sum()),
        "train_class_counts": pd.Series(y_values[~is_val]).value_counts().sort_index().to_dict(),
        "val_class_counts": pd.Series(y_values[is_val]).value_counts().sort_index().to_dict(),
    }
    return (
        x_values.loc[~is_val], y_values[~is_val],
        x_values.loc[is_val], y_values[is_val],
        feature_cols, y_col, counts,
    )


class _SavingTrials(Trials):
    """Trials that persist state after each hyperopt refresh."""

    on_refresh = None

    def refresh(self):
        super().refresh()
        if self.on_refresh is not None:
            self.on_refresh(self)


def _copy_trials(source: Trials) -> _SavingTrials:
    copied = _SavingTrials()
    for doc in source.trials:
        copied.insert_trial_doc(doc)
    copied.on_refresh = None
    copied.refresh()
    return copied


def _ok_results(trials: Trials) -> list:
    return [doc for doc in trials.trials if doc.get("result", {}).get("status") == STATUS_OK]


def _plain_trials(current: Trials) -> Trials:
    """Copy trial documents without the refresh callback, so the pickle stays loadable."""
    callback = getattr(current, "on_refresh", None)
    current.on_refresh = None
    plain = Trials()
    try:
        for doc in current.trials:
            plain.insert_trial_doc(doc)
        plain.refresh()
    finally:
        current.on_refresh = callback
    return plain


def _run_logged_search(
    *,
    kind: str,
    build_estimator,
    df_folds: pd.DataFrame,
    domain: str,
    target_spec: dict,
    feature_prefix: str,
    fold_col: str,
    is_test_col: str,
    domain_col: str,
    selected_fold: int,
    metric: str,
    max_evals: int,
    random_state: int,
    show_progress: bool,
    search_space: Optional[dict],
    fixed_params: Optional[dict],
    n_startup_jobs: Optional[int],
    checkpoint: Optional[dict],
) -> Dict:
    """Shared TPE loop with a per-fit log and an atomic checkpoint.

    ``checkpoint`` is a dict with experiment_id, path, fingerprint, resume and
    progress_csv. A progress CSV is a readable table only. Resume uses the
    pickle, which holds Trials and the random generator state.
    """
    (
        x_train, y_train, x_val, y_val, feature_cols, y_col, counts,
    ) = _prepare_xy(
        df_folds,
        domain=domain,
        target_spec=target_spec,
        feature_prefix=feature_prefix,
        fold_col=fold_col,
        is_test_col=is_test_col,
        domain_col=domain_col,
        selected_fold=selected_fold,
    )
    print(
        f"[DATA] train={counts['n_train']} validation={counts['n_val']} "
        f"train_class_counts={counts['train_class_counts']} "
        f"val_class_counts={counts['val_class_counts']}"
    )
    print(
        "[MODEL] class_weight is omitted when it is null, so the estimator keeps "
        "its own default. Tuning does not enable class_weight='balanced'."
    )

    plain_space = search_space if search_space is not None else (
        RF_PAPER_V2_SPACE if kind == "rf" else LGBM_PAPER_V2_SPACE
    )
    space = build_hyperopt_space(plain_space)
    startup = 20 if n_startup_jobs is None else int(n_startup_jobs)
    experiment_id = None if not checkpoint else checkpoint.get("experiment_id")
    checkpoint_path = None if not checkpoint else checkpoint.get("path")
    progress_csv = None if not checkpoint else checkpoint.get("progress_csv")
    fingerprint = None if not checkpoint else checkpoint.get("fingerprint")
    resume = True if not checkpoint else bool(checkpoint.get("resume", True))

    rng = np.random.default_rng(random_state)
    trials = _SavingTrials()
    if checkpoint_path and not checkpoint.get("force"):
        loaded = ml_utils.load_checkpoint(checkpoint_path, fingerprint, resume=resume)
        if loaded is not None:
            if loaded.get("kind") != "tune":
                raise RuntimeError(f"Checkpoint {checkpoint_path} is not a tuning checkpoint.")
            if "trials" not in loaded or "rng_state" not in loaded:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} has no Trials or RNG state. "
                    "A trials CSV is not a TPE checkpoint and was not used to resume."
                )
            trials = _copy_trials(loaded["trials"])
            rng.bit_generator.state = loaded["rng_state"]
            print(
                f"[RESUME] experiment={experiment_id} completed_trials={len(_ok_results(trials))} "
                f"checkpoint={checkpoint_path}"
            )
            if loaded.get("finished"):
                print(
                    f"[FIT done] experiment={experiment_id} stage=tune status=skipped_completed "
                    f"auc={loaded.get('best_score')} checkpoint={checkpoint_path}"
                )
                return _search_result(
                    trials=trials,
                    best_params=loaded["best_params"],
                    best_score=float(loaded["best_score"]),
                    metric=metric,
                    selected_fold=selected_fold,
                    feature_prefix=feature_prefix,
                    feature_cols=feature_cols,
                    y_col=y_col,
                    counts=counts,
                    plain_space=plain_space,
                    fixed_params=fixed_params,
                    startup=startup,
                    max_evals=max_evals,
                )

    def _persist(current: Trials) -> None:
        ok_rows = []
        best_auc = None
        for index, doc in enumerate(_ok_results(current), start=1):
            result = doc["result"]
            score = float(result["score"])
            best_auc = score if best_auc is None else max(best_auc, score)
            params = result.get("params") or {}
            ok_rows.append({
                "experiment": experiment_id,
                "stage": "tune",
                "trial": index,
                "status": "ok",
                "auc": score,
                "started": result.get("started"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "best_auc": best_auc,
                **{key: params.get(key) for key in params},
            })
        if checkpoint_path:
            ml_utils.atomic_pickle(checkpoint_path, {
                "kind": "tune",
                "fingerprint": fingerprint,
                "trials": _plain_trials(current),
                "rng_state": rng.bit_generator.state,
                "finished": False,
            })
        if progress_csv is not None:
            ml_utils.atomic_text(
                progress_csv,
                json.dumps(ok_rows, indent=2, default=str),
            )

    logged_ok = len(_ok_results(trials))

    def _after_refresh(current: Trials) -> None:
        nonlocal logged_ok
        _persist(current)
        ok_docs = _ok_results(current)
        if len(ok_docs) <= logged_ok:
            return
        scores = [float(doc["result"]["score"]) for doc in ok_docs]
        latest = ok_docs[-1]["result"]
        print(
            f"[FIT done] experiment={experiment_id} stage=tune "
            f"trial={len(ok_docs)}/{max_evals} status=ok auc={latest.get('score')} "
            f"started={latest.get('started')} elapsed_seconds={latest.get('elapsed_seconds')} "
            f"best_auc={max(scores)} checkpoint_saved={checkpoint_path}"
        )
        logged_ok = len(ok_docs)

    trials.on_refresh = _after_refresh

    def objective(params):
        model_params = _model_params(params, fixed_params)
        done = len(_ok_results(trials))
        # The trial with state JOB_STATE_NEW is the next scheduled fit.
        pending = sum(1 for doc in trials.trials if doc.get("state") == JOB_STATE_NEW)
        trial_number = done + max(pending, 1)
        print(
            f"[FIT start] experiment={experiment_id} stage=tune "
            f"trial={trial_number}/{max_evals} params={model_params}"
        )
        started = ml_utils.utc_now()
        started_counter = ml_utils.time.perf_counter()
        estimator = build_estimator(model_params)
        estimator.fit(x_train, y_train)
        probability = estimator.predict_proba(x_val)[:, 1]
        score = _score(y_val, probability, metric)
        elapsed = ml_utils.time.perf_counter() - started_counter
        return {
            "loss": -float(score),
            "status": STATUS_OK,
            "score": float(score),
            "started": started,
            "elapsed_seconds": elapsed,
            "params": model_params,
        }

    n_done = len(_ok_results(trials))
    pbar = tqdm(
        total=max_evals,
        initial=min(n_done, max_evals),
        desc=f"Hyperopt {kind.upper()} {domain}",
        leave=False,
        disable=not show_progress,
    )

    if n_done < max_evals:
        fmin(
            fn=objective,
            space=space,
            algo=partial(tpe.suggest, n_startup_jobs=startup),
            max_evals=max_evals,
            trials=trials,
            rstate=rng,
            show_progressbar=False,
        )
    elif n_done >= max_evals:
        print(
            f"[RESUME] experiment={experiment_id} stage=tune already has {n_done} "
            f"completed trials; budget is {max_evals}."
        )
    _persist(trials)
    pbar.n = min(len(_ok_results(trials)), max_evals)
    pbar.refresh()
    pbar.close()

    ok_docs = _ok_results(trials)
    if len(ok_docs) < max_evals:
        raise RuntimeError(
            f"Tuning is incomplete ({len(ok_docs)}/{max_evals}). "
            "Partial tuning is not marked complete."
        )

    def _best_of(rows: list) -> tuple:
        winner = max(rows, key=lambda item: float(item["score"] if "score" in item else item["auc"]))
        score = float(winner["score"] if "score" in winner else winner["auc"])
        params = dict(winner["params"] if "params" in winner else winner)
        params.pop("experiment", None)
        params.pop("stage", None)
        params.pop("trial", None)
        params.pop("status", None)
        params.pop("auc", None)
        params.pop("started", None)
        params.pop("elapsed_seconds", None)
        params.pop("best_auc", None)
        return score, params

    tuned_rows = []
    for doc in ok_docs:
        result = doc["result"]
        tuned_rows.append({"score": float(result["score"]), "params": dict(result["params"])})
    best_score, best_params = _best_of(tuned_rows)

    if checkpoint_path:
        ml_utils.atomic_pickle(checkpoint_path, {
            "kind": "tune",
            "fingerprint": fingerprint,
            "trials": _plain_trials(trials),
            "rng_state": rng.bit_generator.state,
            "finished": True,
            "best_params": best_params,
            "best_score": best_score,
        })
        print(f"[DONE] experiment={experiment_id} best_auc={best_score} checkpoint_saved={checkpoint_path}")

    return _search_result(
        trials=trials,
        best_params=best_params,
        best_score=best_score,
        metric=metric,
        selected_fold=selected_fold,
        feature_prefix=feature_prefix,
        feature_cols=feature_cols,
        y_col=y_col,
        counts=counts,
        plain_space=plain_space,
        fixed_params=fixed_params,
        startup=startup,
        max_evals=max_evals,
    )


def _search_result(
    *,
    trials,
    best_params,
    best_score,
    metric,
    selected_fold,
    feature_prefix,
    feature_cols,
    y_col,
    counts,
    plain_space,
    fixed_params,
    startup,
    max_evals,
) -> Dict:
    rows = []
    for doc in _ok_results(trials):
        result = doc["result"]
        row = {"loss": result.get("loss"), "score": result.get("score"), "stage": "tune"}
        row.update(result.get("params") or {})
        rows.append(row)
    trials_df = pd.DataFrame(rows)
    if not trials_df.empty:
        trials_df = trials_df.sort_values("loss")
    return {
        "best_params": best_params,
        "best_score": best_score,
        "trials_df": trials_df,
        "metric": metric,
        "selected_fold": selected_fold,
        "feature_prefix": feature_prefix,
        "feature_cols": feature_cols,
        "y_col": y_col,
        "n_train": counts["n_train"],
        "n_val": counts["n_val"],
        "complete": True,
        "search_space": plain_space,
        "fixed_params": fixed_params or {},
        "n_startup_jobs": startup,
        "max_evals": max_evals,
    }


def _rf_estimator(params: dict) -> RandomForestClassifier:
    # class_weight is not set here. sklearn's default is None, unlike build_rf.
    return RandomForestClassifier(**params)


def _lgbm_estimator(params: dict) -> LGBMClassifier:
    params = dict(params)
    params.setdefault("verbosity", -1)
    params.setdefault("n_jobs", -1)
    return LGBMClassifier(**params)


def rf_hyperopt_fold0(
    *,
    df_folds: pd.DataFrame,
    domain: str,
    target_spec: dict,
    feature_prefix: str,
    fold_col: str = "fold",
    is_test_col: str = "is_test",
    domain_col: str = "chem_class",
    selected_fold: int = 0,
    metric: str = "auc",
    max_evals: int = 50,
    random_state: int = 42,
    show_progress: bool = True,
    search_space: Optional[dict] = None,
    fixed_params: Optional[dict] = None,
    n_startup_jobs: Optional[int] = None,
    checkpoint: Optional[dict] = None,
) -> Dict:
    """
    RF hypertuning on train/validation records only (is_test==0).

    The default space is the historical PAPER_V2 space. Pass ``search_space``
    to use the notebook configuration instead. ``fixed_params`` are passed
    through to ``RandomForestClassifier`` and are not sampled.
    """
    if fixed_params is None:
        fixed_params = {"n_jobs": -1, "random_state": random_state}
    return _run_logged_search(
        kind="rf",
        build_estimator=_rf_estimator,
        df_folds=df_folds,
        domain=domain,
        target_spec=target_spec,
        feature_prefix=feature_prefix,
        fold_col=fold_col,
        is_test_col=is_test_col,
        domain_col=domain_col,
        selected_fold=selected_fold,
        metric=metric,
        max_evals=max_evals,
        random_state=random_state,
        show_progress=show_progress,
        search_space=search_space,
        fixed_params=fixed_params,
        n_startup_jobs=n_startup_jobs,
        checkpoint=checkpoint,
    )


def lgbm_hyperopt_fold0(
    *,
    df_folds: pd.DataFrame,
    domain: str,
    target_spec: dict,
    feature_prefix: str,
    fold_col: str = "fold",
    is_test_col: str = "is_test",
    domain_col: str = "chem_class",
    selected_fold: int = 0,
    metric: str = "auc",
    max_evals: int = 50,
    random_state: int = 42,
    show_progress: bool = True,
    search_space: Optional[dict] = None,
    fixed_params: Optional[dict] = None,
    n_startup_jobs: Optional[int] = None,
    checkpoint: Optional[dict] = None,
) -> Dict:
    """
    LGBM hypertuning on train/validation records only (is_test==0).

    The default space is the historical PAPER_V2 space.
    """
    if fixed_params is None:
        fixed_params = {"n_jobs": -1, "random_state": random_state, "verbosity": -1}
    return _run_logged_search(
        kind="lgbm",
        build_estimator=_lgbm_estimator,
        df_folds=df_folds,
        domain=domain,
        target_spec=target_spec,
        feature_prefix=feature_prefix,
        fold_col=fold_col,
        is_test_col=is_test_col,
        domain_col=domain_col,
        selected_fold=selected_fold,
        metric=metric,
        max_evals=max_evals,
        random_state=random_state,
        show_progress=show_progress,
        search_space=search_space,
        fixed_params=fixed_params,
        n_startup_jobs=n_startup_jobs,
        checkpoint=checkpoint,
    )
