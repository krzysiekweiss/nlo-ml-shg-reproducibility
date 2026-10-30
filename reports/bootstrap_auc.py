"""Estimate conditional metric intervals by resampling complete composition groups."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from tqdm.auto import trange

from core.ml_utils import atomic_pickle, json_fingerprint, load_checkpoint

from .io import as_list, load_all_holdouts_legacy, save_table


def _metric_value(metric_key: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    if metric_key == "auc":
        return float(roc_auc_score(y_true, y_prob))
    y_hat = (y_prob >= float(threshold)).astype(int)
    if metric_key == "bacc":
        return float(balanced_accuracy_score(y_true, y_hat))
    return float(matthews_corrcoef(y_true, y_hat))


def _prepare_groups(y: np.ndarray, y_prob: np.ndarray, group_id: np.ndarray):
    """Sort records by composition group and return per-row group labels."""
    if pd.isna(pd.Series(group_id)).any():
        raise ValueError(
            "Holdout group_id contains missing values. "
            "The bootstrap unit is the composition group."
        )
    order = np.argsort(group_id, kind="mergesort")
    y_sorted = np.asarray(y, dtype=float)[order]
    prob_sorted = np.asarray(y_prob, dtype=float)[order]
    groups_sorted = np.asarray(group_id)[order]
    if groups_sorted.size == 0:
        raise ValueError("Cannot bootstrap an empty holdout.")
    if groups_sorted.size == 1:
        change = np.array([], dtype=int)
    else:
        change = np.flatnonzero(groups_sorted[1:] != groups_sorted[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, groups_sorted.size]
    lengths = ends - starts
    row_group = np.repeat(np.arange(starts.size), lengths)
    return y_sorted, prob_sorted, row_group, int(starts.size)


def _holdout_sha256(y: np.ndarray, y_prob: np.ndarray, group_id: np.ndarray) -> str:
    frame = pd.DataFrame({"group_id": group_id, "y": y, "y_prob": y_prob})
    digest = hashlib.sha256(frame.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def _sample_groups(y_sorted, prob_sorted, row_group, n_groups: int, rng: np.random.Generator):
    """Draw composition groups with replacement and keep every record in each draw."""
    chosen = rng.integers(0, n_groups, size=n_groups)
    counts = np.bincount(chosen, minlength=n_groups)
    return np.repeat(y_sorted, counts[row_group]), np.repeat(prob_sorted, counts[row_group])


def bootstrap_metric_from_holdouts(
    cfg,
    *,
    metric: str = "auc",
    runs: Mapping[str, str],
    models: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    mode: str = "final",
    n_boot: int = 2000,
    seed: int = 42,
    ci: tuple[float, float] = (2.5, 97.5),
    threshold: float = 0.5,
    map_N_to_I: bool = True,
    save: bool = True,
    project_dir: Union[str, Path, None] = None,
    save_run_id: str = "paper",
    name: Optional[str] = None,
    warn: bool = True,
    verbose: bool = True,
    force: bool = False,
    resume: bool = True,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Bootstrap a holdout metric by composition group.

    Each replicate draws ``group_id`` values with replacement and keeps every
    record that belongs to a drawn group. The saved point estimate is the
    metric on the full test set. ``ci_low`` and ``ci_high`` are percentiles of
    the bootstrap replicates and are stored separately from that point estimate.

    A matching finished checkpoint is skipped. An interrupted checkpoint
    resumes from the last saved replicate. ``force`` overwrites the checkpoint.
    ``resume=False`` refuses to ignore a checkpoint that already exists.
    """
    metric_key = str(metric).strip().lower()
    metric_label = {"auc": "AUC", "mcc": "MCC", "bacc": "BACC"}.get(metric_key, metric_key.upper())
    if metric_key not in {"auc", "mcc", "bacc"}:
        raise ValueError("Unsupported metric. Choose from: 'auc', 'mcc', 'bacc'.")
    if int(n_boot) < 1:
        raise ValueError("n_boot must be at least 1.")
    if int(progress_every) < 1:
        raise ValueError("progress_every must be at least 1.")
    n_boot = int(n_boot)
    progress_every = int(progress_every)

    models_l = as_list(models)
    if models_l is None:
        models_l = sorted([str(m) for m in runs.keys()])

    domains_l = as_list(domains)

    rows: list[dict] = []
    results_dir = Path(cfg.RESULTS_DIR)
    out_project_dir = Path(project_dir or getattr(cfg, "PROJECT_DIR", "."))
    progress_dir = out_project_dir / "results" / str(save_run_id) / "bootstrap_progress"

    for model in models_l:
        if model not in runs:
            raise ValueError(f"Missing run_id for model='{model}' in runs mapping.")
        run_id = str(runs[model])

        df = load_all_holdouts_legacy(
            results_dir=results_dir,
            run_id=run_id,
            model=str(model),
            project=getattr(cfg, "PROJECT_NAME", None),
            mode=mode,
            domains=None,  # Load predictions before mapping inorganic domain N to display label I.
            warn=warn,
        )
        if df.empty:
            continue

        if map_N_to_I and "domain" in df.columns:
            df = df.copy()
            df["domain"] = df["domain"].replace({"N": "I"})

        if domains_l is not None and "domain" in df.columns:
            df = df[df["domain"].isin(domains_l)]
        if df.empty:
            continue

        if "domain" not in df.columns:
            raise KeyError("Holdout table missing 'domain' column.")
        if "y" not in df.columns or "y_prob" not in df.columns:
            raise KeyError("Holdout table must contain columns 'y' and 'y_prob'.")
        if "group_id" not in df.columns:
            raise KeyError(
                "Holdout table missing 'group_id'. "
                "The bootstrap unit is the composition group, not the record."
            )

        for dom, gg in df.groupby("domain", sort=True):
            y = gg["y"].to_numpy()
            y_prob = gg["y_prob"].to_numpy()
            group_id = gg["group_id"].to_numpy()
            if y.size == 0:
                continue

            y_sorted, prob_sorted, row_group, n_groups = _prepare_groups(y, y_prob, group_id)
            n_records = int(y.size)
            if np.unique(y).size < 2:
                if warn:
                    print(
                        f"[WARN] Test set has one class for proj={cfg.PROJECT_NAME}, "
                        f"model={model}, domain={dom}."
                    )
                continue

            test_value = _metric_value(metric_key, y, y_prob, threshold)
            holdout_sha = _holdout_sha256(y, y_prob, group_id)
            fingerprint = json_fingerprint({
                "unit": "group_id",
                "metric": metric_key,
                "project": getattr(cfg, "PROJECT_NAME", None),
                "model": str(model),
                "run_id": run_id,
                "domain": str(dom),
                "n_boot": n_boot,
                "seed": int(seed),
                "ci": [float(ci[0]), float(ci[1])],
                "threshold": float(threshold) if metric_key in {"mcc", "bacc"} else None,
                "mode": str(mode),
                "holdout_sha256": holdout_sha,
                "n_records": n_records,
                "n_groups": n_groups,
            })
            checkpoint_path = progress_dir / (
                f"bootstrap__{metric_key}__{model}__{dom}__{run_id}.pkl"
            )
            loaded = load_checkpoint(
                checkpoint_path,
                fingerprint,
                resume=bool(resume),
                force=bool(force),
            )

            def _write_progress(*, finished: bool, scores: list[float], n_done: int, rng) -> None:
                atomic_pickle(checkpoint_path, {
                    "kind": "bootstrap",
                    "fingerprint": fingerprint,
                    "finished": finished,
                    "scores": scores,
                    "n_done": int(n_done),
                    "rng_state": rng.bit_generator.state,
                    "test_value": float(test_value),
                    "unit": "group_id",
                    "n_records": n_records,
                    "n_groups": n_groups,
                })

            if loaded and loaded.get("finished"):
                scores = [float(v) for v in loaded["scores"]]
                print(
                    f"[BOOT done] status=skipped_completed proj={cfg.PROJECT_NAME} "
                    f"model={model} domain={dom} metric={metric_key} "
                    f"checkpoint={checkpoint_path}"
                )
            else:
                if loaded:
                    scores = [float(v) for v in loaded["scores"]]
                    n_done = int(loaded["n_done"])
                    if n_done > n_boot or n_done < 0 or len(scores) > n_done:
                        raise RuntimeError(
                            f"Bootstrap checkpoint is inconsistent: {checkpoint_path}. "
                            "It was not modified."
                        )
                    rng = np.random.default_rng(seed)
                    rng.bit_generator.state = loaded["rng_state"]
                    print(
                        f"[BOOT resume] proj={cfg.PROJECT_NAME} model={model} domain={dom} "
                        f"metric={metric_key} attempts={n_done}/{n_boot} "
                        f"checkpoint={checkpoint_path}"
                    )
                    status = "resumed"
                else:
                    scores = []
                    n_done = 0
                    rng = np.random.default_rng(seed)
                    status = "ok"
                    _write_progress(finished=False, scores=scores, n_done=n_done, rng=rng)
                    print(
                        f"[BOOT start] proj={cfg.PROJECT_NAME} model={model} domain={dom} "
                        f"metric={metric_key} unit=group_id records={n_records} "
                        f"groups={n_groups} n_boot={n_boot} seed={seed} "
                        f"threshold={threshold:g} force={bool(force)} resume={bool(resume)} "
                        f"checkpoint={checkpoint_path}"
                    )

                started = time.perf_counter()
                if verbose:
                    iterator = trange(
                        n_done,
                        n_boot,
                        desc=f"Bootstrap {metric_label} ({cfg.PROJECT_NAME}/{model}/{dom})",
                        initial=n_done,
                        total=n_boot,
                    )
                else:
                    iterator = range(n_done, n_boot)

                for attempt in iterator:
                    yt, yp = _sample_groups(y_sorted, prob_sorted, row_group, n_groups, rng)
                    n_done = attempt + 1
                    if np.unique(yt).size >= 2:
                        scores.append(_metric_value(metric_key, yt, yp, threshold))
                    if n_done % progress_every == 0 or n_done == n_boot:
                        _write_progress(finished=False, scores=scores, n_done=n_done, rng=rng)
                        if verbose:
                            elapsed = time.perf_counter() - started
                            print(
                                f"[BOOT progress] proj={cfg.PROJECT_NAME} model={model} "
                                f"domain={dom} metric={metric_key} attempts={n_done}/{n_boot} "
                                f"valid={len(scores)} elapsed_seconds={elapsed:.1f} "
                                f"checkpoint={checkpoint_path}"
                            )

                _write_progress(finished=True, scores=scores, n_done=n_done, rng=rng)
                print(
                    f"[BOOT done] status={status} proj={cfg.PROJECT_NAME} model={model} "
                    f"domain={dom} metric={metric_key} valid={len(scores)}/{n_boot} "
                    f"checkpoint={checkpoint_path}"
                )

            if not scores:
                if warn:
                    print(
                        f"[WARN] No valid bootstrap samples for proj={cfg.PROJECT_NAME}, "
                        f"model={model}, domain={dom}."
                    )
                continue

            arr = np.asarray(scores, dtype=float)
            boot_mean = float(arr.mean())
            ci_low, ci_high = (float(v) for v in np.percentile(arr, list(ci)))
            err_low = float(test_value - ci_low)
            err_high = float(ci_high - test_value)

            if verbose:
                extra = f" | thr={threshold:g}" if metric_key in {"mcc", "bacc"} else ""
                print(
                    f"{metric_label} bootstrap | proj={cfg.PROJECT_NAME} | model={model} | "
                    f"run={run_id} | domain={dom} | unit=group_id | "
                    f"records={n_records} groups={n_groups}{extra}"
                )
                print(
                    f"test={test_value:.4f}, {ci[0]}–{ci[1]}% CI=({ci_low:.4f}, {ci_high:.4f}), "
                    f"boot_mean={boot_mean:.4f}, n_valid={arr.size}/{n_boot}"
                )

            rows.append(
                {
                    "project": getattr(cfg, "PROJECT_NAME", None),
                    "model": str(model),
                    "run_id": run_id,
                    "domain": str(dom),
                    "metric": metric_key,
                    "threshold": float(threshold) if metric_key in {"mcc", "bacc"} else np.nan,
                    "test_value": float(test_value),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "boot_mean": boot_mean,
                    "err_low": err_low,
                    "err_high": err_high,
                    "n_boot": int(arr.size),
                    "n_requested": n_boot,
                    "seed": int(seed),
                    "mode": str(mode),
                    "unit": "group_id",
                    "n_records": n_records,
                    "n_groups": n_groups,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        if warn:
            print("[WARN] No bootstrap results produced (empty selection).")
        return summary

    if save:
        out_name = name or f"bootstrap_{metric_key}_summary__proj_{getattr(cfg, 'PROJECT_NAME', 'unknown')}"
        save_table(summary, project_dir=out_project_dir, run_id=save_run_id, name=out_name)

    return summary


def bootstrap_auc_from_holdouts(
    cfg,
    *,
    runs: Mapping[str, str],
    models: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    mode: str = "final",
    n_boot: int = 2000,
    seed: int = 42,
    ci: tuple[float, float] = (2.5, 97.5),
    map_N_to_I: bool = True,
    save: bool = True,
    project_dir: Union[str, Path, None] = None,
    save_run_id: str = "paper",
    name: Optional[str] = None,
    warn: bool = True,
    verbose: bool = True,
    force: bool = False,
    resume: bool = True,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Backwards-compatible wrapper for ROC-AUC bootstrapping."""
    return bootstrap_metric_from_holdouts(
        cfg,
        metric="auc",
        runs=runs,
        models=models,
        domains=domains,
        mode=mode,
        n_boot=n_boot,
        seed=seed,
        ci=ci,
        map_N_to_I=map_N_to_I,
        save=save,
        project_dir=project_dir,
        save_run_id=save_run_id,
        name=name,
        warn=warn,
        verbose=verbose,
        force=force,
        resume=resume,
        progress_every=progress_every,
    )
