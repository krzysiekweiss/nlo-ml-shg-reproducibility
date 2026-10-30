"""Compute and plot classification metrics over probability thresholds."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve

from .io import as_list, filter_df, resolve_run_map, save_table
from .plots_common import MODEL_DISPLAY, build_legend_label, make_plot_name, save_figure


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x)


def _pretty_model(m) -> str:
    if _is_missing(m):
        return "Model"
    key = str(m).lower()
    return MODEL_DISPLAY.get(key, str(m))


def plot_holdout_threshold_curves(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    project: Optional[str] = None,
    mode: str = "final",
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    n_points: int = 101,
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Wrapper: run threshold-curve plots per (model, domain) and save PNG+CSV per plot."""
    models_l = as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    domains_l = as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []

    run_map = resolve_run_map(df, runs=runs, models=models_l)

    out = []
    for m in models_l:
        run_id = run_map[m]
        for dom in domains_l:
            tbl = _plot_holdout_threshold_curves_single(
                df,
                run_id=run_id,
                project=project,
                mode=mode,
                models=m,
                steps=steps,
                domains=dom,
                y_true_col=y_true_col,
                y_prob_col=y_prob_col,
                n_points=n_points,
                title=title,
                save=save,
                project_dir=project_dir,
            )
            if not tbl.empty:
                out.append(tbl)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def holdout_metrics_at_threshold(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    project: Optional[str] = None,
    mode: str = "final",
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    project_dir: str | Path | None = None,
    save: bool = False,
    threshold: float = 0.5,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    title: Optional[str] = None,
) -> pd.DataFrame:
    """Compute MCC/BACC at a fixed threshold per (model, domain) and save CSV per result."""
    models_l = as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    domains_l = as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []

    run_map = resolve_run_map(df, runs=runs, models=models_l)

    out = []
    for m in models_l:
        run_id = run_map[m]
        for dom in domains_l:
            g = filter_df(
                df,
                run_id=run_id,
                project=project,
                mode=mode,
                models=m,
                steps=steps,
                domains=dom,
            )
            if g.empty:
                continue

            tbl = _holdout_metrics_at_threshold_single(
                g,
                threshold=threshold,
                y_true_col=y_true_col,
                y_prob_col=y_prob_col,
            )

            if tbl.empty:
                continue

            if save:
                name = make_plot_name(
                    kind="holdout_metrics_at_threshold",
                    run_id=run_id,
                    mode=mode,
                    models=m,
                    steps=steps,
                    domains=dom,
                )
                save_table(
                    tbl,
                    project_dir=project_dir,
                    run_id=run_id,
                    name=name,
                )

            out.append(tbl)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_threshold_metrics(
    y_true,
    y_prob,
    *,
    targets=None,
    target_recall: float = 0.8,
    title: Optional[str] = None,
    show_legend: bool = True,
):
    """Plot precision/recall/F1 vs threshold for a single (y_true, y_prob) pair.

    Returns (fig, ax, stats_dict) where stats_dict contains:
      - threshold (for the first target)
      - precision
      - recall
      - f1
      - targets: dict with per-target thresholds and metric values
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve returns len(precision) = len(recall) = len(thresholds) + 1
    # align by dropping the last precision/recall point
    precision = precision[:-1]
    recall = recall[:-1]

    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    if targets is None:
        # Backwards-compatible default: pick threshold closest to target recall.
        targets = [("recall", float(target_recall))]

    metric_series = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Resolve target thresholds: pick threshold index whose metric is closest to desired value.
    targets_stats = {}
    for met_name, met_value in targets:
        key = str(met_name).strip().lower()
        if key not in metric_series:
            raise ValueError(f"Unsupported target metric '{met_name}'. Choose from: 'recall', 'precision', 'f1'.")
        idx = int(np.argmin(np.abs(metric_series[key] - float(met_value))))
        thr = float(thresholds[idx])
        targets_stats[key] = {
            "target": float(met_value),
            "threshold": thr,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
        }

    # Use the first target as the primary one for title + returned top-level fields.
    primary_key = str(targets[0][0]).strip().lower()
    best_thr = float(targets_stats[primary_key]["threshold"])
    best_prec = float(targets_stats[primary_key]["precision"])
    best_rec = float(targets_stats[primary_key]["recall"])
    best_f1 = float(targets_stats[primary_key]["f1"])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    line_prec, = ax.plot(thresholds, precision, label="Precision", linewidth=2)
    line_rec, = ax.plot(thresholds, recall, label="Recall", linewidth=2)
    line_f1, = ax.plot(thresholds, f1, label="F1 score", linewidth=2)

    # Vertical lines for each target (e.g. recall=0.8, precision=0.8)
    metric_line = {"precision": line_prec, "recall": line_rec, "f1": line_f1}
    for k, st in targets_stats.items():
        ax.axvline(
            st["threshold"],
            linestyle="--",
            linewidth=1.5,
            color=getattr(metric_line[k], "get_color", lambda: None)(),
            alpha=0.9,
        )
        # Mark the corresponding metric value at that threshold.
        ax.scatter([st["threshold"]], [st[k]], s=40, zorder=3)

    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Metric value")
    if title is None:
        ax.set_title(f"Threshold analysis | thr ≈ {best_thr:.2f}")
    else:
        ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # optional legend: can be disabled when using a shared legend figure
    if show_legend:
        ax.legend(
            handles=[line_prec, line_rec, line_f1],
            labels=[
                f"Precision (≈ {best_prec:.2f})",
                f"Recall (≈ {best_rec:.2f})",
                f"F1 score (≈ {best_f1:.2f})",
            ],
            frameon=False,
            loc="lower left",
        )
    ax.grid(True, alpha=0.25)

    # Reserve space for threshold annotations.
    plt.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])
    return fig, ax, {
        "threshold": float(best_thr),
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "targets": targets_stats,
    }


def save_threshold_legend(
    project_dir: Union[str, Path],
    *,
    run_id: str = "paper",
    name: str = "threshold_metrics_legend",
) -> None:
    """Create and save a standalone legend figure for threshold metrics.

    This draws three colored entries (Precision, Recall, F1 score) matching the default
    color cycle and saves the figure under <project_dir>/results/<run_id>/figures/<name>.png.
    """
    project_dir = Path(project_dir)
    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.axis("off")

    # Use the default matplotlib color cycle so colors match plot_threshold_metrics.
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    colors = [
        color_cycle[0] if len(color_cycle) > 0 else "C0",
        color_cycle[1] if len(color_cycle) > 1 else "C1",
        color_cycle[2] if len(color_cycle) > 2 else "C2",
    ]

    line_prec, = ax.plot([], [], color=colors[0], label="Precision")
    line_rec, = ax.plot([], [], color=colors[1], label="Recall")
    line_f1, = ax.plot([], [], color=colors[2], label="F1 score")

    ax.legend(
        handles=[line_prec, line_rec, line_f1],
        labels=["Precision", "Recall", "F1 score"],
        loc="center",
        frameon=False,
        ncol=3,
    )
    fig.tight_layout()

    from .plots_common import save_figure
    plt.show()
    save_figure(fig, project_dir=project_dir, run_id=run_id, name=name, close=True)


def _plot_holdout_threshold_curves_single(
    df: pd.DataFrame,
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    mode: Optional[str] = "final",
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    n_points: int = 101,
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Plot MCC and BACC vs threshold for selected groups."""
    g = filter_df(
        df,
        run_id=run_id,
        project=project,
        mode=mode,
        models=models,
        steps=steps,
        domains=domains,
    )
    if g.empty:
        print("[INFO] Empty selection. Nothing to plot.")
        return pd.DataFrame()

    def _infer_domain_text() -> str:
        if isinstance(domains, str):
            return domains
        if domains is not None:
            dd = list(domains)
            return str(dd[0]) if len(dd) == 1 else "multiple"
        if "domain" in g.columns:
            u = [x for x in g["domain"].dropna().unique().tolist() if str(x).strip() != ""]
            if len(u) == 1:
                return str(u[0])
            if len(u) > 1:
                return "multiple"
        return "ALL"

    rid = run_id or (str(g["run_id"].dropna().unique()[0]) if "run_id" in g.columns and g["run_id"].nunique(dropna=True) == 1 else "unknown_run")

    if title is not None:
        base_title = title
    else:
        base_title = f"Holdout threshold curves | run_id={rid} | domain={_infer_domain_text()}"

    group_cols = [c for c in ["model", "step", "domain"] if c in g.columns]
    if models is not None:
        group_cols = [c for c in group_cols if c != "model"]
    if steps is not None:
        group_cols = [c for c in group_cols if c != "step"]
    if domains is not None:
        group_cols = [c for c in group_cols if c != "domain"]

    thresholds = np.linspace(0.0, 1.0, int(n_points))
    rows = []

    fig = plt.figure()
    for key, gg in g.groupby(group_cols, dropna=False) if group_cols else [(("ALL",), g)]:
        if not isinstance(key, tuple):
            key = (key,)
        label = build_legend_label(gg, auc=None, show_domain=False)

        y_true = gg[y_true_col].astype(int).values
        y_prob = gg[y_prob_col].astype(float).values

        if len(np.unique(y_true)) < 2:
            print(f"[WARN] Skipping {label}: only one class present.")
            continue

        mcc_vals = []
        bacc_vals = []
        for thr in thresholds:
            y_hat = (y_prob >= thr).astype(int)
            mcc_vals.append(float(matthews_corrcoef(y_true, y_hat)))
            bacc_vals.append(float(balanced_accuracy_score(y_true, y_hat)))
            rows.append({
                **(
                    {c: v for c, v in zip(group_cols, key)}
                    if group_cols
                    else {"group": "ALL"}
                ),
                "threshold": float(thr),
                "mcc": float(mcc_vals[-1]),
                "bacc": float(bacc_vals[-1]),
                "n": int(len(gg)),
            })

        plt.plot(thresholds, mcc_vals, label=label)

    plt.xlabel("Threshold")
    plt.ylabel("MCC")
    plt.title(f"{base_title} | MCC") if base_title else plt.title(None)
    plt.legend()
    plt.tight_layout()
    if save:
        name = make_plot_name(
            kind="mcc_threshold",
            run_id=run_id,
            mode=mode,
            models=models,
            steps=steps,
            domains=domains,
        )
        print(f"Saving plot: {name}, run: {run_id}, in dir: {project_dir}")
        save_figure(
            fig,
            project_dir=project_dir,
            run_id=run_id,
            name=name,
            close=False,
        )
        save_table(
            pd.DataFrame(rows),
            project_dir=project_dir,
            run_id=run_id,
            name=name,
        )
    plt.show()

    fig1 = plt.figure()
    for key, gg in g.groupby(group_cols, dropna=False) if group_cols else [(("ALL",), g)]:
        if not isinstance(key, tuple):
            key = (key,)
        label = build_legend_label(gg, auc=None, show_domain=False)
        y_true = gg[y_true_col].astype(int).values
        y_prob = gg[y_prob_col].astype(float).values
        if len(np.unique(y_true)) < 2:
            continue

        bacc_vals = []
        for thr in thresholds:
            y_hat = (y_prob >= thr).astype(int)
            bacc_vals.append(float(balanced_accuracy_score(y_true, y_hat)))

        plt.plot(thresholds, bacc_vals, label=label)

    plt.xlabel("Threshold")
    plt.ylabel("Balanced accuracy")
    plt.title(f"{base_title} | BACC") if base_title else plt.title(None)
    plt.legend()
    plt.tight_layout()
    if save:
        name = make_plot_name(
            kind="bacc_threshold",
            run_id=run_id,
            mode=mode,
            models=models,
            steps=steps,
            domains=domains,
        )
        print(f"Saving plot: {name}, run: {run_id}, in dir: {project_dir}")
        save_figure(
            fig1,
            project_dir=project_dir,
            run_id=run_id,
            name=name,
            close=False,
        )
        save_table(
            pd.DataFrame(rows),
            project_dir=project_dir,
            run_id=run_id,
            name=name,
        )
    plt.show()

    return pd.DataFrame(rows)


def _holdout_metrics_at_threshold_single(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute MCC and BACC at a fixed threshold per group."""
    if df.empty:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = [c for c in ["run_id", "project", "mode", "model", "step", "domain"] if c in df.columns]

    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = {c: v for c, v in zip(group_cols, key)}

        y_true = g[y_true_col].astype(int).values
        y_prob = g[y_prob_col].astype(float).values

        if len(np.unique(y_true)) < 2:
            continue

        y_hat = (y_prob >= threshold).astype(int)

        rows.append({
            **meta,
            "n": int(len(g)),
            "mcc": float(matthews_corrcoef(y_true, y_hat)),
            "bacc": float(balanced_accuracy_score(y_true, y_hat)),
            "threshold": float(threshold),
        })

    return pd.DataFrame(rows)
