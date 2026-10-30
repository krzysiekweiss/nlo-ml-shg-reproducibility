"""Compute and plot predictive performance for element-containing subsets."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from .chemistry import ELEMENT_Z, element_category, category_color_map
from .io import as_list, resolve_run_map
from .plots_common import save_figure


# Per-element tables (AUC / metric vs Z)


def per_element_auc_table_simple(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    min_n: int = 30,
) -> pd.DataFrame:
    """Compute per-element ROC-AUC (elements_col = comma-separated string)."""
    if df.empty:
        return pd.DataFrame(columns=["element", "n", "la", "auc"])
    for c in (y_true_col, y_prob_col, elements_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'.")
    y_true_all = df[y_true_col].astype(int).values
    y_prob_all = df[y_prob_col].astype(float).values
    elem_masks: dict[str, np.ndarray] = {}
    vals = df[elements_col].astype(str).values
    n_rows = len(df)
    for i in range(n_rows):
        s = vals[i]
        if not s:
            continue
        elems = [e.strip() for e in s.split(",") if e.strip()]
        for e in set(elems):
            if e not in elem_masks:
                elem_masks[e] = np.zeros(n_rows, dtype=bool)
            elem_masks[e][i] = True
    rows = []
    for e, m in elem_masks.items():
        n = int(m.sum())
        if n < min_n:
            continue
        yt = y_true_all[m]
        if len(np.unique(yt)) < 2:
            continue
        yp = y_prob_all[m]
        auc = float(roc_auc_score(yt, yp))
        rows.append({"element": e, "n": n, "la": float(np.log10(n)), "auc": auc})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["auc", "n"], ascending=[False, False]).reset_index(drop=True)


def per_element_auc_table_Z(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    min_n: int = 30,
) -> pd.DataFrame:
    """Compute per-element ROC-AUC with X = atomic number Z."""
    if df.empty:
        return pd.DataFrame(columns=["element", "Z", "n", "auc"])
    y_true_all = df[y_true_col].astype(int).values
    y_prob_all = df[y_prob_col].astype(float).values
    elems_all = df[elements_col].astype(str).values
    n_rows = len(df)
    elem_masks: dict[str, np.ndarray] = {}
    for i in range(n_rows):
        elems = [e.strip() for e in elems_all[i].split(",") if e.strip()]
        for e in set(elems):
            if e not in ELEMENT_Z:
                continue
            if e not in elem_masks:
                elem_masks[e] = np.zeros(n_rows, dtype=bool)
            elem_masks[e][i] = True
    rows = []
    for e, m in elem_masks.items():
        n = int(m.sum())
        if n < min_n:
            continue
        yt = y_true_all[m]
        if len(np.unique(yt)) < 2:
            continue
        auc = float(roc_auc_score(yt, y_prob_all[m]))
        rows.append({"element": e, "Z": ELEMENT_Z[e], "n": n, "auc": auc})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Z").reset_index(drop=True)


def per_element_metric_table_Z(
    df: pd.DataFrame,
    *,
    metric: str,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
) -> pd.DataFrame:
    """Compute per-element metric with X = atomic number Z. Metrics: auc, mcc, bacc."""
    if df.empty:
        return pd.DataFrame(columns=["element", "Z", "n", metric])
    y_true_all = df[y_true_col].astype(int).values
    y_prob_all = df[y_prob_col].astype(float).values
    elems_all = df[elements_col].astype(str).values
    n_rows = len(df)
    elem_masks: dict[str, np.ndarray] = {}
    for i in range(n_rows):
        elems = [e.strip() for e in elems_all[i].split(",") if e.strip()]
        for e in set(elems):
            if e not in ELEMENT_Z:
                continue
            if e not in elem_masks:
                elem_masks[e] = np.zeros(n_rows, dtype=bool)
            elem_masks[e][i] = True
    rows = []
    for e, m in elem_masks.items():
        n = int(m.sum())
        if n < min_n:
            continue
        yt = y_true_all[m]
        yp = y_prob_all[m]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "auc":
            val = roc_auc_score(yt, yp)
        elif metric == "mcc":
            y_hat = (yp >= threshold).astype(int)
            val = matthews_corrcoef(yt, y_hat)
        elif metric == "bacc":
            y_hat = (yp >= threshold).astype(int)
            val = balanced_accuracy_score(yt, y_hat)
        else:
            raise ValueError("metric must be one of: 'auc', 'mcc', 'bacc'")
        rows.append({"element": e, "Z": ELEMENT_Z[e], "n": n, metric: float(val)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Z").reset_index(drop=True)


# Plots: AUC vs LA, AUC vs Z, metric vs Z


def plot_auc_vs_la_simple(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    min_n: int = 30,
    annotate_top_k: int = 15,
    title: Optional[str] = None,
) -> pd.DataFrame:
    """Plot per-element ROC-AUC (Y) vs LA=log10(abundance) (X)."""
    tbl = per_element_auc_table_simple(
        df, y_true_col=y_true_col, y_prob_col=y_prob_col,
        elements_col=elements_col, min_n=min_n,
    )
    if tbl.empty:
        print("[INFO] No elements met min_n and two-class requirement.")
        return tbl
    fig, ax = plt.subplots()
    ax.scatter(tbl["la"].values, tbl["auc"].values)
    if annotate_top_k and annotate_top_k > 0:
        for _, r in tbl.head(annotate_top_k).iterrows():
            ax.text(r["la"], r["auc"], str(r["element"]))
    ax.set_xlabel("LA = log10(#samples containing element)")
    ax.set_ylabel("ROC-AUC on element subset")
    ax.set_title(title or "Per-element ROC-AUC vs LA")
    fig.tight_layout()
    plt.show()
    return tbl


def plot_auc_vs_Z(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    min_n: int = 30,
    annotate: bool = True,
    title: Optional[str] = None,
) -> pd.DataFrame:
    """Scatter: X = atomic number Z, Y = per-element ROC-AUC."""
    tbl = per_element_auc_table_Z(
        df, y_true_col=y_true_col, y_prob_col=y_prob_col,
        elements_col=elements_col, min_n=min_n,
    )
    if tbl.empty:
        print("[INFO] No elements met min_n and two-class requirement.")
        return tbl
    fig, ax = plt.subplots()
    ax.scatter(tbl["Z"], tbl["auc"])
    if annotate:
        for _, r in tbl.iterrows():
            ax.text(r["Z"], r["auc"], r["element"], fontsize=9)
    ax.set_xlabel("Atomic number Z")
    ax.set_ylabel("ROC-AUC (per-element subset)")
    ax.set_title(title or "Per-element ROC-AUC vs atomic number Z")
    fig.tight_layout()
    plt.show()
    return tbl


def plot_metric_vs_Z(
    df: pd.DataFrame,
    *,
    metric: str,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    annotate: bool = True,
    title: Optional[str] = None,
    save: bool = False,
    project_dir: Union[str, Path, None] = None,
    run_id: Optional[str] = None,
    show_legend: bool = True,
) -> pd.DataFrame:
    """Scatter: X = atomic number Z, Y = selected metric (AUC / MCC / BACC). Colors by element category."""
    tbl = per_element_metric_table_Z(
        df, metric=metric, y_true_col=y_true_col, y_prob_col=y_prob_col,
        elements_col=elements_col, threshold=threshold, min_n=min_n,
    )
    if tbl.empty:
        print("[INFO] No elements met min_n and two-class requirement.")
        return tbl
    tbl = tbl.copy()
    tbl["category"] = tbl["element"].map(element_category)
    cmap = category_color_map()
    tbl["color"] = tbl["category"].map(lambda c: cmap.get(c, "#333333"))
    fig, ax = plt.subplots(figsize=(8, 4))
    for cat, gg in tbl.groupby("category", sort=True):
        ax.scatter(gg["Z"], gg[metric], label=cat, c=gg["color"].iloc[0])
    if annotate:
        for _, r in tbl.iterrows():
            ax.annotate(
                r["element"],
                xy=(r["Z"], r[metric]),
                xytext=(1, 1),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9,
            )
    ylabel_map = {"auc": "ROC-AUC", "mcc": "MCC", "bacc": "Balanced accuracy"}
    ax.set_xlabel("Atomic number Z")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    if title is None:
        ax.set_title(f"Per-element {ylabel_map.get(metric, metric)} vs atomic number Z")
    elif title:
        ax.set_title(title)
    else:
        ax.set_title("")
    # Small top margin above y=1 so points at 1 are fully visible (no clipping)
    ax.set_ylim(0.5, 1.02)
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if save and project_dir is not None:
        parts = [f"metric_{metric}_vs_Z"]
        if "model" in df.columns and df["model"].nunique() == 1:
            parts.append(f"model_{df['model'].iloc[0]}")
        if "domain" in df.columns and df["domain"].nunique() == 1:
            parts.append(f"domain_{df['domain'].iloc[0]}")
        if not annotate:
            parts.append("no_labels")
        name = "__".join(parts)
        save_figure(fig, project_dir=project_dir, run_id=run_id or "paper", name=name, close=False)
    # Reserve consistent right margin for potential legend so that
    # the data area has the same width whether the legend is shown or not.
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
    return tbl


def plot_metric_vs_Z_by_domain(
    df: pd.DataFrame,
    *,
    domain_col: str = "domain",
    domains: Optional[Union[list, tuple]] = None,
    metric: str = "auc",
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    annotate: bool = True,
    title_prefix: Optional[str] = None,
    save: bool = False,
    project_dir: Union[str, Path, None] = None,
    run_id: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Run plot_metric_vs_Z separately for each domain. Returns dict domain -> table."""
    if df.empty:
        return {}
    if domain_col not in df.columns:
        raise KeyError(f"Missing required column '{domain_col}' in df.")
    dom_list = sorted([d for d in df[domain_col].dropna().unique()]) if domains is None else list(domains)
    out: dict[str, pd.DataFrame] = {}
    for dom in dom_list:
        sub = df[df[domain_col] == dom]
        if sub.empty:
            continue
        title = f"{metric.upper()} vs Z | domain={dom}" if title_prefix is None else f"{title_prefix} | domain={dom}"
        tbl = plot_metric_vs_Z(
            sub,
            metric=metric,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            elements_col=elements_col,
            threshold=threshold,
            min_n=min_n,
            annotate=annotate,
            title=title,
            save=save,
            project_dir=project_dir,
            run_id=run_id,
        )
        out[str(dom)] = tbl
    return out


def _plot_metric_vs_Z_by_run_and_domain_single(
    df: pd.DataFrame,
    *,
    run_col: str = "run_id",
    domain_col: str = "domain",
    runs: Optional[Union[list, tuple]] = None,
    domains: Optional[Union[list, tuple]] = None,
    metric: str = "auc",
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    annotate: bool = True,
    title: Optional[str] = None,
    title_prefix: Optional[str] = None,
    save: bool = False,
    project_dir: Union[str, Path, None] = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Run plot_metric_vs_Z for each (run, domain). Returns dict (run_id, domain) -> table."""
    if df.empty:
        return {}
    for col in (run_col, domain_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in df.")
    run_list = sorted(df[run_col].dropna().unique()) if runs is None else list(runs)
    domain_list = sorted(df[domain_col].dropna().unique()) if domains is None else list(domains)
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for run in run_list:
        df_run = df[df[run_col] == run]
        if df_run.empty:
            continue
        for dom in domain_list:
            sub = df_run[df_run[domain_col] == dom]
            if sub.empty:
                continue
            if title is not None:
                plot_title = title
            elif title_prefix:
                plot_title = f"{title_prefix} | domain={dom}"
            else:
                plot_title = f"{metric.upper()} vs Z | run={run} | domain={dom}"
            tbl = plot_metric_vs_Z(
                sub,
                metric=metric,
                y_true_col=y_true_col,
                y_prob_col=y_prob_col,
                elements_col=elements_col,
                threshold=threshold,
                min_n=min_n,
                title=plot_title,
                run_id=str(run),
                save=save,
                project_dir=project_dir,
                annotate=annotate,
                show_legend=False,
            )
            out[(str(run), str(dom))] = tbl
    return out


def plot_metric_vs_Z_by_run_and_domain(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    models: Optional[Union[str, list]] = None,
    domains: Optional[Union[str, list]] = None,
    project_dir: Union[str, Path, None] = None,
    save: bool = False,
    metric: str = "auc",
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    annotate: bool = True,
    title: Optional[str] = None,
    title_prefix: Optional[str] = None,
    show_legend: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Wrapper: run per-element metric vs Z per (model, domain). Accepts runs=dict, models=list."""
    models_l = as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    domains_l = as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []
    run_map = resolve_run_map(df, runs=runs, models=models_l)
    run_list = [run_map[m] for m in models_l]
    return _plot_metric_vs_Z_by_run_and_domain_single(
        df,
        runs=run_list,
        domains=domains_l,
        metric=metric,
        y_true_col=y_true_col,
        y_prob_col=y_prob_col,
        elements_col=elements_col,
        threshold=threshold,
        min_n=min_n,
        annotate=annotate,
        title=title,
        title_prefix=title_prefix,
        save=save,
        project_dir=project_dir,
    )


# Overlay: all domains on one plot per run


_DOMAIN_OVERLAY_COLORS = {"I": "#3182bd", "M": "#deebf7", "O": "#9ecae1", "N": "#3182bd"}


def plot_metric_vs_Z_legend(
    *,
    project_dir: Union[str, Path, None] = None,
    run_id: str = "paper",
    name: str = "metric_vs_Z_legend",
    save: bool = False,
):
    """Create a standalone legend figure for metric-vs-Z plots (element categories).

    Uses category_color_map() to build marker-only legend entries that match
    the scatter style used in plot_metric_vs_Z (circles with solid facecolor).
    """
    from matplotlib.lines import Line2D

    cmap = category_color_map()
    # Stable ordering by category name
    items = sorted(cmap.items(), key=lambda kv: kv[0])

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="none",
        )
        for _, color in items
    ]
    labels = [label for label, _ in items]

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.axis("off")
    ax.legend(handles, labels, loc="center", ncol=3, frameon=False)
    fig.tight_layout()

    if save and project_dir is not None:
        save_figure(fig, project_dir=project_dir, run_id=run_id, name=name, close=False)
    return fig


def plot_metric_vs_Z_overlay_domains(
    df: pd.DataFrame,
    *,
    run_id: str,
    run_col: str = "run_id",
    domain_col: str = "domain",
    domains: Optional[Union[list, tuple]] = None,
    metric: str = "auc",
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    annotate: bool = False,
    title: Optional[str] = None,
    save: bool = False,
    project_dir: Union[str, Path, None] = None,
    model_name: Optional[str] = None,
    title_prefix: Optional[str] = None,
    show_legend: bool = True,
) -> pd.DataFrame:
    """Plot metric vs Z with all domains on one figure for a single run."""
    if df.empty:
        return pd.DataFrame()
    if run_col not in df.columns or domain_col not in df.columns:
        raise KeyError(f"df must have columns '{run_col}' and '{domain_col}'.")
    df_run = df[df[run_col].astype(str) == str(run_id)]
    if df_run.empty:
        print(f"[INFO] No rows for run_id={run_id}.")
        return pd.DataFrame()
    dom_list = (
        sorted(df_run[domain_col].dropna().unique().astype(str).tolist())
        if domains is None
        else [str(d) for d in domains]
    )
    frames = []
    for dom in dom_list:
        sub = df_run[df_run[domain_col].astype(str) == str(dom)]
        if sub.empty:
            continue
        tbl = per_element_metric_table_Z(
            sub, metric=metric, y_true_col=y_true_col, y_prob_col=y_prob_col,
            elements_col=elements_col, threshold=threshold, min_n=min_n,
        )
        if tbl.empty:
            continue
        tbl = tbl.copy()
        tbl[domain_col] = dom
        frames.append(tbl)
    if not frames:
        print("[INFO] No per-element data for any domain.")
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    domain_colors = {d: _DOMAIN_OVERLAY_COLORS.get(d, "#333333") for d in dom_list}
    fig, ax = plt.subplots(figsize=(8, 4))
    for dom in dom_list:
        gg = combined[combined[domain_col] == dom]
        if gg.empty:
            continue
        ax.scatter(gg["Z"], gg[metric], label=f"domain {dom}", c=domain_colors[dom], alpha=0.85, s=40)
    if annotate:
        for _, r in combined.iterrows():
            ax.annotate(f"{r['element']}({r[domain_col]})", xy=(r["Z"], r[metric]),
                        xytext=(1, 1), textcoords="offset points", ha="left", va="bottom", fontsize=7)
    ylabel_map = {"auc": "ROC-AUC", "mcc": "MCC", "bacc": "Balanced accuracy"}
    ax.set_xlabel("Atomic number Z")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    title_str = (title or f"{title_prefix or ''} {metric.upper()} vs Z | run={run_id} | all domains".strip())
    if model_name:
        title_str = f"{title_str} | model={model_name}"
    ax.set_title(title_str)
    # Small top margin above y=1 so points at 1 are fully visible (no clipping)
    ax.set_ylim(0.5, 1.02)
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    if save and project_dir is not None:
        parts = [f"metric_{metric}_vs_Z", "overlay_domains"]
        if model_name:
            parts.insert(1, f"model_{model_name}")
        name = "__".join(parts)
        save_figure(fig, project_dir=project_dir, run_id=run_id, name=name, close=False)
    plt.show()
    return combined


def plot_metric_vs_Z_overlay_domains_by_run(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    models: Optional[Union[str, list]] = None,
    domains: Optional[Union[str, list]] = None,
    project_dir: Union[str, Path, None] = None,
    save: bool = False,
    metric: str = "auc",
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    annotate: bool = False,
    title_prefix: Optional[str] = None,
    show_legend: bool = True,
) -> dict[str, pd.DataFrame]:
    """One overlay plot (all domains on one figure) per run."""
    models_l = as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    domains_l = as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []
    run_map = resolve_run_map(df, runs=runs, models=models_l)
    out: dict[str, pd.DataFrame] = {}
    for model_name, run_id in run_map.items():
        tbl = plot_metric_vs_Z_overlay_domains(
            df,
            run_id=run_id,
            domains=domains_l,
            metric=metric,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            elements_col=elements_col,
            threshold=threshold,
            min_n=min_n,
            annotate=annotate,
            save=save,
            project_dir=project_dir,
            model_name=model_name,
            title_prefix=title_prefix,
            show_legend=show_legend,
        )
        if not tbl.empty:
            out[str(run_id)] = tbl
    return out
