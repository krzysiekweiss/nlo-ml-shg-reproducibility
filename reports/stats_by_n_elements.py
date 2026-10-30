"""Summarize predictive performance by the number of distinct chemical elements."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from .io import as_list, resolve_run_map, save_table
from .plots_common import MODEL_DISPLAY, make_plot_name, save_figure, domain_color_palette, domain_line_color

ELEMENT_BINS = ["1-2", "3", "4", "5+"]
BIN_X = {name: index for index, name in enumerate(ELEMENT_BINS)}
# Domain O is the lightest step of the shared gradients and disappears on white.
# These overrides are used only by the element-count figures.
_O_LINE_COLOR = {
    "rf": "#2171b5",
    "lgbm": "#d94801",
}
_O_BAR_COLOR = "#43a047"


def _overlay_line_style(model: str):
    key = str(model).lower()
    if key == "lgbm":
        return "-"
    if key == "rf":
        return "--"
    return "-"


def _overlay_line_color(model: str, project: str | None, domain: str) -> str:
    if str(domain).strip().upper() == "O":
        override = _O_LINE_COLOR.get(str(model).lower())
        if override is not None:
            return override
    return domain_line_color(model, project, domain)


def _place_legend_below(fig, handles, labels):
    """Put the legend under the axes, in columns, and keep it inside the saved figure."""
    ncol = 3 if len(labels) >= 3 else max(len(labels), 1)
    fig.set_size_inches(6.8, 5.6)
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.8,
        columnspacing=1.4,
    )
    fig.canvas.draw()
    legend_height = legend.get_window_extent().height / fig.bbox.height
    # Keep the legend inside the figure so bbox_inches="tight" does not crop it.
    fig.tight_layout(rect=(0.0, legend_height + 0.04, 1.0, 1.0))
    return legend


def _element_bin(series: pd.Series) -> pd.Series:
    def n_unique(value: str) -> int:
        elems = [item.strip() for item in str(value).split(",") if item.strip()]
        return len(set(elems))

    def to_bin(count: int) -> str:
        if count <= 2:
            return "1-2"
        if count == 3:
            return "3"
        if count == 4:
            return "4"
        return "5+"

    return series.map(n_unique).map(to_bin)


def _bin_counts(df: pd.DataFrame, elements_col: str = "elements_cif") -> pd.Series:
    counts = pd.Series(0, index=ELEMENT_BINS, dtype=int)
    if df.empty or elements_col not in df.columns:
        return counts
    observed = _element_bin(df[elements_col]).value_counts()
    for name in ELEMENT_BINS:
        counts[name] = int(observed.get(name, 0))
    return counts


def _agreed_bin_counts(
    df: pd.DataFrame,
    *,
    source: str,
    domain: str,
    elements_col: str = "elements_cif",
    models: Optional[list[str]] = None,
) -> pd.Series:
    """One count per bin. Models must agree; their rows are not added together."""
    present = {
        str(model): _bin_counts(group, elements_col)
        for model, group in df.groupby("model", dropna=False)
    } if "model" in df.columns else {}
    names = list(models) if models else list(present)
    if len(names) <= 1:
        if names and names[0] in present:
            return present[names[0]]
        return _bin_counts(df, elements_col)
    zero = pd.Series(0, index=ELEMENT_BINS, dtype=int)
    per_model = {name: present.get(name, zero.copy()) for name in names}
    base_name = names[0]
    base = per_model[base_name]
    for name in names[1:]:
        other = per_model[name]
        mismatch = [bin_name for bin_name in ELEMENT_BINS if int(base[bin_name]) != int(other[bin_name])]
        if mismatch:
            detail = ", ".join(
                f"{bin_name}: {base_name}={int(base[bin_name])} {name}={int(other[bin_name])}"
                for bin_name in mismatch
            )
            raise RuntimeError(
                f"Element-bin counts differ between models for source={source} domain={domain}: {detail}"
            )
    return base


def _align_to_element_bins(tbl: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Place metrics on the shared bin axis. A missing bin stays NaN, not zero."""
    aligned = tbl.set_index("bin").reindex(ELEMENT_BINS) if not tbl.empty else pd.DataFrame(index=ELEMENT_BINS)
    for column in metric_cols:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned


def stats_by_n_elements(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
) -> pd.DataFrame:
    """
    Compute holdout metrics grouped by the number of unique elements in 'elements_cif'.
    Bins: 1-2, 3, 4, 5+
    """
    if df.empty:
        return pd.DataFrame(columns=["bin", "n", "auc", "bacc", "mcc"])

    for c in (y_true_col, y_prob_col, elements_col):
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}'.")

    def n_unique_elements(s: str) -> int:
        elems = [e.strip() for e in str(s).split(",") if e.strip()]
        return len(set(elems))

    def to_bin(k: int) -> str:
        if k <= 2:
            return "1-2"
        if k == 3:
            return "3"
        if k == 4:
            return "4"
        return "5+"

    n_el = df[elements_col].map(n_unique_elements)
    bins = n_el.map(to_bin)
    g = df.copy()
    g["_bin"] = bins

    rows = []
    for b, gg in g.groupby("_bin", dropna=False):
        if len(gg) < min_n:
            continue

        y_true = gg[y_true_col].astype(int).values
        y_prob = gg[y_prob_col].astype(float).values

        if len(np.unique(y_true)) < 2:
            continue

        y_hat = (y_prob >= threshold).astype(int)

        rows.append({
            "bin": b,
            "n": int(len(gg)),
            "auc": float(roc_auc_score(y_true, y_prob)),
            "bacc": float(balanced_accuracy_score(y_true, y_hat)),
            "mcc": float(matthews_corrcoef(y_true, y_hat)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    order = pd.Categorical(out["bin"], categories=["1-2", "3", "4", "5+"], ordered=True)
    out = out.assign(_ord=order).sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    return out


def infer_ylim_stats_by_n_elements(
    df: pd.DataFrame,
    *,
    metric_cols: list[str],
    y_true_col: str,
    y_prob_col: str,
    elements_col: str,
    threshold: float,
    min_n: int,
    dataset_col: str = "source",
    datasets: list[str] | None = None,
) -> tuple[float, float] | None:
    if df.empty:
        return None

    if datasets is not None and dataset_col in df.columns:
        df = df[df[dataset_col].astype(str).isin([str(x) for x in datasets])]

    vals = []
    group_cols = [c for c in ["model", "domain"] if c in df.columns]
    if group_cols:
        it = (gg for _, gg in df.groupby(group_cols))
    else:
        it = [df]

    for gg in it:
        tbl = stats_by_n_elements(
            gg,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            elements_col=elements_col,
            threshold=threshold,
            min_n=min_n,
        )
        if tbl is None or tbl.empty:
            continue
        for met in metric_cols:
            if met in tbl.columns:
                v = tbl[met].dropna().to_numpy()
                if v.size:
                    vals.append(v)

    if not vals:
        return None

    y = np.concatenate(vals)
    lo = float(np.min(y))
    hi = float(np.max(y))
    if lo == hi:
        return (lo - 0.01, hi + 0.01)

    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)


def _plot_stats_by_n_elements_single(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    metrics: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
    run_id: str | None = None,
    domain: str | None = None,
    overlay: bool = False,
    ylim=None,
    show_hist: bool = False,
    show_legend: bool = True,
    hist_palette: str | None = None,
) -> pd.DataFrame:
    """Plot metrics vs element-count bins (1-2 / 3-4 / 5+). Returns aggregated table.
    hist_palette: optional 'mp'|'cod'|'green' to force histogram bar colors."""
    if domain is not None and "domain" in df.columns:
        df = df[df["domain"].astype(str) == str(domain)]

    if df.empty:
        print("[INFO] Empty selection. Nothing to plot.")
        return pd.DataFrame()

    metric_cols = ["auc", "bacc", "mcc"]
    if metrics is not None:
        metrics_l = [m.lower() for m in metrics]
        metric_cols = [m for m in metric_cols if m in metrics_l]
    if not metric_cols:
        raise ValueError("No valid metrics selected. Choose from: 'auc', 'bacc', 'mcc'.")

    def _infer_single_value(frame: pd.DataFrame, col: str) -> Optional[str]:
        if col in frame.columns:
            u = [x for x in frame[col].dropna().unique().tolist() if str(x).strip() != ""]
            if len(u) == 1:
                return str(u[0])
        return None

    def _display_model(m: Optional[str]) -> Optional[str]:
        if not m:
            return None
        key = str(m).strip().lower()
        return MODEL_DISPLAY.get(key, str(m))

    metric_marker = {"auc": "o", "bacc": "s", "mcc": "^"}
    metric_label = {"auc": "AUC", "bacc": "BACC", "mcc": "MCC"}

    rid = run_id or _infer_single_value(df, "run_id") or "unknown_run"
    domain_used = domain or _infer_single_value(df, "domain") or ("ALL" if not overlay else "MULTI")

    fig, ax = plt.subplots()
    ax_hist = None
    rows_all = []

    if not overlay:
        tbl = stats_by_n_elements(
            df,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            elements_col=elements_col,
            threshold=threshold,
            min_n=min_n,
        )
        if tbl.empty:
            print("[INFO] No bins met min_n and two-class requirement. Nothing to plot.")
            return tbl

        x = np.array([BIN_X[name] for name in ELEMENT_BINS], dtype=float)
        aligned = _align_to_element_bins(tbl, metric_cols)
        source_name = _infer_single_value(df, "source") or ""
        domain_name = domain_used if domain_used not in {"ALL", "MULTI"} else ""
        counts = _agreed_bin_counts(df, source=source_name, domain=domain_name, elements_col=elements_col)

        if show_hist:
            ax_hist = ax.twinx()
            present = counts[counts > 0]
            if not present.empty:
                ax_hist.bar(
                    [BIN_X[name] for name in present.index],
                    present.to_numpy(dtype=float),
                    width=0.6,
                    color="#DDDDDD",
                    alpha=0.4,
                    label="count (n)",
                )
            ax_hist.set_ylabel("Count (n)")
            ymax = float(counts.max()) if len(counts) else 0.0
            ax_hist.set_ylim(0, max(ymax * 1.1, 1.0))

        for met in metric_cols:
            ax.plot(x, aligned[met].to_numpy(dtype=float), marker=metric_marker[met], label=metric_label[met])

        ax.set_xticks(x)
        ax.set_xticklabels(ELEMENT_BINS, rotation=0)
        ax.set_xlabel("Number of unique elements (bin)")
        ax.set_ylabel("ROC-AUC")

        model_raw = _infer_single_value(df, "model")
        model_disp = _display_model(model_raw)

        if title is None:
            plot_title = (
                f"Holdout metrics vs #elements | run_id={rid} | domain={domain_used} | model={model_disp}"
                if model_disp
                else f"Holdout metrics vs #elements | run_id={rid} | domain={domain_used}"
            )
        else:
            plot_title = title
        ax.set_title(plot_title)
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if ax_hist is not None:
                h2, l2 = ax_hist.get_legend_handles_labels()
                handles += h2
                labels += l2
            ax.legend(handles, labels)
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.tight_layout()

        if save:
            name = make_plot_name(
                kind="stats_vs_n_elements",
                run_id=rid,
                mode="final",
                models=model_raw,
                steps=None,
                domains=domain_used if domain_used != "ALL" else None,
            )
            save_figure(fig, project_dir=project_dir, run_id=rid, name=name, close=False)
            save_table(tbl, project_dir=project_dir, run_id=rid, name=name)

        plt.show()
        return tbl

    group_cols = [c for c in ["model", "domain"] if c in df.columns]
    if not group_cols:
        raise ValueError("overlay=True requires 'model' and 'domain' columns in df.")

    if "run_id" in df.columns and df["run_id"].nunique() > 1:
        rid = "+".join(sorted(df["run_id"].dropna().astype(str).unique().tolist()))

    # Infer project name (for possible palette fallbacks) if available
    project_name = None
    if "project" in df.columns and df["project"].nunique() == 1:
        project_name = str(df["project"].iloc[0])

    for key, gg in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(group_cols, key))
        m_raw = str(meta.get("model", "")).strip()
        d_raw = str(meta.get("domain", "")).strip()

        tbl_g = stats_by_n_elements(
            gg,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            elements_col=elements_col,
            threshold=threshold,
            min_n=min_n,
        )
        if tbl_g.empty:
            continue

        x = np.array([BIN_X[name] for name in ELEMENT_BINS], dtype=float)
        aligned = _align_to_element_bins(tbl_g, metric_cols)

        m_key = m_raw.lower()
        m_disp = _display_model(m_raw) or m_raw
        # Line color: RF blue gradient by domain, LGBM red gradient by domain.
        # LGBM is solid and RF is dashed so the models stay distinct in print.
        color = _overlay_line_color(m_key, project_name, d_raw)

        for met in metric_cols:
            lab = f"{m_disp}–{d_raw} ({metric_label[met]})" if len(metric_cols) > 1 else f"{m_disp}–{d_raw}"
            ax.plot(
                x,
                aligned[met].to_numpy(dtype=float),
                marker=metric_marker[met],
                linewidth=2.0,
                linestyle=_overlay_line_style(m_key),
                color=color,
                label=lab,
            )

        out_tbl = tbl_g.copy()
        out_tbl["model"] = m_raw
        out_tbl["domain"] = d_raw
        out_tbl["run_id"] = _infer_single_value(gg, "run_id") or rid
        rows_all.append(out_tbl)

    if not rows_all:
        print("[INFO] No groups produced valid bins. Nothing to plot.")
        plt.close(fig)
        return pd.DataFrame()

    ax.set_xticks([BIN_X[name] for name in ELEMENT_BINS])
    ax.set_xticklabels(ELEMENT_BINS, rotation=0)
    ax.set_xlabel("Number of unique elements (bin)")
    ax.set_ylabel("ROC-AUC")

    # Optional histogram: one bar series per domain, on the shared bin axis.
    hist_ax = None
    if show_hist and "domain" in df.columns:
        hist_ax = ax.twinx()
        # Domain colors for histogram bars: per-project or hist_palette ('mp'|'cod'|'green').
        domain_color = domain_color_palette(project_name, palette=hist_palette)
        domain_color["O"] = _O_BAR_COLOR

        # Group bars around each bin. One shown domain stays on the bin center.
        # COD keeps I left, M center, O right when all three domains are present.
        base_width = 0.18
        shown_domains = [
            dom for dom in ["I", "M", "O"]
            if (
                df["domain"].astype(str).isin(["I", "N"]).any()
                if dom == "I"
                else (df["domain"].astype(str) == dom).any()
            )
        ]
        if len(shown_domains) <= 1:
            offset_map = {dom: 0.0 for dom in shown_domains}
        else:
            mid = (len(shown_domains) - 1) / 2
            offset_map = {
                dom: (index - mid) * base_width
                for index, dom in enumerate(shown_domains)
            }

        raw_domains = set(df["domain"].astype(str))
        if "I" in raw_domains and "N" in raw_domains:
            raise RuntimeError("Domain column contains both I and N. Refusing to combine their element counts.")
        source_name = _infer_single_value(df, "source") or ""
        model_names = (
            sorted(df["model"].dropna().astype(str).unique().tolist())
            if "model" in df.columns
            else None
        )
        bar_heights: list[int] = []
        for dom in ["I", "M", "O"]:
            if dom == "I":
                mask = df["domain"].astype(str).isin(["I", "N"])
                count_domain = "I" if "I" in raw_domains else "N"
            else:
                mask = df["domain"].astype(str) == dom
                count_domain = dom
            if not mask.any():
                continue

            offset = offset_map[dom]
            counts = _agreed_bin_counts(
                df.loc[mask],
                source=source_name,
                domain=count_domain,
                elements_col=elements_col,
                models=model_names,
            )
            present = [(BIN_X[name] + offset, int(counts[name])) for name in ELEMENT_BINS if int(counts[name]) > 0]
            if not present:
                continue
            xs, heights = zip(*present)
            bar_heights.extend(heights)
            bar_alpha = 0.9 if dom == "O" else 0.5
            hist_ax.bar(
                list(xs),
                list(heights),
                width=base_width * 0.9,
                color=domain_color.get(dom, "#cccccc"),
                alpha=bar_alpha,
                label=f"count {dom}",
                zorder=0,
            )

        hist_ax.set_ylabel("Number of examples")
        if bar_heights:
            hist_ax.set_ylim(0, max(bar_heights) * 1.1)

        # Ensure lines are drawn above the bars
        ax.set_zorder(hist_ax.get_zorder() + 1)
        ax.patch.set_visible(False)

    if title is None:
        plot_title = f"Holdout metrics vs #elements | run_id={rid} | domain={domain if domain is not None else 'I/M/O'} | metrics={','.join(metric_cols).upper()}"
    else:
        plot_title = title
    ax.set_title(plot_title)

    # Legend sits below the axes so it is not drawn over the series.
    if show_legend:
        handles, labels_legend = ax.get_legend_handles_labels()
        if hist_ax is not None:
            h2, l2 = hist_ax.get_legend_handles_labels()
            handles += h2
            labels_legend += l2
        _place_legend_below(fig, handles, labels_legend)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if not show_legend:
        fig.tight_layout()

    ds_tag = None
    if "source" in df.columns:
        ds_vals = df["source"].dropna().astype(str).unique().tolist()
        if len(ds_vals) == 1:
            ds_tag = ds_vals[0]

    if save:
        name = make_plot_name(
            kind="stats_vs_n_elements_overlay",
            run_id=rid,
            mode="final",
            models=None,
            steps=None,
            domains=domain if domain is not None else None,
        )
        if ds_tag:
            name = f"{name}__ds={ds_tag}"
        save_figure(fig, project_dir=project_dir, run_id="paper", name=name, close=False)
        save_table(pd.concat(rows_all, ignore_index=True), project_dir=project_dir, run_id="paper", name=name)

    plt.show()
    return pd.concat(rows_all, ignore_index=True)


def plot_stats_by_n_elements(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    models: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    project_dir: str | Path | None = None,
    save: bool = False,
    y_true_col: str = "y",
    overlay: bool = False,
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    metrics: Optional[Iterable[str]] = None,
    datasets=None,
    dataset_col: str = "source",
    title: Optional[str] = None,
    show_hist: bool = False,
    show_legend: bool = True,
    hist_palette: str | None = None,
) -> pd.DataFrame:
    """Wrapper: plot stats vs #elements. If overlay=True, plot all groups on one figure.
    hist_palette: optional 'mp'|'cod'|'green' for histogram bar colors."""
    models_l = as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    domains_l = as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []

    if dataset_col in df.columns:
        if datasets is None:
            datasets_l = sorted(df[dataset_col].dropna().astype(str).unique().tolist())
        else:
            datasets_l = [str(x) for x in as_list(datasets)] if as_list(datasets) is not None else [str(datasets)]
    else:
        datasets_l = None

    metric_cols = ["auc", "bacc", "mcc"]
    ylim = infer_ylim_stats_by_n_elements(
        df,
        metric_cols=metric_cols,
        y_true_col=y_true_col,
        y_prob_col=y_prob_col,
        elements_col=elements_col,
        threshold=threshold,
        min_n=min_n,
        dataset_col=dataset_col,
        datasets=datasets_l,
    )
    if ylim is not None:
        lo, hi = ylim
        pad = 0.05 * (hi - lo)
        ylim = (lo - pad, hi + pad)

    run_map = resolve_run_map(df, runs=runs, models=models_l)

    if overlay:
        out = []
        forced_domain = domains_l[0] if len(domains_l) == 1 else None
        dataset_iter = datasets_l if datasets_l is not None else [None]

        for ds in dataset_iter:
            sub = df.copy()
            if ds is not None:
                sub = sub[sub[dataset_col].astype(str) == ds]

            if "model" in sub.columns and models_l:
                sub = sub[sub["model"].isin(models_l)]
            if "domain" in sub.columns and domains_l:
                sub = sub[sub["domain"].isin(domains_l)]
            if "run_id" in sub.columns:
                allowed_runs = set(run_map.values())
                sub = sub[sub["run_id"].isin(allowed_runs)]

            if title is not None:
                title_ds = title
            else:
                title_ds = str(ds) if ds is not None else None

            tbl = _plot_stats_by_n_elements_single(
                sub,
                y_true_col=y_true_col,
                y_prob_col=y_prob_col,
                elements_col=elements_col,
                threshold=threshold,
                min_n=min_n,
                metrics=metrics,
                title=title_ds,
                save=save,
                project_dir=project_dir,
                run_id=None,
                domain=forced_domain,
                overlay=True,
                ylim=ylim,
                show_hist=show_hist,
                show_legend=show_legend,
                hist_palette=hist_palette,
            )
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                if ds is not None:
                    tbl = tbl.copy()
                    tbl[dataset_col] = ds
                out.append(tbl)

        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    out = []
    dataset_iter = datasets_l if datasets_l is not None else [None]

    for ds in dataset_iter:
        for m in models_l:
            run_id = run_map[m]
            for dom in domains_l:
                sub = df
                if ds is not None:
                    sub = sub[sub[dataset_col].astype(str) == ds]
                if "model" in sub.columns:
                    sub = sub[sub["model"] == m]
                if "domain" in sub.columns:
                    sub = sub[sub["domain"] == dom]
                if "run_id" in sub.columns:
                    sub = sub[sub["run_id"] == run_id]

                if title is not None:
                    title_ds = title
                else:
                    title_ds = str(ds) if ds is not None else None

                tbl = _plot_stats_by_n_elements_single(
                    sub,
                    y_true_col=y_true_col,
                    y_prob_col=y_prob_col,
                    elements_col=elements_col,
                    threshold=threshold,
                    min_n=min_n,
                    title=title_ds,
                    save=save,
                    project_dir=project_dir,
                    run_id=run_id,
                    domain=dom,
                    metrics=metrics,
                    show_hist=show_hist,
                    show_legend=show_legend,
                    hist_palette=hist_palette,
                )
                if not tbl.empty:
                    tbl = tbl.copy()
                    tbl["run_id"] = run_id
                    tbl["model"] = m
                    tbl["domain"] = dom
                    if ds is not None:
                        tbl[dataset_col] = ds
                    out.append(tbl)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def plot_stats_by_n_elements_legend(
    project_name: str | None = None,
    *,
    project_dir: str | Path | None = None,
    run_id: str = "paper",
    name: str = "stats_by_n_elements_legend",
    save: bool = False,
    hist_palette: str | None = None,
):
    """Create a standalone figure containing only the legend for stats_by_n_elements overlay.

    Entries: RF–I, RF–M, RF–O, LGBM–I, LGBM–M, LGBM–O, count I, count M, count O.
    hist_palette: optional 'mp'|'cod'|'green' for Domain bar colors. In notebook: display(fig).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    dom_colors = domain_color_palette(project_name, palette=hist_palette)
    handles = []
    labels = []

    for model_key, m_disp in [("lgbm", MODEL_DISPLAY.get("lgbm", "LGBM")), ("rf", MODEL_DISPLAY.get("rf", "RF"))]:
        for d in ["I", "M", "O"]:
            color = _overlay_line_color(model_key, project_name, d)
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    linewidth=2,
                    linestyle=_overlay_line_style(model_key),
                    marker="o",
                    markersize=6,
                )
            )
            labels.append(f"{m_disp}–{d}")

    dom_colors = dict(dom_colors)
    dom_colors["O"] = _O_BAR_COLOR
    for d in ["I", "M", "O"]:
        handles.append(
            Patch(
                facecolor=dom_colors.get(d, "#cccccc"),
                alpha=0.9 if d == "O" else 0.5,
                edgecolor="none",
            )
        )
        labels.append(f"count {d}")

    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.axis("off")
    ax.legend(handles, labels, loc="center", ncol=3, frameon=False)
    fig.tight_layout()

    if save and project_dir is not None:
        save_figure(fig, project_dir=project_dir, run_id=run_id, name=name, close=False)
    return fig
