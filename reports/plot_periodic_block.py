"""Summarize per-element predictive performance by periodic-table block."""

from pathlib import Path
from typing import Iterable, Optional, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .chemistry import periodic_block_from_Z
from . import plots_common as pls
from .io import save_table


# Internal helpers


MODEL_DISPLAY = {"rf": "rf", "lgbm": "lgbm"}


def _iqr(x: pd.Series) -> float:
    """Return interquartile range (Q3 - Q1) as float."""
    return float(x.quantile(0.75) - x.quantile(0.25))


# AUC by periodic block: single (model, run, domain)


def _plot_auc_block_single(
    df: pd.DataFrame,
    *,
    run_id: str,
    domain: str,
    project_dir: str | Path,
    model: Optional[str] = None,
    min_n: int = 20,
    title: Optional[str] = None,
    save: bool = True,
    verbose: bool = False,
    box: bool = False,
) -> pd.DataFrame:
    """
    Plot AUC vs periodic block (s/p/d/f) for a single run_id and domain.

    Modes:
      - box=True  → boxplot per block (distribution)
      - box=False → ROC-style line plot (median AUC per block)

    Visual encoding (line mode):
      - color: model (RF vs LGBM)
      - linestyle: domain (I solid, M dashed, O dotted)
    """
    domain_ls = {"I": "-", "M": "--", "O": ":"}
    model_display = MODEL_DISPLAY

    g = df.copy()

    if "domain" in g.columns:
        g = g[g["domain"] == domain]

    if "run_id" in g.columns:
        g = g[g["run_id"] == run_id]

    if model is not None and "model" in g.columns:
        g = g[g["model"] == model]

    if g.empty:
        print("[INFO] Empty selection. Nothing to plot.")
        return pd.DataFrame()

    g = g[g["n"] >= min_n].copy()
    if g.empty:
        print(f"[INFO] No elements with n >= {min_n}.")
        return pd.DataFrame()

    g["block"] = g["Z"].astype(int).apply(periodic_block_from_Z)

    summary = (
        g.groupby("block")
        .agg(
            n_elements=("Z", "count"),
            n_total=("n", "sum"),
            auc_median=("auc", "median"),
            auc_iqr=("auc", lambda x: float(x.quantile(0.75) - x.quantile(0.25))),
        )
        .reset_index()
        .sort_values("block")
    )

    if verbose:
        print("=" * 80)
        print("[AUC vs periodic block]")
        print(f"run_id = {run_id}")
        print(f"domain = {domain}")
        if model:
            print(f"model  = {model}")
        print(f"min_n  = {min_n}")
        print("-" * 80)
        print(summary.to_string(index=False))
        print("=" * 80)

    fig, ax = plt.subplots()
    order = ["s", "p", "d", "f"]
    blocks = [b for b in order if b in summary["block"].values]
    model_key = str(model).strip().lower() if model is not None else ""
    color = pls.model_color.get(model_key, None)


    # MODE 1: BOX PLOT

    if box:
        data = [g[g["block"] == b]["auc"].values for b in blocks]

        bp = ax.boxplot(
            data,
            labels=blocks,
            showfliers=True,
            patch_artist=True,
        )

        face = color

        # Domain style (hatch pattern)
        domain_hatch = {"I": "", "M": "///", "O": ".."}
        hatch = domain_hatch.get(str(domain), "")

        # Color for median and whiskers
        line_color = "#333333"

        # Box plot settings
        for patch in bp["boxes"]:
            patch.set_facecolor(face)
            patch.set_alpha(0.65)
            patch.set_hatch(hatch)

        for element in ("medians", "whiskers", "caps"):
            for ln in bp[element]:
                ln.set_color(line_color)
                ln.set_linewidth(1.0)


    # MODE 2: LINE PLOT (ROC-style)

    else:
        aucs = [summary[summary["block"] == b]["auc_median"].values[0] for b in blocks]

        linestyle = domain_ls.get(str(domain), "-")
        label = f"{model_display.get(model_key, model)} | {domain}"

        ax.plot(
            blocks,
            aucs,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
        )
        ax.legend()

    ax.set_xlabel("Periodic block")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.02)  # Fixed scale for MP/COD; small top margin above 1
    if title is None:
        ax.set_title(f"AUC by periodic block | run_id={run_id} | domain={domain}")
    else:
        ax.set_title(title)
    fig.tight_layout()

    if save:
        project_dir = Path(project_dir)
        name = f"auc_by_periodic_block__domain_{domain}"

        fig_dir = project_dir / "results" / run_id / "figures"
        tab_dir = project_dir / "results" / run_id / "tables"
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(fig_dir / f"{name}.png", dpi=300)
        summary.to_csv(tab_dir / f"{name}.csv", index=False)

    plt.show()
    return summary


# AUC by periodic block: overlay for many (model, run, domain)


def _plot_auc_block_overlay(
    df: pd.DataFrame,
    *,
    groups: list[tuple[str, str, str]],      # list of (model, run_id, domain)
    project_dir: str | Path,
    min_n: int = 20,
    title: Optional[str] = None,
    save: bool = True,
    name: Optional[str] = None,
    verbose: bool = False,
    box: bool = False,
) -> pd.DataFrame:
    """
    Overlay boxplots: AUC grouped by periodic block for many (model, run_id, domain).

    Parameters
    ----------
    df:
        AUC-vs-Z table(s) concatenated together. Must contain at least: Z, auc, n.
        Context columns model, run_id, domain are strongly recommended.
    groups:
        Explicit list of (model, run_id, domain) triples to include.

    Returns
    -------
    Tidy summary table with per-(model, run_id, domain, block) statistics.
    """

    def _display_model(m: str) -> str:
        key = str(m).strip().lower()
        return MODEL_DISPLAY.get(key, str(m)) if "MODEL_DISPLAY" in globals() else str(m)

    # Domain is encoded by fill color: I dark, O medium, M light
    domain_color = {
        "I": "#3182bd",
        "M": "#deebf7",
        "O": "#9ecae1",
        "N": "#3182bd",
    }

    if df is None or df.empty:
        print("[INFO] Empty selection. Nothing to plot.")
        return pd.DataFrame()

    g_all = df.copy()
    for c in ("Z", "auc", "n"):
        if c not in g_all.columns:
            raise KeyError(f"Missing required column '{c}'.")

    if "model" not in g_all.columns:
        g_all["model"] = None
    if "run_id" not in g_all.columns:
        g_all["run_id"] = None
    if "domain" not in g_all.columns:
        g_all["domain"] = None

    g_all = g_all[g_all["n"].astype(float) >= float(min_n)].copy()
    if g_all.empty:
        print(f"[INFO] No elements with n >= {min_n} across requested groups.")
        return pd.DataFrame()

    g_all["block"] = g_all["Z"].astype(int).apply(periodic_block_from_Z)

    # Keep only requested triples (and preserve their order)
    grp_df = pd.DataFrame(groups, columns=["model", "run_id", "domain"]).astype(str)
    g_all["model"] = g_all["model"].astype(str)
    g_all["run_id"] = g_all["run_id"].astype(str)
    g_all["domain"] = g_all["domain"].astype(str)
    g_all = g_all.merge(
        grp_df.assign(_keep=1),
        on=["model", "run_id", "domain"],
        how="inner",
    )
    g_all = g_all.drop(columns=["_keep"], errors="ignore")
    if g_all.empty:
        print("[INFO] No data left after filtering to requested (model, run_id, domain) groups.")
        return pd.DataFrame()

    summary = (
        g_all.groupby(["model", "run_id", "domain", "block"], dropna=False)
        .agg(
            n_elements=("Z", "count"),
            n_total=("n", "sum"),
            auc_median=("auc", "median"),
            auc_iqr=("auc", _iqr),
        )
        .reset_index()
        .sort_values(["model", "domain", "block"])
        .reset_index(drop=True)
    )

    if verbose:
        print("=" * 80)
        print("[AUC vs periodic block | OVERLAY]")
        print(f"groups = {len(groups)}")
        print(f"min_n  = {min_n}")
        print("-" * 80)
        print(summary.to_string(index=False))
        print("=" * 80)

    order = ["s", "p", "d", "f"]
    blocks_present = [b for b in order if b in set(g_all["block"].tolist())]
    if not blocks_present:
        print("[INFO] No periodic blocks present after filtering.")
        return pd.DataFrame()

    fig, ax = plt.subplots()

    n_groups = len(groups)
    if n_groups == 0:
        print("[INFO] No groups requested.")
        return pd.DataFrame()

    base_x = np.arange(len(blocks_present), dtype=float)
    total_w = 0.80
    step = total_w / max(n_groups, 1)
    box_w = min(step * 0.85, 0.25)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * step

    legend_handles = []
    legend_labels = []

    for gi, (m_raw, run_id, dom) in enumerate(groups):
        sub = g_all[
            (g_all["model"] == str(m_raw))
            & (g_all["run_id"] == str(run_id))
            & (g_all["domain"] == str(dom))
        ]
        if sub.empty:
            continue

        data = [sub[sub["block"] == b]["auc"].to_numpy() for b in blocks_present]
        pos = base_x + offsets[gi]

        bp = ax.boxplot(
            data,
            positions=pos,
            widths=box_w,
            patch_artist=True,
            showfliers=True,
            manage_ticks=False,
        )

        m_disp = _display_model(str(m_raw))
        face = domain_color.get(str(dom), "#cccccc")

        for patch in bp["boxes"]:
            patch.set_facecolor(face)
            patch.set_alpha(0.9)
            patch.set_hatch("")

        for element in ("medians", "whiskers", "caps"):
            for ln in bp[element]:
                ln.set_color("#333333")
                ln.set_linewidth(1.0)

        from matplotlib.patches import Patch

        lab = f"{m_disp}–{dom}"
        legend_handles.append(
            Patch(facecolor=face, edgecolor="#333333", alpha=0.9)
        )
        legend_labels.append(lab)

    ax.set_xticks(base_x)
    ax.set_xticklabels(blocks_present)
    ax.set_xlabel("Periodic block")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.02)  # Fixed scale for MP/COD; small top margin above 1

    if title is None:
        plot_title = "AUC by periodic block | overlay"
    else:
        plot_title = title
    ax.set_title(plot_title)
    ax.legend(legend_handles, legend_labels, frameon=False)
    fig.tight_layout()

    if save:
        project_dir = Path(project_dir)
        out_name = name or "auc_by_periodic_block__overlay"
        pls.save_figure(fig, project_dir=project_dir, run_id="paper", name=out_name, close=False)
        save_table(summary, project_dir=project_dir, run_id="paper", name=out_name)

    plt.show()
    return summary


# Loading AUC-vs-Z tables


def load_metric_auc_vs_Z_from_project(
    *,
    project_dir: str | Path,
    run_id: str,
    model: str,
    domain: str,
    filename: str | None = None,
) -> pd.DataFrame:
    """
    Load AUC-vs-Z table from:
      <project_dir>/results/<run_id>/tables/metric_auc_vs_Z__model_<model>__domain_<domain>.csv

    The filename argument overrides the default name relative to the tables directory.
    """
    project_dir = Path(project_dir)
    tables_dir = project_dir / "results" / run_id / "tables"

    if filename is None:
        filename = f"metric_auc_vs_Z__model_{model}__domain_{domain}.csv"

    path = tables_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing AUC-vs-Z file: {path}")

    df = pd.read_csv(path)

    if "run_id" not in df.columns:
        df["run_id"] = run_id
    if "model" not in df.columns:
        df["model"] = model
    if "domain" not in df.columns:
        df["domain"] = domain

    return df


# Core implementation for runs × domains


def _plot_auc_block_for_runs(
    *,
    cfg,
    runs: Mapping[str, str],          # e.g. {"rf": "rf005_ht", "lgbm": "lgbm004_ht"}
    domains: Iterable[str],           # e.g. ["I", "M", "O"]
    min_n: int = 20,
    overlay: bool = False,
    box: bool = True,
    save: bool = True,
    title: Optional[str] = None,
    verbose: bool = False,
    project_dir: str | Path | None = None,
    filename_template: str | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Core implementation for AUC-by-periodic-block plotting across runs and domains.

    Reads per-run tables:
      <project_dir>/results/<run_id>/tables/<filename>

    Default filename pattern:
      metric_auc_vs_Z__model_<model>__domain_<domain>.csv

    Overlay mode:
      - loads all requested (model, domain) tables
      - produces one combined figure for the given project
      - returns {("OVERLAY", "OVERLAY"): summary_df}

    Non-overlay mode:
      - produces one figure per (model, domain)
      - returns {(model, domain): summary_df}
    """
    proj_dir = Path(project_dir or cfg.PROJECT_DIR)

    if filename_template is None:
        filename_template = "metric_auc_vs_Z__model_{model}__domain_{domain}.csv"

    out: dict[tuple[str, str], pd.DataFrame] = {}

    if overlay:
        frames: list[pd.DataFrame] = []
        groups: list[tuple[str, str, str]] = []  # (model, run_id, domain)

        for model, run_id in runs.items():
            for domain in domains:
                fname = filename_template.format(model=model, domain=domain)
                try:
                    df_auc = load_metric_auc_vs_Z_from_project(
                        project_dir=proj_dir,
                        run_id=run_id,
                        model=model,
                        domain=domain,
                        filename=fname,
                    )
                except FileNotFoundError as e:
                    print(f"[WARN] {e}")
                    continue

                frames.append(df_auc)
                groups.append((str(model), str(run_id), str(domain)))

        if not frames:
            return {}

        df_all = pd.concat(frames, ignore_index=True)

        def _slug(s: str) -> str:
            return "".join(ch if ch.isalnum() else "_" for ch in str(s)).strip("_")

        name = "auc_by_periodic_block__overlay"
        if title:
            name = f"{name}__{_slug(title)}"

        summary = _plot_auc_block_overlay(
            df_all,
            groups=groups,
            project_dir=proj_dir,
            min_n=min_n,
            title=title,
            box=box,
            save=save,
            name=name,
            verbose=verbose,
        )
        out[("OVERLAY", "OVERLAY")] = summary
        return out

    # Non-overlay: one plot per (model, domain)
    for model, run_id in runs.items():
        for domain in domains:
            fname = filename_template.format(model=model, domain=domain)
            try:
                df_auc = load_metric_auc_vs_Z_from_project(
                    project_dir=proj_dir,
                    run_id=run_id,
                    model=model,
                    domain=domain,
                    filename=fname,
                )
            except FileNotFoundError as e:
                print(f"[WARN] {e}")
                continue

            summary = _plot_auc_block_single(
                df_auc,
                run_id=run_id,
                domain=domain,
                project_dir=proj_dir,
                model=model,
                min_n=min_n,
                title=title,
                box=box,
                save=save,
                verbose=verbose,
            )
            out[(model, domain)] = summary

    return out


# Public API


def plot_auc_block_for_runs(
    *,
    cfg,
    runs: Mapping[str, str],          # e.g. {"rf": "rf005_ht", "lgbm": "lgbm004_ht"}
    domains: Iterable[str],           # e.g. ["N", "O", "M"]
    min_n: int = 20,
    overlay: bool = False,
    box: bool = True,
    save: bool = True,
    title: Optional[str] = None,
    verbose: bool = False,
    project_dir: str | Path | None = None,
    filename_template: str | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Public wrapper for period-block AUC plots across runs and domains.

    Delegates to `_plot_auc_block_for_runs`, which implements both overlay
    and non-overlay logic to keep behavior centralized and consistent.
    """
    return _plot_auc_block_for_runs(
        cfg=cfg,
        runs=runs,
        domains=domains,
        min_n=min_n,
        overlay=overlay,
        box=box,
        save=save,
        title=title,
        verbose=verbose,
        project_dir=project_dir,
        filename_template=filename_template,
    )


# Internal helpers
