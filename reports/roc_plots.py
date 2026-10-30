"""Generate ROC panels for selected datasets, models and chemical domains."""

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from . import plots_common as pls
from . import io as iopl


# Internal helpers


MODEL_DISPLAY = {"rf": "rf", "lgbm": "lgbm"}


def _plot_roc_panel_for_dataset(
    df: pd.DataFrame,
    *,
    run_map: dict[str, str],                 # model -> run_id
    dataset: str,                            # e.g. "MP" or "COD"
    dataset_col: str = "source",
    domains: Optional[Union[str, Iterable[str]]] = None,
    project: Optional[str] = None,
    mode: str = "final",
    steps: Optional[Union[str, Iterable[str]]] = None,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
    name: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Plot a ROC panel for a single dataset (e.g. MP or COD).

    Visual encoding:
      - color: model (RF vs LGBM)
      - linestyle: domain (I solid, M dashed, O dotted)

    Saves PNG + CSV if save=True.
    Returns a summary table with AUC per (dataset, model, domain).
    """
    assert df is not None, "df is None at entry"

    # Display labels for model identifiers.
    model_display = MODEL_DISPLAY

    # Domain → linestyle mapping
    domain_ls = {"I": "-", "M": "--", "O": ":"}

    # Resolve domain list
    domains_l = iopl.as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().astype(str).unique().tolist()) if "domain" in df.columns else []

    # Filter dataset once
    g0 = df.copy()
    if dataset_col in g0.columns:
        g0 = g0[g0[dataset_col].astype(str) == str(dataset)]
    if g0.empty:
        if verbose:
            print(f"[INFO] Empty dataset selection: dataset={dataset}")
        return pd.DataFrame()

    rows = []
    fig, ax = plt.subplots()

    for model, run_id in run_map.items():
        for dom in domains_l:
            g = iopl.filter_df(
                g0,
                run_id=run_id,
                project=project,
                mode=mode,
                models=model,
                steps=steps,
                domains=dom,
            )
            if g.empty:
                if verbose:
                    print(f"[INFO] empty: dataset={dataset} domain={dom} model={model} run_id={run_id}")
                continue

            y_true = g[y_true_col].astype(int).to_numpy()
            y_prob = g[y_prob_col].astype(float).to_numpy()

            # Skip if only one class present
            if len(np.unique(y_true)) < 2:
                if verbose:
                    print(f"[WARN] one-class: dataset={dataset} domain={dom} model={model} n={len(g)}")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = float(roc_auc_score(y_true, y_prob))

            model_key = str(model).lower()
            model_name = model_display.get(model_key, str(model))
            ls = domain_ls.get(str(dom), "-")

            label = f"{model_name} | {dom} (AUC={auc:.3f})"

            ax.plot(
                fpr,
                tpr,
                label=label,
                linestyle=ls,
                linewidth=2.0,
                color=pls.model_color.get(model_key, None),
            )

            rows.append({
                "dataset": str(dataset),
                "model": str(model),
                "run_id": str(run_id),
                "domain": str(dom),
                "n": int(len(g)),
                "auc": auc,
            })

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="#BBBBBB", zorder=0)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title is None:
        ax.set_title(f"{dataset}")
    elif title:
        ax.set_title(title)
    else:
        ax.set_title("")

    if rows:
        ax.legend()

    fig.tight_layout()
    summary = pd.DataFrame(rows)

    if save:
        pls.save_figure(fig, project_dir=project_dir, run_id="paper", name=name, close=False)
        iopl.save_table(summary, project_dir=project_dir, run_id="paper", name=name)

    plt.show()
    return summary


# Public wrapper: panels by dataset


def plot_roc_panels_by_dataset(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    project: Optional[str] = None,
    mode: str = "final",
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    datasets: Optional[Iterable[str]] = None,
    dataset_col: str = "source",
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Wrapper for paper-style ROC panels:
      - one panel per dataset (MP, COD)
      - color encodes model, linestyle encodes domain

    Saves one PNG + one CSV per dataset panel.
    Returns concatenated summary rows.
    """
    models_l = iopl.as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []

    run_map = iopl.resolve_run_map(df, runs=runs, models=models_l)

    # Infer datasets if not provided
    if datasets is None:
        if dataset_col in df.columns:
            datasets_l = sorted(df[dataset_col].dropna().astype(str).unique().tolist())
        else:
            datasets_l = ["MP", "COD"]
    else:
        datasets_l = list(datasets)

    out = []
    for ds in datasets_l:
        tbl = _plot_roc_panel_for_dataset(
            df,
            run_map=run_map,
            dataset=ds,
            dataset_col=dataset_col,
            domains=domains,
            project=project,
            mode=mode,
            steps=steps,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            title=title,
            save=save,
            project_dir=project_dir,
            name=f"paper_roc_panel__dataset_{ds}",
            verbose=verbose,
        )
        if not tbl.empty:
            out.append(tbl)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# Internal: single ROC selection


def _plot_roc_single_selection(
    df: pd.DataFrame,
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    mode: str = "final",
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Plot ROC curve(s) for a single filtered selection of holdout predictions.
    Returns a summary table with AUC per group.
    """
    g = iopl.filter_df(
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

    def _infer_run_id() -> str:
        if run_id:
            return str(run_id)
        if "run_id" in g.columns:
            u = [x for x in g["run_id"].dropna().unique().tolist() if str(x).strip()]
            if len(u) == 1:
                return str(u[0])
        return "unknown_run"

    def _infer_domain_text() -> str:
        if isinstance(domains, str):
            return domains
        if domains is not None:
            dd = list(domains)
            return dd[0] if len(dd) == 1 else "multiple"
        if "domain" in g.columns:
            u = [x for x in g["domain"].dropna().unique().tolist() if str(x).strip()]
            if len(u) == 1:
                return str(u[0])
            if len(u) > 1:
                return "multiple"
        return "ALL"

    # Determine grouping columns
    group_cols = [c for c in ["model", "step", "domain"] if c in g.columns]
    if models is not None:
        group_cols = [c for c in group_cols if c != "model"]
    if steps is not None:
        group_cols = [c for c in group_cols if c != "step"]
    if domains is not None:
        group_cols = [c for c in group_cols if c != "domain"]

    rows = []
    fig, ax = plt.subplots()

    # Title
    plot_title = title or f"Holdout ROC | run_id={_infer_run_id()} | domain={_infer_domain_text()}"

    # Group and plot
    grouped = g.groupby(group_cols, dropna=False) if group_cols else [(("ALL",), g)]
    for key, gg in grouped:
        y_true = gg[y_true_col].astype(int).values
        y_prob = gg[y_prob_col].astype(float).values

        if len(np.unique(y_true)) < 2:
            print("[WARN] Skipping group: only one class present.")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = float(roc_auc_score(y_true, y_prob))

        label = pls.build_legend_label(gg, auc=auc, show_domain=False)
        ax.plot(fpr, tpr, label=label)

        rows.append({"auc": auc})

    # Diagonal reference
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(plot_title)
    ax.legend()
    fig.tight_layout()

    if save:
        name = pls.make_plot_name(
            kind="holdout_roc",
            run_id=run_id,
            mode=mode,
            models=models,
            steps=steps,
            domains=domains,
        )
        pls.save_figure(fig, project_dir=project_dir, run_id=run_id, name=name, close=False)
        iopl.save_table(pd.DataFrame(rows), project_dir=project_dir, run_id=run_id, name=name)

    plt.show()
    return pd.DataFrame(rows)


# Public wrapper: ROC by model × domain


def plot_roc_by_model_and_domain(
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
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Wrapper: generate ROC plots for each (model, domain) pair.
    Saves PNG + CSV per plot if save=True.
    """
    models_l = iopl.as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []

    domains_l = iopl.as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []

    run_map = iopl.resolve_run_map(df, runs=runs, models=models_l)

    out = []
    for m in models_l:
        run_id = run_map[m]
        for dom in domains_l:
            tbl = _plot_roc_single_selection(
                df,
                run_id=run_id,
                project=project,
                mode=mode,
                models=m,
                steps=steps,
                domains=dom,
                y_true_col=y_true_col,
                y_prob_col=y_prob_col,
                title=title,
                save=save,
                project_dir=project_dir,
            )
            if not tbl.empty:
                tbl = tbl.copy()
                tbl["run_id"] = run_id
                tbl["model"] = m
                tbl["domain"] = dom
                out.append(tbl)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# Internal: overlay ROC for a domain


def _plot_roc_overlay_for_domain(
    df: pd.DataFrame,
    *,
    run_map: dict[str, str],
    domain: str,
    datasets: Iterable[str],
    dataset_col: str = "source",
    project: Optional[str] = None,
    mode: str = "final",
    steps: Optional[Union[str, Iterable[str]]] = None,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
    name: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Plot ROC overlay for a single domain:
      - color encodes dataset (MP/COD)
      - linestyle encodes model (RF solid, LGBM dashed)

    Saves PNG + CSV if save=True.
    Returns a summary table with AUC per (dataset, model).
    """
    model_ls = {"rf": "-", "lgbm": "--"}

    rows = []
    fig, ax = plt.subplots()

    for ds in list(datasets):
        for model, run_id in run_map.items():
            g = iopl.filter_df(
                df,
                run_id=run_id,
                project=project,
                mode=mode,
                models=model,
                steps=steps,
                domains=domain,
            )
            if dataset_col in g.columns:
                g = g[g[dataset_col].astype(str) == str(ds)]

            if g.empty:
                if verbose:
                    print(f"[INFO] empty: domain={domain} ds={ds} model={model} run_id={run_id}")
                continue

            y_true = g[y_true_col].astype(int).to_numpy()
            y_prob = g[y_prob_col].astype(float).to_numpy()

            if len(np.unique(y_true)) < 2:
                if verbose:
                    print(f"[WARN] one-class: domain={domain} ds={ds} model={model} n={len(g)}")
                continue

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = float(roc_auc_score(y_true, y_prob))

            model_name = MODEL_DISPLAY.get(str(model).lower(), str(model)) if "MODEL_DISPLAY" in globals() else str(model)
            label = f"{ds} — {model_name} (AUC={auc:.3f})"

            ax.plot(
                fpr,
                tpr,
                label=label,
                linestyle=model_ls.get(str(model).lower(), "-"),
                linewidth=2.0,
            )

            rows.append({
                "dataset": str(ds),
                "model": str(model),
                "run_id": str(run_id),
                "domain": str(domain),
                "n": int(len(g)),
                "auc": auc,
            })

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title is None:
        ax.set_title(f"ROC overlay | domain={domain}")
    elif title:
        ax.set_title(title)
    else:
        ax.set_title("")
    ax.legend()
    fig.tight_layout()

    summary = pd.DataFrame(rows)

    if save:
        base = name or f"roc_overlay__domain_{domain}"
        pls.save_figure(fig, project_dir=project_dir, run_id="paper", name=base, close=False)
        iopl.save_table(summary, project_dir=project_dir, run_id="paper", name=base)

    plt.show()
    return summary


# Public wrapper: overlay by domain


def plot_roc_overlay_by_domain(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    project: Optional[str] = None,
    mode: str = "final",
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    datasets: Optional[Iterable[str]] = None,
    dataset_col: str = "source",
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    title: Optional[str] = None,
    save: bool = False,
    project_dir: str | Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Wrapper for paper-style ROC overlays:
      - for each domain, plot overlay ROC with dataset colors and model linestyles
      - one PNG + one CSV per domain (via save_figure/save_table)

    Returns concatenated summary rows across all domains.
    """
    # Resolve models to plot
    models_l = iopl.as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []

    # Resolve domains to plot
    domains_l = iopl.as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []

    # Map model -> run_id
    run_map = iopl.resolve_run_map(df, runs=runs, models=models_l)

    # Infer datasets if not provided
    if datasets is None:
        if dataset_col in df.columns:
            datasets_l = sorted(df[dataset_col].dropna().astype(str).unique().tolist())
        else:
            # Fallback: assume MP and COD if the column is missing
            datasets_l = ["MP", "COD"]
    else:
        datasets_l = list(datasets)

    out = []
    for dom in domains_l:
        # Domain-specific dataset selection:
        # - I: MP + COD
        # - M/O: COD only (if present)
        ds_for_dom = datasets_l
        if dom in {"M", "O"} and "COD" in datasets_l:
            ds_for_dom = ["COD"]

        tbl = _plot_roc_overlay_for_domain(
            df,
            run_map=run_map,
            domain=dom,
            datasets=ds_for_dom,
            dataset_col=dataset_col,
            project=project,
            mode=mode,
            steps=steps,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            title=title,
            save=save,
            project_dir=project_dir,
            name=f"paper_roc_overlay__domain_{dom}",
            verbose=verbose,
        )
        if not tbl.empty:
            out.append(tbl)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# Internal helpers
