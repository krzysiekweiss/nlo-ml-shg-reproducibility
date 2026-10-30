"""Calculate per-element metrics for selected model, domain and run combinations."""

import pandas as pd
from pathlib import Path
from typing import Iterable, Optional, Union

from .io import save_table, as_list, resolve_run_map
from .metric_vs_Z import per_element_metric_table_Z

def _per_element_metric_table_Z_by_run_and_domain_single(
    df: pd.DataFrame,
    *,
    run_col: str = "run_id",
    domain_col: str = "domain",
    runs: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[str]] = None,
    metric: str,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
    save: bool = False,
    project_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Return one tidy table with columns:
      run_id, domain, element, Z, n, <metric>
    """
    if df.empty:
        return pd.DataFrame(columns=["run_id", "domain", "element", "Z", "n", metric])

    for col in (run_col, domain_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in df.")

    if runs is None:
        run_list = sorted(df[run_col].dropna().unique())
    else:
        run_list = list(runs)

    if domains is None:
        domain_list = sorted(df[domain_col].dropna().unique())
    else:
        domain_list = list(domains)

    frames = []

    for run in run_list:
        df_run = df[df[run_col] == run]
        if df_run.empty:
            continue

        for dom in domain_list:
            sub = df_run[df_run[domain_col] == dom]
            if sub.empty:
                continue

            tbl = per_element_metric_table_Z(
                sub,
                metric=metric,
                y_true_col=y_true_col,
                y_prob_col=y_prob_col,
                elements_col=elements_col,
                threshold=threshold,
                min_n=min_n,
            )
            if tbl.empty:
                continue

            tbl = tbl.copy()
            tbl["run_id"] = run
            tbl["domain"] = dom
            frames.append(tbl)

    if not frames:
        return pd.DataFrame(columns=["run_id", "domain", "element", "Z", "n", metric])

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["run_id", "domain", "Z"])
        .reset_index(drop=True)
    )


def per_element_metric_table_Z_by_run_and_domain(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    models: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
    project_dir: str | Path | None = None,
    save: bool = False,
    # Metric settings.
    metric: str,
    y_true_col: str = "y",
    y_prob_col: str = "y_prob",
    elements_col: str = "elements_cif",
    threshold: float = 0.5,
    min_n: int = 30,
) -> pd.DataFrame:
    """Compute per-element metric table for each (model, domain); save CSV per (model, domain)."""
    models_l = as_list(models)
    if models_l is None:
        models_l = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
    domains_l = as_list(domains)
    if domains_l is None:
        domains_l = sorted(df["domain"].dropna().unique().tolist()) if "domain" in df.columns else []

    run_map = resolve_run_map(df, runs=runs, models=models_l)

    # Combine metrics for all requested runs and domains.
    run_list = [run_map[m] for m in models_l]
    out_df = _per_element_metric_table_Z_by_run_and_domain_single(
        df,
        runs=run_list,
        domains=domains_l,
        metric=metric,
        y_true_col=y_true_col,
        y_prob_col=y_prob_col,
        elements_col=elements_col,
        threshold=threshold,
        min_n=min_n,
        save=False,
        project_dir=project_dir,
    )

    if out_df.empty:
        return out_df

    if save:
        # Save one CSV per (model, domain) using the same base name as plot_metric_vs_Z
        for m in models_l:
            run_id = run_map[m]
            for dom in domains_l:
                sub = out_df[(out_df["run_id"] == run_id) & (out_df["domain"] == dom)]
                if sub.empty:
                    continue
                name = "__".join([f"metric_{metric}_vs_Z", f"model_{m}", f"domain_{dom}"])
                save_table(
                    sub,
                    project_dir=project_dir,
                    run_id=run_id,
                    name=name,
                )

    return out_df
