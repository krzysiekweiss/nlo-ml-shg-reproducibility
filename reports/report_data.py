"""Assemble labelled prediction tables for within-domain reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from .io import load_all_holdouts_legacy


def _stack_sources(
    df_mp: Optional[pd.DataFrame],
    df_cod: Optional[pd.DataFrame],
    *,
    source_col: str = "source",
) -> pd.DataFrame:
    """Stack MP and COD holdout tables, tagging rows with a source column."""
    parts = []
    if df_mp is not None and not df_mp.empty:
        a = df_mp.copy()
        a[source_col] = "MP"
        parts.append(a)
    if df_cod is not None and not df_cod.empty:
        b = df_cod.copy()
        b[source_col] = "COD"
        parts.append(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_project_holdouts(
    cfg,
    *,
    runs: Dict[str, str],
    models: Iterable[str],
) -> pd.DataFrame:
    """Load and concatenate holdout CSVs for a single project (cfg)."""
    results_dir = Path(cfg.RESULTS_DIR)
    frames = []
    for m in models:
        run_id = runs[m]
        df_m = load_all_holdouts_legacy(
            results_dir=results_dir,
            run_id=run_id,
            model=m,
            project=cfg.PROJECT_NAME,
        )
        frames.append(df_m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_reports_data(
    cfg_cod,
    cfg_mp,
    *,
    runs: Dict[str, str],
    models: Iterable[str],
    domains: Iterable[str],
) -> dict:
    """Assemble the prediction tables used by the reporting notebook.

    Returns a dict with:
      - df_cod: COD holdouts (all models)
      - df_mp: MP holdouts (all models)
      - combined_holdouts: stacked COD+MP with 'source' column
      - df_clean_all: combined_holdouts with MP domains M/O removed (used in stats_by_n_elements)
    """
    models = list(models)
    domains = list(domains)

    df_cod = load_project_holdouts(cfg_cod, runs=runs, models=models)
    df_mp = load_project_holdouts(cfg_mp, runs=runs, models=models)

    # Align domains: map N -> I in both tables (as used in plots)
    if "domain" in df_mp.columns:
        df_mp = df_mp.copy()
        df_mp["domain"] = df_mp["domain"].replace({"N": "I"})
    if "domain" in df_cod.columns:
        df_cod = df_cod.copy()
        df_cod["domain"] = df_cod["domain"].replace({"N": "I"})

    combined_holdouts = _stack_sources(df_mp, df_cod, source_col="source")

    # Filter out MP M/O domains for df_clean_all (as used in stats_by_n_elements)
    if {"source", "domain"} <= set(combined_holdouts.columns):
        mask_drop = (combined_holdouts["source"] == "MP") & (combined_holdouts["domain"].isin(["M", "O"]))
        df_clean_all = combined_holdouts[~mask_drop].copy()
    else:
        df_clean_all = combined_holdouts.copy()

    return {
        "df_cod": df_cod,
        "df_mp": df_mp,
        "combined_holdouts": combined_holdouts,
        "df_clean_all": df_clean_all,
    }


def build_element_tables(
    combined_holdouts: pd.DataFrame,
    *,
    results_dir: str | Path,
    all_results_name: str = "all_results.csv",
    all_elements_name: str = "all_result-ele.csv",
) -> dict:
    """Build per-element table from combined_holdouts and save helper CSVs.

    Parameters
    ----------
    combined_holdouts:
        Stacked dataframe with at least columns: 'elements_cif' and 'source'.
    results_dir:
        Base results directory (e.g. cfg_cod.RESULTS_DIR or cfg_mp.RESULTS_DIR).
    all_results_name:
        Filename for saving combined_holdouts.
    all_elements_name:
        Filename for saving df_ele (per-element table).

    Returns
    -------
    dict with:
      - df_ele: per-element table with one-hot-coded element presence
      - path_all_results: Path to saved all_results CSV
      - path_all_result_ele: Path to saved per-element CSV
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if "elements_cif" not in combined_holdouts.columns:
        raise KeyError("combined_holdouts must contain 'elements_cif' column.")

    df_ele = combined_holdouts.copy()
    df_ele["elements_list"] = df_ele["elements_cif"].astype(str).str.split(",")

    elements_dummies = (
        df_ele["elements_list"]
        .explode()
        .str.strip()
        .pipe(pd.get_dummies)
        .groupby(level=0)
        .sum()
    )

    df_ele = pd.concat([df_ele.drop(columns=["elements_list"]), elements_dummies], axis=1)

    # Save CSVs
    path_all_results = results_dir / all_results_name
    path_all_result_ele = results_dir / all_elements_name

    combined_holdouts.to_csv(path_all_results, index=False)
    df_ele.to_csv(path_all_result_ele, index=False)

    return {
        "df_ele": df_ele,
        "path_all_results": path_all_results,
        "path_all_result_ele": path_all_result_ele,
    }
