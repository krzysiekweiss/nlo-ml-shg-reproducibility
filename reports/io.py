"""Load prediction tables and write numerical report outputs."""

from pathlib import Path
import pandas as pd
from typing import Iterable, Optional, Union


def as_list(x: Optional[Union[str, Iterable[str]]]) -> Optional[list[str]]:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)

def filter_df(
    df: pd.DataFrame,
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    mode: Optional[str] = None,
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    g = df
    if run_id is not None:
        g = g[g["run_id"] == run_id]
    if project is not None:
        g = g[g["project"] == project]
    if mode is not None and "mode" in g.columns:
        g = g[g["mode"] == mode]

    models_l = as_list(models)
    steps_l = as_list(steps)
    domains_l = as_list(domains)

    if models_l is not None:
        g = g[g["model"].isin(models_l)]
    if steps_l is not None:
        g = g[g["step"].isin(steps_l)]
    if domains_l is not None:
        g = g[g["domain"].isin(domains_l)]

    return g

def resolve_run_map(
    df: pd.DataFrame,
    *,
    runs: Optional[dict[str, str]] = None,
    models: Optional[Union[str, Iterable[str]]] = None,
    run_col: str = "run_id",
    model_col: str = "model",
) -> dict[str, str]:
    """Resolve mapping model -> run_id.

    Rule enforced: one run_id per model (otherwise raise).
    """
    models_l = as_list(models)
    if models_l is None:
        models_l = sorted(df[model_col].dropna().unique().tolist()) if model_col in df.columns else []

    if runs is not None:
        missing = [m for m in models_l if m not in runs]
        if missing:
            raise ValueError(f"Missing run_id for models: {missing}. Provide runs={{model: run_id}}.")
        return {m: str(runs[m]) for m in models_l}

    if run_col not in df.columns or model_col not in df.columns:
        raise ValueError("Cannot infer runs from df; provide runs={model: run_id}.")

    out: dict[str, str] = {}
    for m in models_l:
        uniq = sorted(df.loc[df[model_col] == m, run_col].dropna().unique().tolist())
        if len(uniq) == 1:
            out[m] = uniq[0]
        elif len(uniq) == 0:
            raise ValueError(f"Cannot infer run_id for model='{m}' (no values in df). Provide runs={{model: run_id}}.")
        else:
            raise ValueError(
                f"Multiple run_id for model='{m}': {uniq}. According to project rule this is an error; "
                f"provide runs={{model: run_id}} to disambiguate."
            )
    return out


def save_table(
    df: pd.DataFrame,
    *,
    project_dir: Union[str, Path],
    run_id: str,
    name: str,
    subdir: str = "tables",
) -> Path:
    """Save DataFrame to <project_dir>/results/<run_id>/<subdir>/<name>.csv"""
    project_dir = Path(project_dir)
    out_dir = project_dir / "results" / run_id / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved table as: {out_path}")
    return out_path


def load_all_holdouts_legacy(
    *,
    results_dir: Union[str, Path],
    run_id: str,
    model: str,
    project: Optional[str] = None,
    mode: str = "final",
    steps: Optional[Iterable[str]] = None,
    domains: Optional[Iterable[str]] = None,
    warn: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate legacy holdout / test / oof CSVs.

    Supported legacy filename templates:
      test_preds_<MODEL>_domain_<DOMAIN>_<RUN_ID>_TT.csv
      oof_<MODEL>_domain_<DOMAIN>_<RUN_ID>_CV.csv

    Return predictions with model, chemical-domain and run identifiers.
    """
    results_dir = Path(results_dir)
    steps_set = set(steps) if steps is not None else None
    domains_set = set(domains) if domains is not None else None

    rows: list[pd.DataFrame] = []
    for p in sorted(results_dir.glob("*.csv")):
        name = p.name

        split = None

        if name.startswith("test_preds_") and name.endswith("_TT.csv"):
            split = "final"   # TT ≈ final test split
            core = name.replace("test_preds_", "").replace("_TT.csv", "")
        elif name.startswith("oof_") and name.endswith("_CV.csv"):
            split = "cv"
            core = name.replace("oof_", "").replace("_CV.csv", "")
        else:
            continue

        # core format:
        # <MODEL>_domain_<DOMAIN>_<RUN_ID>
        parts = core.split("_domain_")
        if len(parts) != 2:
            continue

        model_got = parts[0]
        rest = parts[1].split("_", 1)
        if len(rest) != 2:
            continue

        domain = rest[0]
        run_part_id = rest[1]


        if run_part_id != run_id:
            continue
        if model_got != model:
            continue
        if split != mode:
            continue
        if domains_set is not None and domain not in domains_set:
            continue
        if steps_set is not None and "na" not in steps_set:
            continue

        df = pd.read_csv(p)

        df["run_id"] = run_part_id
        df["project"] = project
        df["mode"] = split
        df["model"] = model_got
        df["step"] = "na"
        df["domain"] = domain
        df["timestamp"] = None
        df["source_file"] = p.name

        rows.append(df)

    if not rows:
        if warn:
            msg = (
                f"[WARN] No legacy holdout files matched: run_id={run_id}, model={model}, "
                f"mode={mode}, project={project}, steps={steps}, domains={domains} "
                f"in {results_dir}"
            )
            print(msg)
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)
