"""Extract complete feature-importance tables from fitted model bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Mapping

import pandas as pd

from core.bundle import load_bundle, get_final_estimator, tree_feature_importance


def _model_path(proj_dir: str | Path, model: str, domain: str, run_id: str, *, split: str = "TT") -> Path:
    """Return the model alias path for the specified model, domain, run and split."""
    return Path(proj_dir) / "models" / f"model_{model}_domain_{domain}_{run_id}_{split}.pkl"


def importance_table(
    cfg,
    *,
    runs: Mapping[str, str],              # e.g. {"rf": "rf005_ht", "lgbm": "lgbm004_ht"}
    domains: Iterable[str],               # e.g. ["I", "M", "O"]
    split: str = "TT",
    top_n: Optional[int] = 15,            # None => all
    save: bool = True,
    project_dir: str | Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build and (optionally) save feature-importance tables per domain.
    Output: one CSV per domain, written under: <project_dir>/results/<run_id>/tables/

    Note:
      Each model maps to one run identifier. A separate table is written
      for each requested domain within that run directory.
    """
    proj_dir = Path(project_dir or cfg.PROJECT_DIR)
    proj_name = getattr(cfg, "project_name", getattr(cfg, "PROJECT_NAME", "UNKNOWN_PROJECT"))

    def _n_tag() -> str:
        return "all" if top_n is None else f"top{int(top_n)}"

    logs: list[dict] = []

    for model, run_id in runs.items():
        for domain in domains:
            model_file = _model_path(proj_dir, model, domain, run_id, split=split)

            if not model_file.exists():
                print(f"[WARN] Missing model file: {model_file}")
                continue

            pipe, feature_cols, _ = load_bundle(model_file)
            est = get_final_estimator(pipe)
            imp_df, imp_kind = tree_feature_importance(est, feature_cols)

            out = imp_df.copy() if top_n is None else imp_df.head(int(top_n)).copy()
            out.insert(0, "model", model)
            out.insert(1, "run_id", run_id)
            out.insert(2, "domain", domain)
            out.insert(3, "importance_kind", imp_kind)

            if verbose:
                k = 10 if top_n is None else min(10, int(top_n))
                print("=" * 80)
                print(f"[IMPORTANCE] model={model} | run_id={run_id} | domain={domain} | kind={imp_kind}")
                print(out.head(k).to_string(index=False))

            if save:
                out_dir = proj_dir / "results" / run_id / "tables"
                out_dir.mkdir(parents=True, exist_ok=True)

                name = f"feature_importance__magpie__domain_{domain}__{imp_kind}__{_n_tag()}"
                out_path = out_dir / f"{name}.csv"
                out.to_csv(out_path, index=False)

                logs.append({
                    "proj": proj_name,
                    "model": model,
                    "run_id": run_id,
                    "domain": domain,
                    "importance_kind": imp_kind,
                    "csv_path": str(out_path),
                })
            else:
                logs.append({
                    "proj": proj_name,
                    "model": model,
                    "run_id": run_id,
                    "domain": domain,
                    "importance_kind": imp_kind,
                    "csv_path": None,
                })

            print(f"[OK] {proj_name} | {model} | run={run_id} | domain={domain} | {imp_kind}")

    return pd.DataFrame(logs)
