"""Resolve run metadata and prepare verified prediction or model aliases."""
from __future__ import annotations

import filecmp
import json
import shutil
from pathlib import Path
from typing import Mapping


def select_run(cfg, model: str, domain: str, *, mode: str = "final",
               paper_run: str = "PAPER_V2", overrides: Mapping | None = None) -> str:
    """Return the unique matching run, or validate an explicitly selected run."""
    key = (cfg.PROJECT_NAME, model, domain)
    explicit = (overrides or {}).get(key)
    prefix = f"{paper_run}__{cfg.PROJECT_NAME}__{mode}__{model}__magpie__{domain}__"
    results = Path(cfg.RESULTS_DIR)
    paths = [results / f"meta_{explicit}.json"] if explicit else sorted(results.glob(f"meta_{prefix}*.json"))
    matches = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        run = meta.get("run_id", "")
        identity = (meta.get("project"), meta.get("model"), meta.get("domain"), meta.get("mode"), meta.get("step"))
        if identity != (cfg.PROJECT_NAME, model, domain, mode, "magpie") or not run.startswith(prefix):
            raise ValueError(f"Metadata identity does not match {key}, mode={mode}: {path}")
        if path.name != f"meta_{run}.json" or (explicit and run != explicit):
            raise ValueError(f"Metadata run_id does not match its selected filename: {path}")
        matches.append(run)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {mode} run for {key}; found {len(matches)}: {matches}. "
            "Set RUN_OVERRIDES[(project, model, domain)] to the intended run_id."
        )
    return matches[0]


def selected_runs(configs, *, paper_run="PAPER_V2", mode="final", overrides=None,
                  models=("rf", "lgbm"), domains=("N", "O", "M")):
    """Resolve each project/model/domain exactly once."""
    return [(cfg, model, domain, select_run(cfg, model, domain, mode=mode,
             paper_run=paper_run, overrides=overrides))
            for cfg in configs for model in models for domain in domains]


def _copy_if_identical_or_absent(source: Path, destination: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
            return
        raise FileExistsError(
            f"Report alias belongs to different data: {destination}. "
            "Use another report alias or archive the old alias explicitly."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_report_inputs(configs, runs, *, kind="holdout", paper_run="PAPER_V2", overrides=None):
    """Create verified legacy aliases from current per-domain training outputs.

    kind='holdout' prepares CSV aliases for notebooks 04/06. kind='model'
    prepares trusted local model bundles for notebook 05, including joblib
    sidecars. Existing differing aliases are never silently replaced.
    """
    if kind not in {"holdout", "model"}:
        raise ValueError("kind must be 'holdout' or 'model'")
    selection = selected_runs(configs, paper_run=paper_run, overrides=overrides,
                              models=tuple(runs))
    pairs = []
    records = []
    for cfg, model, domain, run in selection:
        if kind == "holdout":
            root = Path(cfg.RESULTS_DIR)
            source = root / f"holdout_{run}.csv"
            target = root / f"test_preds_{model}_domain_{domain}_{runs[model]}_TT.csv"
        else:
            root = Path(cfg.MODELS_DIR)
            source = root / f"model_{run}.pkl"
            target = root / f"model_{model}_domain_{domain}_{runs[model]}_TT.pkl"
        pairs.append((source, target))
        if kind == "model":
            for extra in sorted(root.glob(source.name + "*")):
                if extra != source:
                    pairs.append((extra, root / (target.name + extra.name[len(source.name):])))
        records.append({"project": cfg.PROJECT_NAME, "model": model, "domain": domain,
                        "source_run": run, "alias": target.name})
    # Validate the complete selection before creating any aliases.
    for source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() and not (target.is_file() and filecmp.cmp(source, target, shallow=False)):
            raise FileExistsError(f"Conflicting report alias: {target}")
    for source, target in pairs:
        _copy_if_identical_or_absent(source, target)
    for cfg in configs:
        output = Path(cfg.RESULTS_DIR) / "paper" / f"selected_{kind}_runs.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([r for r in records if r['project'] == cfg.PROJECT_NAME], indent=2), encoding="utf-8")
    return records
