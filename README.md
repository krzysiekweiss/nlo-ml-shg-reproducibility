# Reproducing the PAPER_V2 analyses

This source implements composition-based prediction of crystal noncentrosymmetry with Random Forest and LightGBM classifiers. The positive class is noncentrosymmetric (`N=1`); centrosymmetric structures are `C=0`. Chemical-domain codes `N`, `O` and `M` denote inorganic, organic and metal-organic compounds; figures display inorganic as `I`.

## Environment and paths

Use Python 3.10 or newer. From the repository root, create an isolated environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp private_paths_template.py private_paths.py
python -m jupyterlab
```

On Windows, activate `.venv\Scripts\Activate.ps1` and copy the template with `Copy-Item private_paths_template.py private_paths.py`. Configure `NLO_DATA_DIR` and `NLO_PROJECTS_BASE`, or edit the local paths in `private_paths.py`. Keep that machine-specific file local. Open notebooks from the repository root or its `notebooks/` directory.

The dependency list specifies direct requirements; it is not a recovered environment lockfile from the original training run. Serialized models require a compatible scikit-learn/LightGBM environment.

## Methods and execution order

| Stage | Entry point | Main inputs and outputs |
|---|---|---|
| CIF processing, labels and 132 Magpie descriptors | `notebooks/01_prepare_data.ipynb` | Raw CIFs or retained intermediate tables; cleaned composition records and features |
| Composition-group splitting | `notebooks/02_create_splits.ipynb` | Feature tables; fixed test sets and five inner folds per domain |
| Parameter search, cross-validation and final fitting | `notebooks/03_train_models.ipynb` | Split tables; parameter registries, models, OOF and held-out predictions |
| ROC, thresholds, element counts, element subsets and periodic blocks | `notebooks/04_generate_reports.ipynb` | Saved held-out predictions; figures and numerical source tables |
| Feature importance | `notebooks/05_feature_importance.ipynb` | Trusted fitted models; complete importance tables and selected figures |
| Composition-group bootstrap | `notebooks/06_bootstrap.ipynb` | Saved held-out predictions and group IDs; percentile intervals from 2,000 resamples |
| Cross-domain and cross-database evaluation | `notebooks/07_cross_evaluation.ipynb` | Twelve final models and six target split tables; 72 cases under two overlap protocols |
| Publication export | `scripts/export_publication.py` | Completed experiment artifacts; verified tables, selected figures and file manifests |

For fixed-parameter refits, skip parameter search and copy each `config/best_params_<project>_PAPER_V2.json` to `<projects-base>/<project>/tuning/best_params_PAPER_V2.json` before running the training cells in notebook 03. To regenerate reports from archived runs, skip notebooks 01-03 and supply the saved inputs listed in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The original final and CV identifiers are listed in `config/paper_v2_runs.json`. Reporting notebooks select runs by project/model/domain metadata. If several matching final runs exist, set `RUN_OVERRIDES[(project, model, domain)]` to the corresponding `final` identifier. In `07_cross_evaluation.ipynb`, the final cell calls the evaluation routine; `FORCE=False` reuses compatible completed tasks.

## Export the publication results

After the required reporting stages, run from the repository root:

```bash
python scripts/export_publication.py --projects-base /path/to/NLO_ML --runs config/paper_v2_runs.json --cross-version v0001 --output /path/to/new_export
```

The output directory must not exist. `--tables-only` exports numerical tables when figure assets are unavailable. The exporter verifies split integrity, prediction identifiers and labels, saved metrics, bootstrap coverage and cross-check provenance. It reads completed artifacts without retraining models or regenerating bootstrap samples.

[RESULTS.md](docs/RESULTS.md) maps the exported files to the manuscript and supplementary information. [METHODS.md](docs/METHODS.md) documents preparation rules, search distributions, weighting, split design and uncertainty estimates.

## Reproduction scope

Exact reported numbers require the original dataset snapshot and split assignments. Rebuilding data from current database downloads does not reproduce the historical study automatically. The MP dataset retains its original preparation; the updated COD canonicalization must not be retroactively attributed to MP.

PAPER_V2 tuning uses one inner validation fold and unweighted classifiers; subsequent five-fold CV and final fits use balanced class weights. CV is not fully nested. Bootstrap intervals cover COD-I/O/M and MP-I; MP-O/M have point estimates only. These choices are preserved in the source.

`SOURCE_MANIFEST.json` records source hashes and the completed validation checks. Code is distributed under [LICENSE](LICENSE).
