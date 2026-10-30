# Reproduction inputs and artifact contract

All paths below are relative to `NLO_PROJECTS_BASE` (or `PROJECTS_BASE` in `private_paths.py`). Both project directories are required: `COD_CANON_V2/` and `MP_ALL/`.

## Choose the reproduction level

1. **Prepare a new dataset:** configure raw COD/MP CIF directories and run 01–02, then 03 onward. This executes the current algorithm on the supplied snapshot. It does not guarantee the archived PAPER_V2 record counts, especially for MP_ALL, whose historical preparation was retained in the reported study.
2. **Refit on the archived dataset:** provide all six original split tables, including descriptors, and run 03 onward. The two included parameter registries permit fixed-parameter refits without repeating tuning. Exact model predictions also depend on the original software environment.
3. **Reproduce reports from saved runs:** provide the original splits, prediction/metadata files, final bundles for 05/cross-check, and completed bootstrap/cross artifacts for export. Training is unnecessary.

A fresh clone alone cannot reproduce the reported numerical results: the data artifacts listed below must be supplied separately. Do not replace missing study data with current database downloads while claiming identical reproduction. The current checkout contains no dataset download service or archived environment lockfile.

## Dataset preparation

Full preparation needs `.cif` files under `COD_DIR` or `MP_DIR`. Notebook 01's `from_cif` mode runs listing, metadata, symmetry, chemical classification, composition grouping, master assembly and conflict filtering.

The optional `resume` mode needs all three files in an explicitly selected directory:

- `dataset_raw_cif_listing.csv`
- `dataset_metadata_cif.csv`
- `dataset_symmetry_labels.csv`

It recomputes classes and the master/filtering stages. Existing canonical groups may be reused only with unchanged metadata, labels and grouping rules. Available Magpie feature caches are validated before reuse. Step 8 writes `datasets/reports/` exclusion and group-validation records; the feature stage adds generation/exclusion summaries. Notebook 02 writes final split tables.

## Training inputs

For each project, retain `datasets/dataset_splits_folds_N.csv`, `_O.csv` and `_M.csv`. Each contains `id`, target `centrosymmetry`, composition `group_id`, `chem_class`, composition/formula metadata, `is_test`, `fold`, outer assignments and the 132 `MagpieData` descriptor columns. Use the full tables rather than reduced reporting CSVs. Training uses the same feature ordering recorded with the models.

The fixed-parameter registry destination is `tuning/best_params_PAPER_V2.json` inside each project. Copies of the study registries are included under `config/`. Completed trial/checkpoint directories are required to resume an interrupted optimization; they are not required for a new run or a fixed-parameter refit. Preserve existing artifacts before using `FORCE=True`.

## Reporting inputs

For each of the 12 final jobs, retain:

- `results/meta_<final_run>.json`
- `results/holdout_<final_run>.csv`
- `results/metrics_<final_run>.csv`
- `results/feature_cols_<final_run>.txt`
- `models/model_<final_run>.pkl` and any joblib sidecar files (05 and cross-check)

For each of the 12 CV jobs, numerical export additionally reads:

- `results/meta_<cv_run>.json`
- `results/oof_<cv_run>.csv`
- `results/metrics_<cv_run>.csv`

The original 24 identifiers are in `config/paper_v2_runs.json`. New runs have new timestamp suffixes. `reports/paper_inputs.py` matches metadata identity, never file modification time. When several candidates exist, set notebook `RUN_OVERRIDES[(project, model, domain)]` to the selected final run. For export, pass a JSON list with `project`, `model`, `domain`, `final` and `cv` fields via `--runs`.

Notebooks 04/06 create compatibility prediction aliases; 05 creates model aliases. Existing aliases must be byte-identical to the selected source. When switching experiments, use distinct aliases or explicitly archive the prior aliases and their derived report folders. Derived figure names are reused within a project, so preserve each completed report set before switching runs.

## Bootstrap and cross-check artifacts

Notebook 06 writes `results/paper/tables/bootstrap_*_summary__proj_*.csv`, per-job progress checkpoints and a combined `COD_CANON_V2/results/paper/tables/bootstrap_summary.csv`. The combined summary contains 24 rows: three metrics for eight model/domain cases, each with 2,000 group-bootstrap replicates, seed 42. Export uses this completed summary and checks point estimates and coverage; it does not regenerate or unpickle checkpoints.

Cross-check writes `COD_CANON_V2/results/cross/PAPER_V2_CROSS/<version>/`. Export reads the completed `state.json`, `cross_metrics.csv` and `cross_overlap_summary.csv` for an explicitly named version. Original version: `v0001`. Preserve its task predictions, overlap records and per-element outputs for independent verification. Source/target fold hashes and source metadata identities must match the selected runs.

## Environment and numerical precision

`requirements.txt` lists direct dependencies and notebook tooling. It is not a historical environment freeze, and minimum versions are not a claim that every permitted combination has been tested. Archive the actual environment (`python -m pip freeze`), Python version, OS/architecture, source revision and input hashes for each new run. Load trusted serialized models only with a compatible environment.

The exporter reads CSV floating-point probabilities using round-trip precision. Default CSV parsing can merge or separate near-identical probabilities and slightly alter average precision. CV means and sample standard deviations are calculated from the five actual folds, excluding pooled `ALL` rows.

The export validates numerical source provenance. Existing PNGs are copied as supplied; their visual identity and correspondence to the selected run must be ensured by running 04/05 with the same selections. Input hashes record which assets were copied.
