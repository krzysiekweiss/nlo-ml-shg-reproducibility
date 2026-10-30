# Publication results and source mapping

The export command creates `paper/`, `supplement/`, `input_manifest.csv` and `output_manifest.csv`. Table and figure names are stable publication-facing names; source paths and hashes are preserved in the manifests. Numerical tables are machine-readable sources for manuscript formatting, not generated Word layouts.

## Main tables

| Output under `paper/tables/` | Source and computation |
|---|---|
| `Table01_dataset_counts.csv` | Six split tables: records, train/test sizes, composition groups, target counts and feature count |
| `Table02_cv_metrics.csv` | Twelve OOF files: eight metrics, means and sample SD across five actual folds; checked against saved metrics |
| `Table03_test_metrics_CI.csv` | Twelve final holdouts: full-test metrics joined to the completed group-bootstrap intervals; MP-O/M marked `not_run` |
| `Table04a_cross_all_test.csv` | All 72 cases of the unfiltered cross-evaluation protocol |
| `Table04b_cross_composition_disjoint_test.csv` | All 72 cases after removing source-training composition overlap; source for main Table 4 |

## Main figures

| Output | Generator | Input scope |
|---|---|---|
| `Fig01a_ROC_MP`, `Fig01b_ROC_COD` | Notebook 04, `reports/roc_plots.py` | MP-I and COD-I/O/M; RF and LightGBM |
| `Fig02a_threshold_RF_MP_I`, `Fig02b_threshold_RF_COD_I` | Notebook 04, `reports/holdout_threshold.py` | RF inorganic test probabilities, exploratory recall target 0.8 |
| `Fig03a_n_elements_MP`, `Fig03b_n_elements_COD` | Notebook 04, `reports/stats_by_n_elements.py` | Distinct-element bins 1–2, 3, 4, 5+ |
| `Fig04a_auc_vs_Z_RF_MP_I`, `Fig04b_auc_vs_Z_RF_COD_M` | Notebook 04, `reports/metric_vs_Z.py` | Overlapping element-containing subsets, n≥30, both classes, Z≤92 |
| `Fig05a_periodic_blocks_RF_MP`, `Fig05b_periodic_blocks_RF_COD` | Notebook 04, `reports/plot_periodic_block.py` | Distributions of per-element RF AUC across s/p/d/f blocks; all domains |
| `Fig06a_importance_RF_MP`, `Fig06b_importance_RF_COD` | Notebook 05, `reports/plot_importance.py` | RF impurity importance; shared top 15 selected by inorganic ranking |

PNGs are placed in `paper/figures/`. Source CSVs accompany ROC, element-count, per-element and periodic-block panels in `paper/tables/`. Separate legends are `Fig02_legend.png` and `Fig04_legend.png`. `Fig02_legend_s.png` in older working folders is an editorial variant, not a separate analysis required to reproduce the curves.

## Supplementary material

`FigS01_auc_vs_Z_RF_COD_I.png` and `FigS02_auc_vs_Z_RF_COD_O.png` use notebook 04's additional per-element panels. `FigS03_importance_LGBM_MP.png` and `FigS04_importance_LGBM_COD.png` use notebook 05's LightGBM panels. LightGBM importance is split count, even though compatibility table filenames contain `tree_mdi`.

The exporter provides these numerical sources under `supplement/tables/`:

| File | Use in supplementary information |
|---|---|
| `best_hyperparameters.csv` | Selected RF/LightGBM parameters; exact search distributions are documented in Methods |
| `cv_fold_metrics.csv` | All five fold-level scores for every model/domain; means/SD are in main Table02 CSV |
| `bootstrap_details.csv` | Full-test points, percentile intervals and bootstrap diagnostics |
| `cross_overlap_summary.csv` | Source-training overlap and remaining target sizes |
| `exploratory_RF_thresholds.csv` | Recall-0.8 operating points for both inorganic RF models |
| `additional_Np_Pu_metrics.csv` | Qualifying Np/Pu subsets outside the historical Z≤92 plotting range |
| `run_identifiers.csv` | Exact final and CV identifiers for all model/domain combinations |
| `importance_<model>_<dataset>_<domain>.csv` | All 132 feature values for each final model, including undisplayed domains |

Supplementary counts and additional held-out/CV metrics are included in the extended main-table CSVs. Thus the same numerical exports support Supplementary Tables S1 and S3–S10 without maintaining separate hand-edited numeric copies. Table S2's search distributions are fixed in `core/hyperopt_tuning.py` and described in Methods.

The bootstrap intervals are conditional on fitted models and fixed test-set sampling. Per-element subsets overlap; block plots are not distributions over disjoint sets of crystals. Neither figure type establishes statistical significance between model families.
