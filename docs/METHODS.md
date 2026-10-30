# Methods for the PAPER_V2 experiments

This document records the procedures implemented for the `PAPER_V2` results from `COD_CANON_V2` and `MP_ALL`, with completed bootstrap results validated on 14 October 2026. It describes the archived experimental protocol. Paths below are relative to this repository or to the external project-artifact root. See REPRODUCIBILITY.md for the input contract and the distinction between archived-result reproduction and a new dataset build.

## Data and target

The chemical domains are inorganic, organic and metal-organic, encoded as `N`, `O` and `M` in the datasets. Reports display the inorganic domain as `I`. These domain codes are distinct from the symmetry labels: centrosymmetric structures are labelled `C` and mapped to 0; noncentrosymmetric structures are labelled `N` and mapped to 1. Predicted probabilities refer to the latter class.

The preparation pipeline obtains symmetry labels with spglib using `symprec=0.01` angstrom. For the present COD update, previously generated CIF metadata and symmetry labels were reused from `COD_all`; chemical classification, composition grouping and filtering produced `COD_CANON_V2`. Thus the updated composition tolerance does not imply that crystallographic symmetry was recomputed with a different positional tolerance. `MP_ALL` retained its existing dataset and split files. The new COD canonicalization rules must not be retrospectively attributed to MP without evidence that the MP dataset was rebuilt with those rules.

Sources: `notebooks/01_prepare_data.ipynb`, `core/data_preparation_csv.py`, `targets/centrosymmetry.py` and the retained dataset artifacts.

## COD composition equivalence and exclusions

CIF composition coefficients are parsed as exact rational numbers. Element amounts are combined and reduced to a primitive integer ratio, so exactly proportional formulas share an exact composition key without rounding or a numerical tolerance. Subsequently, compositions with the same set of elements are compared using atomic fractions. Two exact keys are directly matched when the maximum absolute difference between corresponding atomic fractions is at most `1e-5`, with zero relative tolerance.

The final canonical groups are connected components of these matches. Consequently, the endpoints of a chain can differ by more than `1e-5`; the tolerance bounds direct pairwise matches, not the diameter of an entire component. `canonical_match_type` describes a complete group: `exact` when it contains one exact key and `tolerance` when multiple exact keys were joined. The split grouping column `group_id` is an alias of `canonical_group_id` for this COD dataset.

Groups containing both original symmetry labels C and N are identified globally, before record filtering, and all members of each conflicting group are excluded. Missing canonical groups, symmetry-conflict flags and invalid target labels are also filtered. Removal reports use an ordered primary reason for mutually exclusive counts; auxiliary reason flags may overlap. Domain-specific group counts are not necessarily additive.

The saved COD preparation report contains 521,901 master records, 334,929 canonical groups, 12,497 groups with conflicting C/N labels, 275 groups containing exact merges and 193 tolerance groups. Filtering leaves 375,417 records; 146,484 records are removed at this stage. These are master-to-cleaned counts, not additional exclusions to add to earlier parsing failures. The feature report records 375,325 reused valid feature rows and 92 records sent to the generator without a valid result, leaving 375,325 feature-complete records.

Sources: `group_by_composition_and_fix_labels` in `core/data_preparation_csv.py`; step 8 of `pipelines/data_prep_pipeline.py`; `COD_CANON_V2/datasets/reports/{dataset_summary.csv,parameters.txt,feature_summary_magpie.csv}`.

## Features and cache

Models use the 132 composition descriptors returned by matminer's `ElementProperty.from_preset("magpie")`. When a cache is available and `force=False`, matching records reuse their saved feature values. Cache validation checks unique non-null identifiers, the expected feature-column set, finite values and agreement of the source formula for each reused identifier. Records absent from the cache are submitted to the existing feature generator. With no cache or `force=True`, all requested records are submitted. Records still lacking a complete finite feature vector after generation are excluded and reported; malformed cache schemas and generator exceptions are explicit errors.

The Magpie statistic `avg_dev` is a mean absolute deviation, not a standard deviation. Figure labels use μ for the atomic-fraction-weighted mean and d for the atomic-fraction-weighted mean absolute deviation from that mean (MAD). Descriptors of elemental ground-state space-group numbers are composition-derived elemental statistics, not the space group of the target crystal.

Sources: `pipelines/features_pipeline.py`, `core/composition_features.py` and saved feature-column lists for each model.

## Splits and model evaluation

Splits are generated independently within each chemical domain using `StratifiedGroupKFold`, with the symmetry label as the stratification target and composition `group_id` as the grouping variable. The configuration uses five shuffled outer partitions with seed 42; outer fold 0 is the held-out test set. The remaining approximately 80% of records are divided into five shuffled inner folds with the same seed. A composition group cannot span training and test or multiple inner validation folds within that domain. Grouping makes the record proportions approximately, rather than exactly, 80/20.

Hyperparameter selection uses inner fold 0 for validation and the other four inner folds for training; all `is_test=1` records are excluded. The selected parameters are subsequently used for five-fold cross-validation over the non-test records and for a final fit on all non-test records. Test evaluation uses only the fixed held-out set. CV mean and sample standard deviation summarize the five fold-level scores; they are not confidence intervals. Because tuning used one of these inner validation folds, the reported CV is not a fully nested estimate of the entire hyperparameter-selection procedure.

ROC-AUC is calculated from positive-class probabilities. Balanced accuracy and Matthews correlation coefficient use threshold 0.5 unless an output explicitly describes an exploratory threshold analysis. Thresholds inspected or selected using held-out labels must not be presented as independently validated operating points.

Sources: `notebooks/02_create_splits.ipynb`, `core/folds.py`, `notebooks/03_train_models.ipynb`, `core/rf_train.py`, `core/lgbm_train.py` and `core/ml_utils.py`.

The synchronized prediction identifiers and target labels were checked against the local fold files. The saved source and target fold hashes for all 72 cross-check tasks match those local files. The verified record counts are:

| Dataset | Domain in figures | All records | Non-test records | Test records | Composition groups |
|---|---|---:|---:|---:|---:|
| COD_CANON_V2 | I | 80,347 | 64,271 | 16,076 | 62,639 |
| COD_CANON_V2 | O | 84,035 | 67,169 | 16,866 | 65,500 |
| COD_CANON_V2 | M | 210,943 | 168,619 | 42,324 | 194,073 |
| MP_ALL | I | 116,942 | 93,372 | 23,570 | 94,780 |
| MP_ALL | O | 560 | 451 | 109 | 488 |
| MP_ALL | M | 2,626 | 2,107 | 519 | 2,432 |

All six datasets contain 132 feature columns and five non-test validation folds. CV CSVs contain an additional row labelled `ALL`, which summarizes pooled out-of-fold predictions and must be excluded when calculating a mean and standard deviation across the five folds.

## Hyperparameter optimization

Each model/domain experiment is configured for 50 Hyperopt TPE evaluations, optimizing validation ROC-AUC, with random seed 42 and 20 startup trials. For the initial completed COD RF N/O runs, the number of startup trials was not explicitly stored in the original run record; 20 is the historical default represented by the retained code and compatibility checks. It should not be described as an independently logged observation for those two runs.

| Model | Parameter | Search specification |
|---|---|---|
| RF | n_estimators | quniform(100, 1500, 50) |
| RF | max_depth | quniform(5, 60, 1) |
| RF | max_features | choice(sqrt, log2, None) |
| RF | min_samples_split | quniform(2, 10, 1) |
| RF | min_samples_leaf | quniform(1, 5, 1) |
| LightGBM | n_estimators | quniform(200, 2000, 50) |
| LightGBM | learning_rate | log-uniform from 0.01 to 0.2 |
| LightGBM | num_leaves | quniform(15, 255, 4) |
| LightGBM | max_depth | quniform(3, 16, 1) |
| LightGBM | min_child_samples | quniform(5, 50, 1) |
| LightGBM | subsample, colsample_bytree | uniform from 0.6 to 1.0 |
| LightGBM | reg_alpha, reg_lambda | log-uniform from 1e-8 to 10 |

The quniform entries above specify the actual Hyperopt distributions, including their quantization, rather than an assumed inclusive integer sequence. Legacy trial CSVs encode RF `max_features` as a choice index: 0=sqrt, 1=log2, 2=None. The stored best-parameter registry supplies the parameters for subsequent training.

Tuning uses unweighted classifiers, `n_jobs=-1` and `random_state=42`; LightGBM additionally sets `verbosity=-1`. The CV/final classifier builders use `class_weight="balanced"` by default. RF's builder defaults to `n_jobs=4`; LightGBM's defaults to `n_jobs=-1`, and explicit registry parameters can override defaults. Therefore tuning and final training do not use identical class-weight settings. This discrepancy is preserved in PAPER_V2 and must be disclosed, rather than silently described as a single identical estimator configuration.

The current tuning implementation records completed fits and checkpoints the Hyperopt Trials object and random generator state. Resume compatibility is checked against the experiment configuration and source data. This is a reproducibility mechanism, not a change in the search objective.

Sources: `core/hyperopt_tuning.py`, `pipelines/hyper_tuning_pipeline.py`, each project's `tuning/best_params_PAPER_V2.json`, trial CSVs and logs.

## Cross-domain evaluation

`PAPER_V2_CROSS/v0001` evaluates the 12 explicitly selected final models against six held-out target datasets, giving 72 model-source/target combinations. Each combination reports both `all_test` and `composition_disjoint_test`. The latter excludes target test records belonging to composition groups shared with the source model's training data, using exact rational grouping followed by the same `1e-5` atomic-fraction matching rule and connected components. Overlap exclusions and sample sizes accompany the metrics. Results from the two protocols must remain separately labelled.

The cross-check uses target mapping C=0/N=1 and threshold 0.5. Saved state records the model, source and target fold hashes, feature schema, protocol and artifact hashes. With `FORCE=False`, complete matching tasks can be reused; `FORCE=True` starts a new version. Cross-domain results do not involve retraining the selected models.

Sources: `notebooks/07_cross_evaluation.ipynb` and `COD_CANON_V2/results/cross/PAPER_V2_CROSS/v0001/{state.json,cross_metrics.csv,cross_overlap_summary.csv}`.

## Feature importance and descriptive plots

RF importance is the estimator's impurity-based feature importance. The shared displayed top 15 is selected from the first requested domain; values for all displayed domains are then retrieved from their complete feature tables. If shown, Other is the sum outside this same shared list. These panels are not independent top-15 rankings for each domain.

The current LightGBM extraction accesses `feature_importances_` through the generic branch and writes `tree_mdi` in the filename. For the default LightGBM importance setting these values are split counts, not RF impurity importance or gain. Interpret these tables as split counts; the supplementary figure captions identify the measure explicitly. The regenerated RF and LightGBM figures use μ and d after each property name and omit panel titles. Their values and feature rankings were preserved; RF importance and LightGBM split counts remain different measures.

Per-element metrics evaluate the subset of records containing an element, so subsets overlap. The current periodic-block report groups these per-element AUC values by the element's s/p/d/f block and reports their distribution or median/IQR. It does not partition crystals by their highest periodic block, and its summed per-element counts are not counts of distinct crystals. Bins for the number of distinct elements are 1-2, 3, 4 and 5+; counts are shown once per dataset/domain, not summed across models. The current ROC and distinct-element-count MP panels retain MP-I and omit MP-O/M, although all six domains were trained. The selected periodic-block RF panels include all three domains for both projects. Each box describes variation across element-specific AUC values, not uncertainty in one fitted model metric.

Sources: `core/bundle.py`, `reports/plot_importance.py`, `reports/plot_periodic_block.py`, `reports/stats_by_n_elements.py`, `reports/report_data.py`.

## Test-set uncertainty

The PAPER_V2 composition-group bootstrap completed on 14 October 2026 covers RF and LightGBM for COD-I/O/M and MP-I. Each of the 24 metric jobs (AUC, MCC and BACC for eight model/dataset cases) produced 2,000 valid replicates with seed 42. MP-O/M were disabled in notebook 06 and have no bootstrap intervals from this run.

For each replicate, G composition groups are drawn uniformly with replacement from the G groups in the fixed test set. Every record belonging to a drawn group is retained, with the multiplicity of that group draw. The number of records can therefore vary across replicates. This retains within-composition dependence while evaluating the record-level metric. The group counts are 12,527 for COD-I, 13,100 for COD-O, 38,814 for COD-M and 18,955 for MP-I.

The original full-test metric is stored as test_value. The 95% percentile interval is given by the 2.5th and 97.5th percentiles of the replicate scores, stored as ci_low and ci_high. The bootstrap mean is a separate diagnostic and does not replace the original test estimate. MCC and balanced accuracy use the fixed threshold 0.5. Single-class resamples would be skipped; none were skipped in the 24 completed jobs. Point estimates and interval endpoints should remain distinct, because the intervals need not be symmetric around the point estimate.

These intervals are conditional on the fitted model and fixed test-set sampling framework. They do not quantify variation from retraining or tuning. Checkpoints store completed attempts, replicate scores and random-generator state; fingerprint checks cover the test labels/probabilities/groups and the bootstrap configuration. Progress is logged and saved every 100 attempts. With force=False, matching completed jobs are reused and interrupted jobs resume; force=True restarts the affected checkpoint.

Read-only validation confirmed the canonical holdout aliases, all 24 checkpoint fingerprints, full-test metrics and percentile endpoints from saved scores. No replicates were regenerated during validation. The selected table retains point estimates for MP-O/M and explicitly marks their bootstrap_status as not_run.

Sources: notebooks/06_bootstrap.ipynb, reports/bootstrap_auc.py, each project's results/paper/tables/bootstrap_*_summary__proj_*.csv, COD_CANON_V2/results/paper/tables/bootstrap_summary.csv and the associated bootstrap checkpoints/log.
