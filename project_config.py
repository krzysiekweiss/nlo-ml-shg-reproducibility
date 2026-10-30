"""Construct dataset, model, result and log paths for each experiment."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    # Load local paths from the untracked configuration file.
    from private_paths import DATA_DIR, TEST_DIR, COD_DIR, MP_DIR, PROJECTS_BASE
except ImportError as e:
    raise RuntimeError(
        "Missing private_paths.py. Copy private_paths_template.py and configure the local data directories."
    ) from e


# Helper utilities


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _seed_tag(seed: Optional[int]) -> str:
    return f"seed{seed}" if seed is not None else "seedRANDOM"


def _sampled_listing_name(base_listing_path: str, n: int, seed: Optional[int]) -> str:
    """
    Generates a deterministic file name for sampled raw listing.
    Example:
      dataset_raw_cif_listing__sample_n1000_seed123.csv
      dataset_raw_cif_listing__sample_n1000_seedRANDOM.csv
    """
    base = Path(base_listing_path)
    tag = _seed_tag(seed)
    return str(base.with_name(f"{base.stem}__sample_n{n}_{tag}{base.suffix}"))


# Runtime configuration


@dataclass
class RuntimeCfg:
    # Experiment selection.
    PROJECT_NAME: str
    ROOT_CIF: str

    RAW_SAMPLE_N: Optional[int] = None
    RAW_SAMPLE_SEED: Optional[int] = None

    LOG_TIME: bool = True

    # Resolved project directories.
    PROJECT_DIR: str = ""
    DATASETS_DIR: str = ""
    MODELS_DIR: str = ""
    RESULTS_DIR: str = ""
    LOGS_DIR: str = ""

    # dataset artifacts
    DATASET_RAW_CIF_LISTING: str = ""
    DATASET_RAW_CIF_LISTING_SAMPLED: Optional[str] = None

    DATASET_METADATA_CIF: str = ""
    DATASET_SYMMETRY_LABELS: str = ""
    DATASET_CHEM_CLASS: str = ""
    DATASET_GROUPS: str = ""
    DATASET_MASTER: str = ""
    DATASET_CLEANED_MASTER: str = ""

    DATASET_FEATURES: str = ""
    DATASET_FEATURES_SIMPLE: str = ""
    DATASET_FEATURES_FRAC: str = ""

    # folds/splits (domain-specific)
    DATASET_FOLDS_TEMPLATE: str = ""
    DATASET_FOLDS_BAD_TEMPLATE: str = ""
    DATASET_FOLDS_TEMPLATE_S: str = ""
    DATASET_FOLDS_BAD_TEMPLATE_S: str = ""
    DATASET_FOLDS_TEMPLATE_F: str = ""
    DATASET_FOLDS_BAD_TEMPLATE_F: str = ""

    # Convenience aliases (filled after template is set)
    DATASET_FOLDS_N: str = ""
    DATASET_FOLDS_O: str = ""
    DATASET_FOLDS_M: str = ""

    # model artifacts
    MODEL_FILE_TEMPLATE: str = ""
    MODEL_FILE_RUN_TEMPLATE: str = ""
    MODEL_BEST_N: str = ""
    MODEL_BEST_O: str = ""
    MODEL_BEST_M: str = ""
    MODEL_REGISTRY: str = ""

    # Local data and artifact roots.
    DATA_DIR: str = ""
    TEST_DIR: str = ""
    COD_DIR: str = ""
    MP_DIR: str = ""
    PROJECTS_BASE: str = ""


def make_cfg(
    *,
    project_name: str,
    root_cif: str,
    raw_sample_n: Optional[int] = None,
    raw_sample_seed: Optional[int] = None,
    projects_base: str = PROJECTS_BASE,
    log_time: bool = True,
) -> RuntimeCfg:
    """Create artifact directories and resolve paths for one experiment.

    Parameters
    ----------
    project_name : str
        Dataset or experiment directory name below projects_base.
    root_cif : str
        Directory containing the source CIF files.
    raw_sample_n : int or None
        Optional number of CIF records to sample; None selects all records.
    raw_sample_seed : int or None
        Random seed for sampling; None uses a nondeterministic seed.
    projects_base : str
        Parent directory for dataset, model, result and log artifacts.
    log_time : bool
        Enable execution-time reporting in the pipelines.

    Returns
    -------
    RuntimeCfg
        Dataset selection and resolved input and output paths.
    """

    # Validate sampling settings
    if raw_sample_n is not None:
        if not isinstance(raw_sample_n, int) or raw_sample_n <= 0:
            raise ValueError("raw_sample_n must be a positive integer or None.")
        # seed is optional; None => non-deterministic sampling
        if raw_sample_seed is not None and (not isinstance(raw_sample_seed, int) or raw_sample_seed < 0):
            raise ValueError("raw_sample_seed must be a non-negative integer or None.")

    project_dir = str(Path(projects_base) / project_name)
    datasets_dir = _ensure_dir(str(Path(project_dir) / "datasets"))
    models_dir   = _ensure_dir(str(Path(project_dir) / "models"))
    results_dir  = _ensure_dir(str(Path(project_dir) / "results"))
    logs_dir     = _ensure_dir(str(Path(project_dir) / "logs"))

    # Core dataset file names (stable)
    dataset_raw_listing = str(Path(datasets_dir) / "dataset_raw_cif_listing.csv")
    dataset_metadata    = str(Path(datasets_dir) / "dataset_metadata_cif.csv")
    dataset_symmetry    = str(Path(datasets_dir) / "dataset_symmetry_labels.csv")
    dataset_chem        = str(Path(datasets_dir) / "dataset_step4_chem_class.csv")
    dataset_groups      = str(Path(datasets_dir) / "dataset_groups.csv")
    dataset_master      = str(Path(datasets_dir) / "dataset_master.csv")
    dataset_clean       = str(Path(datasets_dir) / "dataset_cleaned_master.csv")

    dataset_features        = str(Path(datasets_dir) / "dataset_features.csv")
    dataset_features_simple = str(Path(datasets_dir) / "dataset_features_s.csv")
    dataset_features_frac   = str(Path(datasets_dir) / "dataset_features_f.csv")

    # Sampled listing path: only if sampling is requested
    dataset_raw_listing_sampled = None
    if raw_sample_n is not None:
        dataset_raw_listing_sampled = _sampled_listing_name(
            base_listing_path=dataset_raw_listing,
            n=raw_sample_n,
            seed=raw_sample_seed,
        )

    folds_template        = str(Path(datasets_dir) / "dataset_splits_folds_{domain}.csv")
    folds_bad_template    = str(Path(datasets_dir) / "dataset_splits_folds_{domain}_bad.csv")
    folds_template_s      = str(Path(datasets_dir) / "dataset_splits_folds_s_{domain}.csv")
    folds_bad_template_s  = str(Path(datasets_dir) / "dataset_splits_folds_s_{domain}_bad.csv")
    folds_template_f      = str(Path(datasets_dir) / "dataset_splits_folds_f_{domain}.csv")
    folds_bad_template_f  = str(Path(datasets_dir) / "dataset_splits_folds_f_{domain}_bad.csv")

    model_file_template = str(Path(models_dir) / "model_{model_type}_domain{domain}.pkl")
    model_file_run_template = str(Path(models_dir) / "model_{model_type}_domain_{domain}_run{run_id}.pkl")

    cfg = RuntimeCfg(
        PROJECT_NAME=project_name,
        ROOT_CIF=root_cif,

        RAW_SAMPLE_N=raw_sample_n,
        RAW_SAMPLE_SEED=raw_sample_seed,
        LOG_TIME=log_time,

        PROJECT_DIR=project_dir,
        DATASETS_DIR=datasets_dir,
        MODELS_DIR=models_dir,
        RESULTS_DIR=results_dir,
        LOGS_DIR=logs_dir,

        DATASET_RAW_CIF_LISTING=dataset_raw_listing,
        DATASET_RAW_CIF_LISTING_SAMPLED=dataset_raw_listing_sampled,

        DATASET_METADATA_CIF=dataset_metadata,
        DATASET_SYMMETRY_LABELS=dataset_symmetry,
        DATASET_CHEM_CLASS=dataset_chem,
        DATASET_GROUPS=dataset_groups,
        DATASET_MASTER=dataset_master,
        DATASET_CLEANED_MASTER=dataset_clean,

        DATASET_FEATURES=dataset_features,
        DATASET_FEATURES_SIMPLE=dataset_features_simple,
        DATASET_FEATURES_FRAC=dataset_features_frac,

        DATASET_FOLDS_TEMPLATE=folds_template,
        DATASET_FOLDS_BAD_TEMPLATE=folds_bad_template,
        DATASET_FOLDS_TEMPLATE_S=folds_template_s,
        DATASET_FOLDS_BAD_TEMPLATE_S=folds_bad_template_s,
        DATASET_FOLDS_TEMPLATE_F=folds_template_f,
        DATASET_FOLDS_BAD_TEMPLATE_F=folds_bad_template_f,

        DATASET_FOLDS_N=folds_template.format(domain="N"),
        DATASET_FOLDS_O=folds_template.format(domain="O"),
        DATASET_FOLDS_M=folds_template.format(domain="M"),

        MODEL_FILE_TEMPLATE=model_file_template,
        MODEL_FILE_RUN_TEMPLATE=model_file_run_template,

        MODEL_BEST_N=str(Path(models_dir) / "model_best_domainN.pkl"),
        MODEL_BEST_O=str(Path(models_dir) / "model_best_domainO.pkl"),
        MODEL_BEST_M=str(Path(models_dir) / "model_best_domainM.pkl"),

        MODEL_REGISTRY=str(Path(models_dir) / "model_registry.csv"),

        # Include local data roots in the runtime configuration.
        DATA_DIR=DATA_DIR,
        TEST_DIR=TEST_DIR,
        COD_DIR=COD_DIR,
        MP_DIR=MP_DIR,
        PROJECTS_BASE=projects_base,
    )

    return cfg


# Provide the default MP configuration for module-level consumers.

cfg = make_cfg(
    project_name="MP_ALL",
    root_cif=MP_DIR,
    raw_sample_n=None,
    raw_sample_seed=None,
    projects_base=PROJECTS_BASE,
    log_time=True,
)
