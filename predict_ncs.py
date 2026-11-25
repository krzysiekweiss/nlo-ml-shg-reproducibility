"""Screen a chemical composition with a fitted PAPER_V2 NCS model."""

import argparse
import json
import math
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = {"mp": "MP_ALL", "cod": "COD_CANON_V2"}
DOMAINS = {"inorganic": "N", "organic": "O", "metal-organic": "M"}


def magpie_features(formula):
    """Compute the Magpie features used during training for one formula."""
    import pandas as pd
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    try:
        composition = Composition(formula, strict=True)
    except ValueError as exc:
        raise ValueError(f"Invalid chemical formula {formula!r}. {exc}") from exc
    if not composition or any(
        not math.isfinite(amount) or amount <= 0 for amount in composition.values()
    ):
        raise ValueError(f"Supply a formula with positive, finite amounts: {formula!r}.")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"MagpieData\(impute_nan=False\):", category=UserWarning,
        )
        featuriser = ElementProperty.from_preset("magpie", impute_nan=False)
    values = featuriser.featurize(composition)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Magpie data are incomplete for {formula!r}.")
    return pd.DataFrame([values], columns=featuriser.feature_labels())


def model_run_id(dataset, domain, model):
    """Resolve the final fitted run for the selected model."""
    registry = json.loads((ROOT / "config" / "paper_v2_runs.json").read_text(encoding="utf-8"))
    return next(
        entry["final"] for entry in registry
        if (entry["project"], entry["domain"], entry["model"])
        == (DATASETS[dataset], DOMAINS[domain], model)
    )


def predict_formula(formula, dataset, domain, model, models_dir):
    """Return the NCS score and the class at the paper's fixed threshold."""
    run_id = model_run_id(dataset, domain, model)
    model_path = models_dir / f"model_{run_id}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file is missing: {model_path}\n"
            "Download the matching file from the project's GitHub Releases."
        )

    from core.ml_utils import load_bundle

    features = magpie_features(formula)
    bundle = load_bundle(model_path)
    if bundle.run_id != run_id or bundle.domain != DOMAINS[domain]:
        raise ValueError(f"The model metadata do not match the selected run: {model_path}.")
    if len(bundle.feature_cols) != len(features.columns) or set(bundle.feature_cols) != set(features):
        raise ValueError("The available Magpie descriptors do not match the fitted model.")

    positive_class = list(bundle.pipeline.classes_).index(1)
    with warnings.catch_warnings():
        # The fitted imputer passes an array to LightGBM after the input schema has been checked.
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        score = float(bundle.pipeline.predict_proba(features[bundle.feature_cols])[0, positive_class])
    threshold = bundle.threshold if bundle.threshold is not None else 0.5
    return {
        "formula": formula,
        "dataset": dataset,
        "domain": domain,
        "model": model,
        "run_id": run_id,
        "ncs_score": score,
        "ncs_probability_percent": round(100 * score, 2),
        "predicted_class": "NCS" if score >= threshold else "CS",
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formula", required=True, help="Chemical formula, for example LiNbO3.")
    parser.add_argument("--dataset", required=True, choices=DATASETS, help="Training database.")
    parser.add_argument("--domain", required=True, choices=DOMAINS, help="Chemical domain.")
    parser.add_argument("--model", required=True, choices=("rf", "lgbm"), help="Learning algorithm.")
    parser.add_argument(
        "--models-dir", type=Path, default=ROOT / "models", metavar="DIR",
        help="Path to the downloaded trusted model files (default: models/ beside the script).",
    )
    args = parser.parse_args()
    try:
        result = predict_formula(args.formula.strip(), args.dataset, args.domain, args.model, args.models_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    except ImportError as exc:
        parser.error(f"Model dependencies could not be imported. See the README for installation. {exc}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
