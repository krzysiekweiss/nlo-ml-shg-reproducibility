"""Load fitted model bundles and export ranked feature-importance tables."""

from pathlib import Path
from . import ml_utils as mlu
import pandas as pd


def load_bundle(path: str | Path):

    """Return the fitted pipeline, ordered feature names and target specification.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a trusted serialized model bundle.
    """
    bundle = mlu.load_bundle(path)
    pipe = bundle.pipeline
    feature_cols = bundle.feature_cols
    target_spec = bundle.target_spec

    return pipe, feature_cols, target_spec


def get_final_estimator(pipe):
    # Read the estimator stored by a scikit-learn pipeline.
    """Return the final estimator from a pipeline, or the supplied estimator itself."""
    if hasattr(pipe, "_final_estimator"):
        return pipe._final_estimator
    # Use the final pipeline step when the estimator attribute is unavailable.
    if hasattr(pipe, "steps"):
        return pipe.steps[-1][1]
    return pipe


def tree_feature_importance(est, feature_cols, *, top_n=15):
    # Estimators exposing feature_importances_.
    """Return the complete ranked feature table and its importance identifier.

    Parameters
    ----------
    est : fitted estimator
        Estimator exposing feature_importances_ or a LightGBM booster.
    feature_cols : sequence of str
        Feature names in the order used during training.
    top_n : int
        Accepted by the interface; truncation is applied by the caller.

    The generic feature_importances_ branch is checked first. For LightGBM
    with its default importance setting, these values are split counts.
    """
    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
        kind = "tree_mdi"
        df = pd.DataFrame({"feature": feature_cols, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True), kind

    # LightGBM sklearn API
    if hasattr(est, "booster_"):
        booster = est.booster_
        imp = booster.feature_importance(importance_type="gain")
        kind = "lgbm_gain"
        df = pd.DataFrame({"feature": feature_cols, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True), kind

    raise TypeError("Final estimator does not expose tree-based feature importance.")
