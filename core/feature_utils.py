"""Select numeric feature columns from prepared datasets."""

from __future__ import annotations
import pandas as pd


def detect_feature_prefix(df: pd.DataFrame, step: str | None = None) -> str:
    """
    Detect feature column prefix based on featurization step and dataframe columns.

    Notes:
      - Some datasets use 'atom' (no underscore) for simple features
      - Others use 'atom_' for fractional/expanded atom features
    """
    s = (step or "").strip().lower()

    def _first_present(prefixes: tuple[str, ...]) -> str | None:
        for p in prefixes:
            if any(c.startswith(p) for c in df.columns):
                return p
        return None

    if s in ("magpie", "default"):
        p = _first_present(("MagpieData", "Magpie"))
        if p:
            return p
        raise ValueError("Expected Magpie features, but none found ('MagpieData' or 'Magpie').")

    if s in ("simple",):
        p = _first_present(("atom", "atom_"))
        if p:
            return p
        raise ValueError("Expected simple atom features, but none found ('atom' or 'atom_').")

    if s in ("frac", "fractional", "f"):
        p = _first_present(("atom_", "atom"))
        if p:
            return p
        raise ValueError("Expected fractional atom features, but none found ('atom_' or 'atom').")

    # Fallback autodetection
    p = _first_present(("atom_", "atom", "MagpieData", "Magpie"))
    if p:
        return p

    raise ValueError("Could not detect feature prefix from dataframe columns.")
