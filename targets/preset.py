"""Define target mappings and task specifications for model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Literal
import pandas as pd

TaskType = Literal["classification", "regression"]

@dataclass(frozen=True)
class TargetSpec:
    """
    Target specification for ML training.

    Attributes
    ----------
    source_col : str
        Name of the column in the input DataFrame containing the raw target.
    out_col : str
        Name of the output column to be used by the pipeline (typically "y").
    task : TaskType
        "classification" or "regression".
    mapping : Optional[Dict]
        Optional value mapping for classification targets.
    allow_values : Optional[Sequence]
        Optional allowed values check for classification targets.
    """
    source_col: str
    out_col: str = "y"
    task: TaskType = "classification"
    mapping: Optional[Dict] = None
    allow_values: Optional[Sequence] = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and materialize the target column in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        pd.DataFrame
            A copy of df with df[out_col] created.
        """
        if self.source_col not in df.columns:
            raise KeyError(f"Missing target source column: '{self.source_col}'")

        out = df.copy()

        if self.task == "classification":
            y = out[self.source_col]
            if self.allow_values is not None:
                bad = set(y.dropna().unique()) - set(self.allow_values)
                if bad:
                    raise ValueError(f"Unexpected values in '{self.source_col}': {sorted(bad)}")

            if self.mapping is not None:
                out[self.out_col] = y.map(self.mapping)
            else:
                out[self.out_col] = y

            if out[self.out_col].isna().any():
                raise ValueError(
                    f"Target mapping produced NaNs in '{self.out_col}'. "
                    f"Check mapping/values in '{self.source_col}'."
                )

            out[self.out_col] = out[self.out_col].astype(int)

        elif self.task == "regression":
            out[self.out_col] = pd.to_numeric(out[self.source_col], errors="coerce")
            if out[self.out_col].isna().any():
                raise ValueError(
                    f"Regression target contains NaNs after conversion from '{self.source_col}'."
                )
            out[self.out_col] = out[self.out_col].astype(float)

        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return out

def make_target_spec(name: str) -> TargetSpec:
    """
    Create a TargetSpec from a preset name.
    """
    name = name.strip()

    if name == "centrosymmetry_CN":
        return TargetSpec(
            source_col="centrosymmetry",
            out_col="y",
            task="classification",
            mapping={"C": 0, "N": 1},
            allow_values=["C", "N"],
        )

    if name == "shg_y_log_fro":
        return TargetSpec(
            source_col="y_log_fro",
            out_col="y",
            task="regression",
        )

    if name == "shg_y_log_max":
        return TargetSpec(
            source_col="y_log_max",
            out_col="y",
            task="regression",
        )

    raise ValueError(f"Unknown target preset: '{name}'")
