"""Define the interface for extracting target labels from structures."""

from __future__ import annotations
from typing import Protocol, Sequence
import pandas as pd

class Labeler(Protocol):
    """A pluggable target/label generator.

    Implementations should:
      - declare which input columns they require
      - add label columns to the dataframe (either by calling existing project functions
        that write to disk, or by operating in-memory).
    """

    @property
    def name(self) -> str: ...

    @property
    def required_columns(self) -> Sequence[str]: ...

    def build(
        self,
        *,
        root_cif: str,
        input_csv: str,
        output_csv: str,
        bad_csv: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Create target labels and write them to output_csv.

        Returns: (df_labeled, df_bad_or_None)
        """
        ...
