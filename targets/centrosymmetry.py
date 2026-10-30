"""Extract centrosymmetry labels with spglib through the CIF processing interface."""

from __future__ import annotations

from .base import Labeler

class CentrosymmetryLabeler(Labeler):
    """Centrosymmetry labeler using existing project symmetry code.

    This is a thin adapter around core.data_preparation_csv.build_symmetry_labels.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return "centrosymmetry"

    @property
    def required_columns(self):
        # build_symmetry_labels expects at least 'filepath' in the metadata csv
        return ["filepath"]

    def build(
        self,
        *,
        root_cif: str,
        input_csv: str,
        output_csv: str,
        bad_csv: str | None = None,
    ):
        from core import data_preparation_csv as dp  # imported here to keep dependency localized

        df_sym, df_bad = dp.build_symmetry_labels(
            root_cif=root_cif,
            input_csv=input_csv,
            output_csv=output_csv,
            bad_csv=bad_csv,
        )
        return df_sym, df_bad
