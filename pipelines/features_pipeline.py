"""Generate composition descriptors with validated feature-cache reuse."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

class FeaturesPipeline:
    """Feature generation pipeline operating only on the cleaned master dataset."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.input_clean = cfg.DATASET_CLEANED_MASTER

        # Feature files are read from the configured dataset paths.
        self.output_magpie = getattr(cfg, "DATASET_FEATURES", str(Path(cfg.DATASETS_DIR) / "dataset_features_magpie.csv"))
        self.output_simple = getattr(cfg, "DATASET_FEATURES_SIMPLE", str(Path(cfg.DATASETS_DIR) / "dataset_features_simple.csv"))
        self.output_frac = getattr(cfg, "DATASET_FEATURES_FRAC", str(Path(cfg.DATASETS_DIR) / "dataset_features_frac.csv"))

    def _timed(self, name: str):
        try:
            from core.log_timing import LogTiming
            return LogTiming(name, enabled=getattr(self.cfg, "LOG_TIME", False))
        except Exception:
            from contextlib import nullcontext
            return nullcontext()

    def run(self, *, steps: Sequence[str] = ("magpie", "simple", "frac"),
            aggregate: bool = True, force: bool = False):
        """Reuse cached values, generate missing ids, report failed records.

        No cache or force=True generates all input records. Existing generators
        run on only the missing subset otherwise. Only records still lacking
        finite features after generation are excluded; malformed cache/schema
        errors and generator exceptions remain explicit errors.
        """
        from copy import copy
        from tempfile import TemporaryDirectory
        import numpy as np
        import pandas as pd
        from matminer.featurizers.composition import ElementProperty

        if not Path(self.input_clean).exists():
            raise FileNotFoundError(f"Missing cleaned master dataset: {self.input_clean}")
        current = pd.read_csv(self.input_clean, dtype={"id": str})
        if "id" not in current or current["id"].isna().any() or current["id"].duplicated().any():
            raise ValueError("cleaned master: id must be present, non-null and unique")
        self.aggregate = aggregate
        outputs = {"magpie": self.output_magpie, "simple": self.output_simple, "frac": self.output_frac}
        for s in ([steps] if isinstance(steps, str) else steps):
            s = s.lower()
            if s not in outputs:
                raise ValueError(f"Unknown feature step: {s}")
            path = Path(outputs[s])
            if s == "magpie":
                expected = ElementProperty.from_preset("magpie").feature_labels()
                prefix = "Magpie"
            else:
                prefix = "atom"
                if s == "simple":
                    expected = [f"atom{i}_{name}" for i in range(20)
                                for name in ["Z", "Mass", "Radius", "Electronegativity", "NumAtoms"]]
                elif aggregate:
                    names = ["Z", "Mass", "Radius", "EN", "NumAtoms"] + [f"FC_{j}" for j in range(5, 49)]
                    expected = ["atom_Z_sum", "atom_Mass_sum", "atom_NumAtoms_total"] + [
                        f"atom_{name}_{stat}" for name in names for stat in ("mean", "std", "min", "max")]
                else:
                    subshells = ["1s", "2s", "2p", "3s", "3p", "3d"] + [f"{n}{orb}" for n in range(4, 8) for orb in "spdf"]
                    names = ["Z", "Mass", "Radius", "EN", "NumAtoms"] + [
                        f"FracCount_{sub}_{stat}" for sub in subshells for stat in ("e", "frac")]
                    expected = [f"atom{i}_{name}" for i in range(20) for name in names]

            def validate_table(table, name):
                if "id" not in table or table["id"].isna().any() or table["id"].duplicated().any():
                    raise ValueError(f"{s} {name}: id must be present, non-null and unique")
                actual = [c for c in table if c.startswith(prefix)]
                if set(actual) != set(expected) or len(actual) != len(expected):
                    raise ValueError(f"{s} {name}: incompatible feature columns")

            reused = pd.DataFrame(columns=["id"] + expected)
            if path.exists() and not force:
                cached = pd.read_csv(path, dtype={"id": str}, float_precision="round_trip")
                validate_table(cached, "cache")
                if not np.isfinite(cached[expected].to_numpy(dtype=float)).all():
                    raise ValueError(f"{s}: cache contains NaN/Inf; use force=True to regenerate")
                matched = current.loc[current["id"].isin(cached["id"])]
                selected = cached.set_index("id").loc[matched["id"]].reset_index()
                if "chem_formula_cif" not in selected or not selected["chem_formula_cif"].equals(matched["chem_formula_cif"].reset_index(drop=True)):
                    raise ValueError(f"{s}: formulas changed for cached ids; use force=True")
                reused = selected[["id"] + expected]

            pending = current.loc[~current["id"].isin(reused["id"])].copy()
            generated = pd.DataFrame(columns=["id"] + expected)
            nonfinite_ids = set()
            path.parent.mkdir(parents=True, exist_ok=True)
            # Temporary outputs keep the old cache intact if generation fails.
            with TemporaryDirectory(prefix=f".{s}_features_", dir=path.parent) as temp:
                temp = Path(temp)
                if len(pending):
                    print(f"[FEATURES] {s}: reuse={len(reused)}, generate={len(pending)}")
                    generator_input = pending.drop(columns=expected + ["composition"], errors="ignore").copy()
                    # Prefix temporary ids so CSV readers in existing generators
                    # preserve the original spelling, including leading zeros.
                    generator_input["id"] = "feature_id:" + generator_input["id"]
                    generator_input.to_csv(temp / "input.csv", index=False)
                    worker = copy(self)
                    worker.input_clean = str(temp / "input.csv")
                    setattr(worker, f"output_{s}", str(temp / "generated.csv"))
                    getattr(worker, f"step_{s}")()
                    generated = pd.read_csv(temp / "generated.csv", dtype={"id": str}, float_precision="round_trip")
                    validate_table(generated, "generator output")
                    if not generated["id"].str.startswith("feature_id:").all():
                        raise ValueError(f"{s}: generator returned unexpected ids")
                    generated["id"] = generated["id"].str[len("feature_id:"):]
                    if not generated["id"].isin(pending["id"]).all():
                        raise ValueError(f"{s}: generator returned ids outside the requested subset")
                    finite = np.isfinite(generated[expected].to_numpy(dtype=float)).all(axis=1)
                    nonfinite_ids = set(generated.loc[~finite, "id"])
                    generated = generated.loc[finite, ["id"] + expected]

                values = pd.concat([reused, generated], ignore_index=True)
                result = current.loc[current["id"].isin(values["id"])].drop(columns=expected, errors="ignore")
                result = result.merge(values, on="id", how="left", sort=False, validate="one_to_one")
                excluded = pending.loc[~pending["id"].isin(generated["id"])].copy()
                excluded["exclusion_reason"] = excluded["id"].map(
                    lambda value: "nonfinite_generated_features" if value in nonfinite_ids else "not_returned_by_generator")
                if len(result) + len(excluded) != len(current):
                    raise ValueError(f"{s}: feature row balance failed")
                result.to_csv(temp / "complete.csv", index=False)
                (temp / "complete.csv").replace(path)

            reports = path.parent / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            excluded.to_csv(reports / f"feature_exclusions_{s}.csv", index=False)
            pd.DataFrame([{
                "feature_step": s, "records_input": len(current),
                "records_reused": len(reused), "records_sent_to_generator": len(pending),
                "records_generated_valid": len(generated), "records_excluded": len(excluded),
                "records_output": len(result), "force": force, "feature_file": str(path),
            }]).to_csv(reports / f"feature_summary_{s}.csv", index=False)
            print(f"[FEATURES] {s}: reused={len(reused)}, generated={len(generated)}, "
                  f"excluded={len(excluded)}, saved={len(result)} -> {path}")
        return outputs

    def step_magpie(self):
        from core.composition_features  import magpie_feature_generation

        print("[FEATURES] Magpie features")
        print(f"  → input  : {self.input_clean}")
        print(f"  → output : {self.output_magpie}")
        with self._timed("magpie_feature_generation"):
            magpie_feature_generation(input_csv=self.input_clean, output_csv=self.output_magpie)

    def step_simple(self):
        from core.composition_features  import simple_feature_generation_padded

        print("[FEATURES] Simple features")
        print(f"  → input  : {self.input_clean}")
        print(f"  → output : {self.output_simple}")
        with self._timed("simple_feature_generation_padded"):
            simple_feature_generation_padded(input_csv=self.input_clean, output_csv=self.output_simple)

    def step_frac(self):
        from core.composition_features  import frac_count_feature_generation_padded

        print("[FEATURES] Fractional-count features")
        print(f"  → input     : {self.input_clean}")
        print(f"  → output    : {self.output_frac}")
        print(f"  → aggregate : {self.aggregate}")

        with self._timed("frac_count_feature_generation_padded"):
            frac_count_feature_generation_padded(
                input_csv=self.input_clean,
                output_csv=self.output_frac,
                aggregate_feat=self.aggregate,
            )
