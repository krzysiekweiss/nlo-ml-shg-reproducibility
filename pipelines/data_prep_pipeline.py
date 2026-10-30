"""Orchestrate CIF parsing, symmetry labelling and dataset filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from targets.base import Labeler

@dataclass(frozen=True)
class DataPrepArtifacts:
    raw_listing: str
    raw_listing_sampled: Optional[str]
    metadata_cif: str
    labels: str
    chem_class: str
    groups: str
    master: str
    cleaned_master: str

class DataPrepPipeline:
    """Step-based data preparation pipeline.

    Steps:
      1. Build raw CIF listing
      2. Sample raw CIF listing (optional)
      3. Build CIF metadata
      4. Build labels (via Labeler plugin)
      5. Build chemical class
      6. Group by composition
      7. Build master dataset
      8. Filter canonical-group conflicts + write cleaned master and audit reports
    """

    def __init__(self, cfg, *, labeler: Labeler, root_cif: Optional[str] = None):
        self.cfg = cfg
        self.root_cif = root_cif or cfg.ROOT_CIF
        self.labeler = labeler

        # Default artifact locations from project_config
        self.artifacts = DataPrepArtifacts(
            raw_listing=getattr(cfg, "DATASET_RAW_CIF_LISTING", getattr(cfg, "DATASET_RAW", None)),
            raw_listing_sampled=getattr(cfg, "DATASET_RAW_CIF_LISTING_SAMPLED", None),
            metadata_cif=cfg.DATASET_METADATA_CIF,
            labels=getattr(cfg, "DATASET_SYMMETRY_LABELS", None),
            chem_class=cfg.DATASET_CHEM_CLASS,
            groups=cfg.DATASET_GROUPS,
            master=cfg.DATASET_MASTER,
            cleaned_master=cfg.DATASET_CLEANED_MASTER,
        )
        if self.artifacts.raw_listing is None:
            raise ValueError("cfg must define DATASET_RAW_CIF_LISTING (or DATASET_RAW).")
        if self.artifacts.labels is None:
            raise ValueError("cfg must define DATASET_SYMMETRY_LABELS (labels output).")

        self._raw_listing_sampled_runtime: Optional[str] = self.artifacts.raw_listing_sampled


    def _selected_raw_listing_path(self) -> str:
        sampled = self._raw_listing_sampled_runtime
        if sampled and Path(sampled).exists():
            return sampled
        return self.artifacts.raw_listing

    def _default_sampled_path(self) -> str:
        # Fallback if cfg.DATASET_RAW_CIF_LISTING_SAMPLED is not defined
        n = getattr(self.cfg, "RAW_SAMPLE_N", None)
        seed = getattr(self.cfg, "RAW_SAMPLE_SEED", None)
        base = Path(self.artifacts.raw_listing)
        return str(base.with_name(f"{base.stem}__sample_n{n}_seed{seed}{base.suffix}"))


    # public API
    def run(self, *, steps=None, from_step=None, to_step=None):
        from core.notebook_logger import NotebookLogger
        import os

        if steps is None:
            from_step = 1 if from_step is None else from_step
            to_step = 8 if to_step is None else to_step
            steps = list(range(from_step, to_step + 1))

        logger = NotebookLogger(
            project_dir=self.cfg.PROJECT_DIR,
            prefix=self.cfg.PROJECT_NAME,
        )

        logger.start(header_lines=[
            f"Project        : {self.cfg.PROJECT_NAME}",
            'Pipeline       : DataPrepPipeline',
            f"Steps          : {steps}",
            f"CWD            : {os.getcwd()}",
            f"ROOT_CIF       : {self.root_cif}",
            f"Output dataset : {self.artifacts.cleaned_master}",
        ])

        try:
            for s in steps:
                fn = getattr(self, f"step_{s}", None)
                if fn is None:
                    raise ValueError(f"Unknown step: {s}")
                fn()
        finally:
            logger.stop()

        return self.artifacts.cleaned_master

    # helpers
    def _ensure_exists(self, path: str, what: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact for {what}: {p}")
        return p

    def _timed(self, name: str):
        # Uses existing timing helper if available; otherwise a no-op context manager
        try:
            from core.log_timing import LogTiming
            return LogTiming(name, enabled=getattr(self.cfg, "LOG_TIME", False))
        except Exception:
            from contextlib import nullcontext
            return nullcontext()

    # steps
    def step_1(self):
        from core import data_preparation_csv as dp
        from core.step_print import step_header, kv, done, StepTimer

        step_header(1, "Build raw CIF listing")
        kv("scan_dir", self.root_cif)
        kv("output", self.artifacts.raw_listing)

        with StepTimer() as t, self._timed("build_raw_cif_listing"):
            dp.build_raw_cif_listing(
                root_cif=self.root_cif,
                output_csv=self.artifacts.raw_listing,
            )

        n_files = len(pd.read_csv(self.artifacts.raw_listing))
        done({
            "cif_files_found": n_files,
            "elapsed_s": f"{t.elapsed:.1f}",
        })

    def step_2(self):
        """
        Optional: sample N rows from the full raw listing to create a deterministic subset.
        Downstream steps will use the sampled listing if present.
        """
        self._ensure_exists(self.artifacts.raw_listing, "STEP 2 (sampling input)")

        n = getattr(self.cfg, "RAW_SAMPLE_N", None)
        seed = getattr(self.cfg, "RAW_SAMPLE_SEED", 123)

        if n is None:
            print("[STEP 2] Sample raw CIF listing")
            print("  → RAW_SAMPLE_N is None; skipping sampling")
            return

        out = self.artifacts.raw_listing_sampled or self._default_sampled_path()
        self._raw_listing_sampled_runtime = out

        print("[STEP 2] Sample raw CIF listing")
        print(f"  → input     : {self.artifacts.raw_listing}")
        print(f"  → output    : {out}")
        print(f"  → sample_n  : {n}")
        print(f"  → seed      : {seed}")

        df = pd.read_csv(self.artifacts.raw_listing)
        rows_in = len(df)

        if rows_in == 0:
            raise ValueError("Raw listing is empty; cannot sample.")

        if n > rows_in:
            print(f"  → note      : requested n={n} > rows_in={rows_in}; sampling all rows")
            df_out = df.sample(n=rows_in, random_state=seed).reset_index(drop=True)
        else:
            df_out = df.sample(n=n, random_state=seed).reset_index(drop=True)

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out, index=False)

        print("  ✓ done")
        print(f"    rows_in   : {rows_in}")
        print(f"    rows_out  : {len(df_out)}")

    def step_3(self):
        input_listing = self._selected_raw_listing_path()
        from core import data_preparation_csv as dp
        self._ensure_exists(input_listing, "STEP 3 (metadata input)")

        bad_csv = str(Path(self.cfg.DATASETS_DIR) / "dataset_metadata_bad.csv")

        print("[STEP 3] Build CIF metadata")
        print(f"  → input  : {input_listing}")
        print(f"  → output : {self.artifacts.metadata_cif}")
        print(f"  → bad_csv: {bad_csv}")

        with self._timed("build_cif_metadata"):
            dp.build_cif_metadata(
                root_cif=self.root_cif,
                input_csv=input_listing,
                output_csv=self.artifacts.metadata_cif,
                bad_csv=bad_csv,
            )

    def step_4(self):
        from core.step_print import step_header, kv, done, StepTimer

        self._ensure_exists(self.artifacts.metadata_cif, "STEP 4 (labels input)")
        bad_csv = str(Path(self.cfg.DATASETS_DIR) / "dataset_symmetry_bad.csv")

        step_header(4, f"Apply target labels: {self.labeler.name}")
        kv("input", self.artifacts.metadata_cif)
        kv("output", self.artifacts.labels)
        kv("bad_csv", bad_csv)

        with StepTimer() as t, self._timed(f"labels_{self.labeler.name}"):
            self.labeler.build(
                root_cif=self.root_cif,
                input_csv=self.artifacts.metadata_cif,
                output_csv=self.artifacts.labels,
                bad_csv=bad_csv,
            )

        df = pd.read_csv(self.artifacts.labels)
        stats = {
            "rows_labeled": len(df),
        }

        if "centrosymmetric" in df.columns:
            vc = df["centrosymmetric"].value_counts(dropna=False)
            stats["centro"] = int(vc.get(True, 0))
            stats["noncentro"] = int(vc.get(False, 0))

        stats["elapsed_s"] = f"{t.elapsed:.1f}"
        done(stats)


    def step_5(self):
        self._ensure_exists(self.artifacts.metadata_cif, "STEP 5 (chem class input)")
        print("[STEP 5] Chemical classification")
        from core.chem_class import chem_classify_csv
        with self._timed("chem_classify_csv"):
            chem_classify_csv(path=self.artifacts.metadata_cif, save_path=self.artifacts.chem_class)

    def step_6(self):
        from core import data_preparation_csv as dp
        print("[STEP 6] Group by composition")
        self._ensure_exists(self.artifacts.metadata_cif, "STEP 6 (group input: metadata)")
        self._ensure_exists(self.artifacts.labels, "STEP 6 (group input: labels)")
        with self._timed("group_by_composition_csv"):
            dp.group_by_composition_csv(
                metadata_csv=self.artifacts.metadata_cif,
                symmetry_csv=self.artifacts.labels,
                output_csv=self.artifacts.groups,
            )

    def step_7(self):
        from core import data_preparation_csv as dp
        # Inputs used in notebook when building master
        raw_listing = self._selected_raw_listing_path()
        self._ensure_exists(raw_listing, "STEP 7 (master input: raw listing)")
        self._ensure_exists(self.artifacts.metadata_cif, "STEP 7 (master input: metadata)")
        self._ensure_exists(self.artifacts.labels, "STEP 7 (master input: labels)")
        self._ensure_exists(self.artifacts.chem_class, "STEP 7 (master input: chem class)")
        self._ensure_exists(self.artifacts.groups, "STEP 7 (master input: groups)")

        with self._timed("build_master_dataset"):
            df1 = pd.read_csv(raw_listing)
            df2 = pd.read_csv(self.artifacts.metadata_cif)
            df3 = pd.read_csv(self.artifacts.labels)
            df4 = pd.read_csv(self.artifacts.chem_class)
            dfg = pd.read_csv(self.artifacts.groups)

            df_master = dp.build_master_dataset(
                datasets=[
                    df1,
                    df2,
                    df3,
                    df4[["filepath", "chem_class"]],
                    dfg[["filepath", "group_id", "group_id_comp", "label_type",
                         "canonical_group_id", "canonical_exact_key",
                         "canonical_match_type", "canonical_tolerance"]],
                ],
                key="filepath",
                add_symmetry_conflict=True,
                cif_symbol_col="sg_symbol_cif",
                spglib_symbol_col="sg_symbol_spglib",
            )
            df_master.to_csv(self.artifacts.master, index=False)

    def step_8(self):
        """Filter conflicts using step-6 groups, without reparsing formulas.

        C/N conflicts are evaluated on the complete master before any row
        filtering. Every member of a conflicting group is removed globally.
        Reports describe this step only; feature/fold filtering is subsequent.
        """
        import hashlib
        from core.step_print import step_header, kv, done, StepTimer

        self._ensure_exists(self.artifacts.master, "STEP 8 (input: new canonical master)")
        step_header(8, "Canonical group conflicts and clean dataset")
        kv("input", self.artifacts.master)
        kv("output", self.artifacts.cleaned_master)
        report_dir = Path(self.cfg.DATASETS_DIR) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        with StepTimer() as t, self._timed("canonical_conflict_filtering"):
            df = pd.read_csv(self.artifacts.master, dtype={"id": str})
            required = ["id", "filepath", "chem_formula_cif", "centrosymmetry",
                        "chem_class", "symmetry_conflict", "group_id", "group_id_comp",
                        "canonical_group_id", "canonical_exact_key",
                        "canonical_match_type", "canonical_tolerance"]
            missing = [c for c in required if c not in df]
            if missing:
                raise ValueError(f"Rerun steps 6 and 7; master lacks: {missing}")
            key = "canonical_group_id"
            valid_group = df[key].notna() & df[key].astype(str).str.strip().ne("")
            group_rows = df.loc[valid_group]
            checks = []

            def check(name, passed):
                checks.append({"check": name, "passed": bool(passed)})

            check("unique_nonmissing_ids", df["id"].notna().all() and not df["id"].duplicated().any())
            check("unique_filepaths", df["filepath"].notna().all() and not df["filepath"].duplicated().any())
            check("group_aliases_equal", (group_rows["group_id"] == group_rows[key]).all()
                  and (group_rows["group_id_comp"] == group_rows[key]).all())
            check("exact_keys_present", group_rows["canonical_exact_key"].notna().all())
            check("exact_key_in_one_group", group_rows.groupby("canonical_exact_key")[key].nunique().le(1).all())
            check("valid_match_types", group_rows["canonical_match_type"].isin(["exact", "tolerance"]).all())
            check("one_tolerance", not group_rows.empty and group_rows["canonical_tolerance"].notna().all()
                  and group_rows["canonical_tolerance"].nunique() == 1
                  and group_rows["canonical_tolerance"].between(0, 1, inclusive="left").all())
            exact_counts = group_rows.groupby(key)["canonical_exact_key"].nunique()
            expected_type = group_rows[key].map(exact_counts).gt(1).map({True: "tolerance", False: "exact"})
            check("match_type_matches_group", group_rows["canonical_match_type"].eq(expected_type).all())

            # The master retains original labels; grouping's replacement U
            # labels are deliberately not used to discover C/N conflicts.
            label_counts = df.loc[df["centrosymmetry"].isin(["C", "N"])].groupby(key)["centrosymmetry"].nunique()
            conflict_ids = label_counts.index[label_counts > 1]
            df["centrosymmetry_conflict"] = df[key].isin(conflict_ids)
            if "point_group" in df:
                df["point_group_conflict"] = df.groupby(key)["point_group"].transform("nunique").gt(1)
            symmetry = df["symmetry_conflict"].astype(str).str.strip().str.lower()
            check("valid_symmetry_flags", symmetry.isin(["true", "false", "1", "0", "1.0", "0.0"]).all())
            df["symmetry_conflict"] = symmetry.isin(["true", "1", "1.0"])
            flags = pd.DataFrame({
                "missing_canonical_group": ~valid_group,
                "conflicting_CN_group": df["centrosymmetry_conflict"],
                "symmetry_conflict": df["symmetry_conflict"],
                "invalid_CN_label": ~df["centrosymmetry"].isin(["C", "N"]),
            }, index=df.index)
            reason = pd.Series("kept", index=df.index, dtype=object)
            for name in flags:
                reason.loc[reason.eq("kept") & flags[name]] = name
            removed = flags.any(axis=1)
            clean = df.loc[~removed].copy()
            check("row_balance", len(df) == len(clean) + int(removed.sum()))
            check("no_CN_conflicts_after", clean.groupby(key)["centrosymmetry"].nunique().le(1).all())
            check("all_conflict_members_removed", not clean[key].isin(conflict_ids).any())
            validation = pd.DataFrame(checks)
            validation.to_csv(report_dir / "group_validation.csv", index=False)
            if not validation["passed"].all():
                raise ValueError(f"Group validation failed: {validation.loc[~validation.passed, 'check'].tolist()}; see {report_dir}")

            audit_cols = ["id", "filepath", "chem_formula_cif", "chem_class", "centrosymmetry",
                          "canonical_group_id", "canonical_exact_key", "canonical_match_type"]
            rejected = df.loc[removed, audit_cols].copy()
            rejected["primary_reason"] = reason.loc[removed]
            for name in flags:
                rejected[name] = flags.loc[removed, name]
            rejected.to_csv(report_dir / "excluded_records.csv", index=False)
            df.loc[df["centrosymmetry_conflict"], audit_cols].to_csv(report_dir / "conflicting_groups.csv", index=False)
            # Member list (not a claim that every pair is directly within tolerance).
            df.loc[df["canonical_match_type"].eq("tolerance"), audit_cols].to_csv(report_dir / "tolerance_matches.csv", index=False)
            groups = group_rows.groupby(key).agg(
                records_before=("id", "size"), exact_keys=("canonical_exact_key", "nunique"),
                formula_spellings=("chem_formula_cif", "nunique"),
                match_type=("canonical_match_type", "first"),
                example_formula=("chem_formula_cif", "first"), example_id=("id", "first"),
            )
            groups["records_after"] = clean.groupby(key).size().reindex(groups.index, fill_value=0)
            groups["records_removed"] = groups["records_before"] - groups["records_after"]
            groups["conflicting_CN"] = groups.index.isin(conflict_ids)
            spellings_per_exact = group_rows.groupby("canonical_exact_key")["chem_formula_cif"].nunique()
            merged_exact = group_rows["canonical_exact_key"].map(spellings_per_exact).gt(1)
            groups["contains_exact_merge"] = merged_exact.groupby(group_rows[key]).any()
            groups.to_csv(report_dir / "canonicalization_report.csv")

            summaries = []
            domains = df["chem_class"].fillna("UNKNOWN").astype(str)
            for domain in ["ALL"] + sorted(domains.unique().tolist()):
                mask = pd.Series(True, index=df.index) if domain == "ALL" else domains.eq(domain)
                before, after = df.loc[mask], clean.loc[clean.index.intersection(df.index[mask])]
                ids = before[key].dropna().unique()
                local_groups = groups.reindex(ids)
                row = {"domain": domain, "records_before": len(before), "records_after": len(after),
                       "records_removed": int((mask & removed).sum()),
                       "groups_before": len(ids), "groups_after": after[key].nunique(),
                       "conflicting_CN_groups": int(local_groups["conflicting_CN"].sum()),
                       "groups_with_exact_merges": int(local_groups["contains_exact_merge"].sum()),
                       "tolerance_groups": int(local_groups["match_type"].eq("tolerance").sum())}
                for name in flags:
                    row["removed_" + name] = int((mask & reason.eq(name)).sum())
                for label in ["C", "N"]:
                    row[label + "_before"] = int(before["centrosymmetry"].eq(label).sum())
                    row[label + "_after"] = int(after["centrosymmetry"].eq(label).sum())
                summaries.append(row)
            summary = pd.DataFrame(summaries)
            summary.to_csv(report_dir / "dataset_summary.csv", index=False)
            digest = hashlib.sha256()
            with open(self.artifacts.master, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            (report_dir / "parameters.txt").write_text(
                f"project={self.cfg.PROJECT_NAME}\nmaster={self.artifacts.master}\n"
                f"master_sha256={digest.hexdigest()}\n"
                f"atomic_fraction_atol={group_rows['canonical_tolerance'].iloc[0]}\n"
                "rtol=0; exact rational normalization first; same element support; connected components\n"
                "C/N conflicts assessed globally before filtering; all members removed.\n"
                "Removal counts use the first matching reason, in excluded_records column order.\n"
                "Other boolean reasons may overlap. Groups can span domains; domain group counts are not additive.\n"
                "tolerance_matches lists component members, not all pairwise matches.\n"
                "Scope: master to cleaned master. Feature and fold exclusions are separate.\n"
                "No split seed here: folds are generated by notebook 02.\n", encoding="utf-8")
            clean.to_csv(self.artifacts.cleaned_master, index=False)
            print(summary.to_string(index=False))
            kv("reports", report_dir)
        done({"rows_input": len(df), "rows_clean": len(clean),
              "records_removed": int(removed.sum()), "elapsed_s": f"{t.elapsed:.1f}"})
