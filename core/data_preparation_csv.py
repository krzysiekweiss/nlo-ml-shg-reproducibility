"""Parse CIF files, assign symmetry labels and construct composition groups."""


import os
from pathlib import Path
import contextlib
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import pandas as pd
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*get_structures is deprecated.*",
    category=FutureWarning,
)


# pymatgen CIF parsing: harmless numerical rounding warnings
warnings.filterwarnings(
    "ignore",
    message=".*fractional coordinates rounded to ideal values.*",
    category=UserWarning,
)


def collect_cif_files(root_dir: str, show_progress: bool = True) -> list[str]:
    """Collect all CIF files recursively from a given directory."""
    cif_files: list[str] = []

    walker = os.walk(root_dir)
    if show_progress:
        walker = tqdm(walker, desc="Scanning directories", unit="dir")

    for root, _, files in walker:
        for fname in files:
            if fname.lower().endswith(".cif"):
                cif_files.append(os.path.join(root, fname))

        # Update the number of discovered CIF files.
        if show_progress:
            walker.set_postfix(cif=len(cif_files))

    return cif_files


def build_raw_cif_listing(root_cif: str, output_csv: str) -> pd.DataFrame:
    """
    Build raw CIF listing: relative filepath and an ID derived from filename.
    """

    cif_files = collect_cif_files(root_cif)
    print(f"Found {len(cif_files)} CIF files in {root_cif}")

    rows: list[list[str]] = []
    for full_path in tqdm(cif_files, desc="Building dataset_raw_cif_listing"):
        rel_path = os.path.relpath(full_path, root_cif)
        file_id = os.path.splitext(os.path.basename(full_path))[0]
        rows.append([rel_path, file_id])

    df = pd.DataFrame(rows, columns=["filepath", "id"])
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} with {len(df)} records")

    return df

class TimeoutException(Exception):
    """Raised when a per-task operation exceeds the time limit."""


@contextlib.contextmanager
def time_limit(seconds: float):
    """Context manager enforcing a time limit (float seconds) in a worker process."""
    if seconds is None or seconds <= 0:
        yield
        return

    def handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds:.3f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))

    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_timer[0])
        signal.signal(signal.SIGALRM, old_handler)


def _site_elements(site) -> set[str]:
    """Extract element symbols from a pymatgen site (ordered or disordered)."""
    try:
        sp = site.species  # works for ordered and disordered sites
        if hasattr(sp, "keys"):  # Composition-like mapping
            return {el.symbol for el in sp.keys()}
        if hasattr(sp, "symbol"):  # Element-like
            return {sp.symbol}
    except Exception:
        pass

    # Fallback for older patterns
    try:
        sp2 = site.specie
        if hasattr(sp2, "symbol"):
            return {sp2.symbol}
    except Exception:
        pass

    return set()


def _process_cif_metadata(root_cif: str, rel_path: str, parse_timeout: float = 1.0):
    """
    Helper for parallel CIF metadata extraction with a per-file timeout.

    Returns:
        (rel_path, elements_csv, sg_symbol, sg_number, formula, error, stoich_warn)
    """
    import warnings
    from pymatgen.io.cif import CifParser

    full_path = os.path.join(root_cif, rel_path)

    try:
        with time_limit(parse_timeout):
            # Capture warnings raised during parsing inside the worker process
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")  # Capture warnings regardless of the parent process filters.

                struct = CifParser(full_path).get_structures()[0]

            # Extract "Incorrect stoichiometry" warning (if present)
            stoich_warn = None
            for w in wlist:
                if issubclass(w.category, UserWarning):
                    msg = str(w.message)
                    if "Incorrect stoichiometry" in msg:
                        # Store a single-line warning with a bounded field length.
                        stoich_warn = msg.replace("\n", " ")[:2000]
                        break

        elems = set()
        for site in struct.sites:
            elems |= _site_elements(site)
        elements = ",".join(sorted(elems)) if elems else None

        formula = struct.composition.reduced_formula

        try:
            spg_symbol, spg_number = struct.get_space_group_info()
        except Exception:
            spg_symbol, spg_number = None, None

        return (rel_path, elements, spg_symbol, spg_number, formula, None, stoich_warn)

    except TimeoutException:
        return (rel_path, None, None, None, None, f"parse_timeout_{parse_timeout}s", None)
    except Exception as e:
        return (rel_path, None, None, None, None, f"parse_error:{e}", None)

def build_cif_metadata(
    root_cif: str,
    input_csv: str,
    output_csv: str,
    bad_csv: str,
    max_workers: int = 4,
    fast_timeout: float = 1.0,
    retry_timeouts: bool = False,
    slow_timeout: float = 10.0,
    slow_workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build CIF metadata dataset in parallel from the raw listing.

    Strategy:
        - FAST pass with short timeout (high throughput)
        - Optional SLOW retry pass for FAST timeouts only (does not block all workers)
    """

    df_step1 = pd.read_csv(input_csv)
    rel_paths = df_step1["filepath"].tolist()

    rows = []
    bad_files = []
    timeout_paths = []
    stoich_files = []


    # FAST pass
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_cif_metadata, root_cif, rel_path, fast_timeout): rel_path
            for rel_path in rel_paths
        }
        ok = 0
        bad = 0
        timeouts = 0

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing CIF with CifParser (FAST)",
        )

        for future in pbar:
            rel_path, elements, sg_symbol, sg_number, formula, error, stoich_warn = future.result()

            if error is None:
                rows.append((rel_path, elements, sg_symbol, sg_number, formula))
                ok += 1
            else:
                bad_files.append((rel_path, error))
                bad += 1

                if isinstance(error, str) and error.startswith("parse_timeout_"):
                    timeout_paths.append(rel_path)
                    timeouts += 1

            if stoich_warn is not None:
                stoich_files.append((rel_path, stoich_warn))

            # Update parsing counts.
            pbar.set_postfix(ok=ok, bad=bad, timeout=timeouts)


    print(
        f"FAST pass summary: OK={len(rows)} BAD={len(bad_files)} TIMEOUTS={len(timeout_paths)} "
        f"(workers={max_workers}, timeout={fast_timeout}s)"
    )

    # Optional SLOW retry for FAST timeouts only
    if retry_timeouts and timeout_paths:
        print(
            f"Retrying FAST timeouts (SLOW): n={len(timeout_paths)} "
            f"(workers={slow_workers}, timeout={slow_timeout}s)"
        )

        retried_rows = []
        retried_bad = []

        with ProcessPoolExecutor(max_workers=slow_workers) as executor:
            futures = {
                executor.submit(_process_cif_metadata, root_cif, rel_path, slow_timeout): rel_path
                for rel_path in timeout_paths
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing CIF with CifParser (SLOW retry)",
            ):
                rel_path, elements, sg_symbol, sg_number, formula, error = future.result()
                if error is None:
                    retried_rows.append((rel_path, elements, sg_symbol, sg_number, formula))
                else:
                    retried_bad.append((rel_path, error))

        # Remove FAST-timeout entries from bad_files and replace with retry results
        timeout_set = set(timeout_paths)
        bad_files = [
            x for x in bad_files
            if not (x[0] in timeout_set and isinstance(x[1], str) and x[1].startswith("parse_timeout_"))
        ]

        rows.extend(retried_rows)
        bad_files.extend(retried_bad)

        print(f"SLOW retry summary: recovered={len(retried_rows)} still_bad={len(retried_bad)}")

    # Build outputs
    df_ok = pd.DataFrame(
        rows,
        columns=[
            "filepath",
            "elements_cif",
            "sg_symbol_cif",
            "sg_number_cif",
            "chem_formula_cif",
        ],
    )
    df_bad = pd.DataFrame(bad_files, columns=["filepath", "error"])

    df_ok.to_csv(output_csv, index=False)
    df_bad.to_csv(bad_csv, index=False)

    # Save incorrect stoichiometry list (non-fatal)
    stoich_csv = str(Path(bad_csv).with_name("dataset_metadata_bad_stoichiometry.csv"))
    df_stoich = pd.DataFrame(stoich_files, columns=["filepath", "warning"])
    df_stoich.to_csv(stoich_csv, index=False)

    print(f"Saved {stoich_csv} with {len(df_stoich)} incorrect-stoichiometry records")


    total = len(df_ok) + len(df_bad)
    ok_ratio = (len(df_ok) / total) if total else 0.0

    print(f"Saved {output_csv} with {len(df_ok)} records")
    print(f"Saved {bad_csv} with {len(df_bad)} bad CIF records")
    print(f"OK ratio: {ok_ratio:.2%}")

    return df_ok, df_bad


# Configuration for spglib run


# Skip structures whose unit-cell atom count exceeds this threshold.
MAX_SITES = 500


SPGLIB_TIMEOUT = 5.0

# Manual blacklist of problematic CIF files (basename)
MANUAL_BAD = {
    "mp-979420.cif",

}


CS_GROUPS = {
    2,
    10, 11, 12, 13, 14, 15,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74,
    83, 84, 85, 86, 87, 88,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
    133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
    147, 148,
    162, 163, 164, 165, 166, 167,
    175, 176,
    191, 192, 193, 194,
    200, 201, 202, 203, 204, 205, 206,
    221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
}
NCS_GROUPS = set(range(1, 231)) - CS_GROUPS


def _spglib_get(dataset, key: str):
    """Read spglib dataset field via attribute-first, dict fallback."""
    if dataset is None:
        return None
    if hasattr(dataset, key):
        return getattr(dataset, key)
    try:
        return dataset[key]
    except Exception:
        return None


def process_spglib(
    root_cif: str,
    rel_path: str,
    symprec: float = 0.01,
    max_sites: int = MAX_SITES,
    spglib_timeout: float = SPGLIB_TIMEOUT,
) -> tuple[str, int | None, str | None, str | None, str | None, str | None, list[str]]:

    """
    Analyze symmetry using spglib for a single CIF file.

    Args:
        root_cif: Root directory with CIF files (base for relative paths).
        rel_path: Relative path to a CIF file (relative to root_cif).
        symprec: Symmetry precision parameter for spglib.
        max_sites: Maximum number of atomic sites allowed.
        spglib_timeout: Timeout in seconds for a single spglib call.

    Returns:
        (rel_path, sg_number, sg_symbol, point_group, centrosym, err_msg, quality_warns)

        centrosym is one of:
            'C' - centrosymmetric
            'N' - non-centrosymmetric
            'U' - unknown / unclassified / error
    """
    from pymatgen.io.cif import CifParser
    import spglib

    # defaults
    sg_symbol = None
    sg_number = None
    centrosym = None
    point_group  = "U"
    error = None
    quality_warns: list[str] = []

    full_path = os.path.join(root_cif, rel_path)
    basename = os.path.basename(rel_path)

    # 0. Manual blacklist
    if basename in MANUAL_BAD:
        error = f"manual_blacklist:{basename}"
        return (rel_path, sg_number, sg_symbol, point_group, centrosym, error, quality_warns)

    # 1. Parse CIF
    import warnings

    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            struct = CifParser(full_path).get_structures()[0]

        # capture quality warnings
        quality_warns = []
        for w in wlist:
            if issubclass(w.category, UserWarning):
                msg = str(w.message)
                if (
                    "Incorrect stoichiometry" in msg
                    or "Missing elements" in msg
                ):
                    quality_warns.append(msg.replace("\n", " ")[:2000])

    except Exception as e:
        error = f"parse_error:{e}"
        return (rel_path, sg_number, sg_symbol, point_group, centrosym, error, quality_warns)


    # 2. Filter by number of sites
    if len(struct.sites) > max_sites:
        error = f"too_many_sites:{len(struct.sites)}"
        return (rel_path, sg_number, sg_symbol, point_group, centrosym, error, quality_warns)

    # 3. Build cell for spglib
    lattice = struct.lattice.matrix
    positions = [site.frac_coords for site in struct.sites]

    # Build atomic numbers robustly (ordered and disordered sites)
    numbers = []
    for site in struct.sites:
        try:
            sp = site.species  # works for ordered and disordered sites
            if hasattr(sp, "items"):
                # Disordered site: pick the dominant species by occupancy
                el = max(sp.items(), key=lambda kv: kv[1])[0]
                numbers.append(int(el.Z))
            else:
                # Ordered site
                numbers.append(int(sp.Z))
        except Exception as e:
            error = f"species_error:{e}"
            return (rel_path, sg_number, sg_symbol, point_group, centrosym, error, quality_warns)

    cell = (lattice, positions, numbers)

    # 4. Call spglib with timeout
    try:
        with time_limit(spglib_timeout):
            dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

        if dataset is None:
            error = f"spglib_failed_at_symprec={symprec}"
            return (rel_path, sg_number, sg_symbol, point_group, centrosym, error, quality_warns)

        # Use dictionary access to be robust across spglib versions
        sg_number = _spglib_get(dataset, "number")
        sg_symbol = _spglib_get(dataset, "international")
        point_group = _spglib_get(dataset, "pointgroup")

        if sg_number is None:
            error = "spglib_missing_number"

            return (rel_path, sg_number, sg_symbol, point_group, centrosym, error, quality_warns)

        sg_number = int(sg_number)


        # Centrosymmetry classification
        centrosym = "N" if sg_number in NCS_GROUPS else "C"

        return (rel_path, sg_number, sg_symbol, point_group, centrosym, None, quality_warns)

    except TimeoutException:
        error = f"spglib_timeout_{spglib_timeout}s"
        centrosym = "U"
        return (rel_path, None, None, None, centrosym, error, quality_warns)

    except Exception as e:
        error = f"spglib_error:{e}"
        centrosym = "U"
        return (rel_path, None, None, None, centrosym, error, quality_warns)

def build_symmetry_labels(
    root_cif: str,
    input_csv: str,
    output_csv: str,
    bad_csv: str,
    max_workers: int = 4,
    symprec: float = 0.01,
    fast_timeout: float = 1.0,
):
    """
    Build symmetry labels (spglib) in parallel.

    Prints FAST-pass summary and OK ratio, similar to build_cif_metadata.
    """
    df_in = pd.read_csv(input_csv)
    rel_paths = df_in["filepath"].tolist()

    rows = []
    bad_files = []
    quality_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_spglib,
                root_cif=root_cif,
                rel_path=rel_path,
                symprec=symprec,
                max_sites=MAX_SITES,
                spglib_timeout=fast_timeout,
            ): rel_path
            for rel_path in rel_paths
        }

        ok = 0
        bad = 0
        timeouts = 0

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Computing symmetry with spglib",
        )

        for future in pbar:
            rel_path, sg_number, sg_symbol, point_group, centrosym, error, quality_warns = future.result()

            if error is None:
                rows.append((rel_path, sg_number, sg_symbol, point_group, centrosym))
                ok += 1
            else:
                bad_files.append((rel_path, error))
                bad += 1

                if isinstance(error, str) and error.startswith("spglib_timeout_"):
                    timeouts += 1

            if quality_warns:
                for msg in quality_warns:
                    quality_rows.append((rel_path, msg))


            pbar.set_postfix(ok=ok, bad=bad, timeout=timeouts)


    df_ok = pd.DataFrame(
        rows,
        columns=[
            "filepath",
            "sg_number_spglib",
            "sg_symbol_spglib",
            "point_group",
            "centrosymmetry",
        ],
    )
    df_bad = pd.DataFrame(bad_files, columns=["filepath", "error"])

    df_ok.to_csv(output_csv, index=False)
    df_bad.to_csv(bad_csv, index=False)

    total = len(df_ok) + len(df_bad)
    ok_ratio = (len(df_ok) / total) if total else 0.0

    print(
        f"FAST pass summary: OK={len(df_ok)} BAD={len(df_bad)} TIMEOUTS={timeouts} "
        f"(workers={max_workers}, timeout={fast_timeout}s)"
    )
    print(f"Saved {output_csv} with {len(df_ok)} records")
    print(f"Saved {bad_csv} with {len(df_bad)} symmetry failures")
    print(f"OK ratio: {ok_ratio:.2%}")

    quality_csv = str(Path(output_csv).with_name("dataset_symmetry_bad_quality.csv"))
    df_quality = pd.DataFrame(quality_rows, columns=["filepath", "warning"], )
    df_quality.to_csv(quality_csv, index=False)
    print(f"Saved {quality_csv} with {len(df_quality)} symmetry quality warnings")

    return df_ok, df_bad


def build_master_dataset(
    datasets: List[pd.DataFrame],
    key: str = "filepath",
    add_symmetry_conflict: bool = True,
    cif_symbol_col: str = "sg_symbol_cif",
    spglib_symbol_col: str = "sg_symbol_spglib",
) -> pd.DataFrame:
    """
    Merge a list of datasets on a common key (default: 'filepath')
    to build a master table.

    Args:
        datasets:
            List of DataFrames to merge, in the desired order.
            All must contain the merge key column.
        key:
            Column name used as merge key (default: 'filepath').
        add_symmetry_conflict:
            If True and both `cif_symbol_col` and `spglib_symbol_col` are present,
            adds a boolean 'symmetry_conflict' column.
        cif_symbol_col:
            Column name for space group symbol from CIF.
        spglib_symbol_col:
            Column name for space group symbol from spglib.

    Returns:
        Merged master DataFrame.
    """


    if not datasets:
        raise ValueError("build_master_dataset: 'datasets' list is empty.")

    # Start from a copy of the first dataset
    df_final = datasets[0].copy()

    # Sequential left-merge on the chosen key
    for df in datasets[1:]:
        df_final = df_final.merge(df, on=key, how="left")

    # Optional symmetry_conflict flag
    if add_symmetry_conflict:
        if cif_symbol_col in df_final.columns and spglib_symbol_col in df_final.columns:
            df_final["symmetry_conflict"] = (
                df_final[cif_symbol_col] != df_final[spglib_symbol_col]
            )

    return df_final

def summarize_final_dataset(df_final: pd.DataFrame) -> None:
    """
    Print basic summary of the final master dataset:
    - number of rows
    - number of symmetry conflicts (if present)
    - centrosymmetry distribution
    - chemical class distribution (N/O/M printed as full names)
    """
    n_rows = len(df_final)
    print("Number of rows in final dataset:", n_rows)

    if "symmetry_conflict" in df_final.columns:
        conflict_count = int(df_final["symmetry_conflict"].sum())
        print("Number of symmetry_conflict=True:", conflict_count)

    # Centrosymmetry
    if "centrosymmetry" in df_final.columns:
        print("\nCentrosymmetry distribution:")
        print(df_final["centrosymmetry"].value_counts())

    # Chemical classes with mapping
    if "chem_class" in df_final.columns:
        print("\nChemical class distribution (chem_class):")

        mapping = {
            "N": "nonorganic",
            "O": "organic",
            "M": "metaloorganic",
            "U": "unknown",
        }

        counts = df_final["chem_class"].value_counts()

        for cls, count in counts.items():
            full_name = mapping.get(cls, "unknown")
            print(f"{cls} ({full_name}): {count}")

    print("\nHead of final dataset:")
    print(df_final.head())


def group_by_composition_and_fix_labels(
    df_metadata: pd.DataFrame,
    df_symmetry: pd.DataFrame,
    *,
    tolerance: float = 1e-5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group exact proportional compositions, then near-equivalent CIF formulas.

    Decimal coefficients are parsed as exact rationals before normalization.
    Tolerance means max_i |x_i - y_i| <= tolerance for atomic fractions,
    with identical element support and no relative tolerance. Connected
    components keep every directly matched pair in the same group, even
    when the endpoints of a chain differ by more than tolerance.
    canonical_match_type describes the whole group: exact or tolerance.
    Return grouped records and records without a parseable formula.
    """
    from collections import defaultdict
    from fractions import Fraction
    from functools import reduce
    from math import gcd, lcm
    import re
    import numpy as np
    from pymatgen.core import Element
    from scipy.spatial import cKDTree

    threshold = Fraction(str(tolerance))
    if not np.isfinite(float(threshold)) or not 0 <= threshold < 1:
        raise ValueError("tolerance must be finite and in [0, 1).")

    def exact_composition(formula):
        # Preserve decimal text; never round individual coefficients first.
        text = re.sub(r"\s+", "", str(formula))
        tokens = re.findall(r"[A-Z][a-z]?|(?:\d+(?:\.\d*)?|\.\d+)|[()\[\]{}]", text)
        if not tokens or "".join(tokens) != text:
            raise ValueError(f"Unsupported composition: {formula!r}")
        stack = [defaultdict(Fraction)]
        brackets = []
        matching = {')': '(', ']': '[', '}': '{'}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ('(', '[', '{'):
                brackets.append(token)
                stack.append(defaultdict(Fraction))
                i += 1
                continue
            if token in matching:
                if not brackets or brackets.pop() != matching[token] or not stack[-1]:
                    raise ValueError(f"Unbalanced composition: {formula!r}")
                amounts = stack.pop()
            elif token[0].isupper():
                Element(token)  # Reject unknown element symbols.
                amounts = {token: Fraction(1)}
            else:
                raise ValueError(f"Unexpected coefficient: {formula!r}")
            i += 1
            multiplier = Fraction(1)
            if i < len(tokens) and (tokens[i][0].isdigit() or tokens[i][0] == '.'):
                multiplier = Fraction(tokens[i])
                i += 1
            if multiplier <= 0:
                raise ValueError(f"Nonpositive coefficient: {formula!r}")
            for element, amount in amounts.items():
                stack[-1][element] += amount * multiplier
        if brackets or len(stack) != 1 or not stack[0]:
            raise ValueError(f"Unbalanced composition: {formula!r}")
        elements = tuple(sorted(stack[0]))
        amounts = [stack[0][el] for el in elements]
        denominator = lcm(*(v.denominator for v in amounts))
        integers = [int(v * denominator) for v in amounts]
        divisor = reduce(gcd, integers)
        key = "".join(f"{el}{n // divisor}" for el, n in zip(elements, integers))
        total = sum(amounts)
        return key, elements, tuple(v / total for v in amounts)

    df = df_metadata.merge(
        df_symmetry[["filepath", "centrosymmetry"]],
        on="filepath", how="inner", validate="one_to_one",
    )
    formulas = df["chem_formula_cif"].astype(str).str.strip()
    missing = df["chem_formula_cif"].isna() | formulas.str.lower().isin(["nan", "none", "null", ""])
    df_bad = df.loc[missing].copy()
    df = df.loc[~missing].copy()
    parsed = {f: exact_composition(f) for f in df["chem_formula_cif"].unique()}
    exact = {value[0]: value[1:] for value in parsed.values()}
    parent = {key: key for key in exact}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    by_elements = defaultdict(list)
    for key, (elements, fractions) in exact.items():
        by_elements[elements].append(key)
    if threshold:
        for keys in by_elements.values():
            keys.sort()
            points = np.array([[float(x) for x in exact[k][1]] for k in keys])
            tree = cKDTree(points)
            # Float search is only a candidate filter; rational comparison
            # below decides inclusion at the stated boundary exactly.
            radius = float(threshold) + 8 * np.finfo(float).eps
            for i, point in enumerate(points):
                for j in tree.query_ball_point(point, radius, p=np.inf):
                    if j <= i:
                        continue
                    if max(abs(x - y) for x, y in zip(exact[keys[i]][1], exact[keys[j]][1])) <= threshold:
                        left, right = find(keys[i]), find(keys[j])
                        parent[max(left, right)] = min(left, right)

    df["canonical_exact_key"] = df["chem_formula_cif"].map(lambda f: parsed[f][0])
    roots = {key: find(key) for key in exact}
    component_sizes = pd.Series(list(roots.values()), dtype=str).value_counts()
    df["canonical_group_id"] = df["canonical_exact_key"].map(roots)
    df["canonical_match_type"] = df["canonical_group_id"].map(
        lambda key: "tolerance" if component_sizes[key] > 1 else "exact"
    )
    df["canonical_tolerance"] = float(threshold)
    df["group_id_comp"] = df["canonical_group_id"]
    labels = df.groupby("group_id_comp")["centrosymmetry"].agg(lambda values: set(values) & {"C", "N"})
    label_types = labels.map(lambda values: "only_C" if values == {"C"} else "only_N" if values == {"N"} else "mixed_or_empty")
    df["label_type"] = df["group_id_comp"].map(label_types)
    df.loc[df["label_type"] == "mixed_or_empty", "centrosymmetry"] = "U"
    df["group_id_struct"] = -1
    df["group_id"] = df["group_id_comp"]
    print(f"[GROUP] records={len(df)}, missing_formula={len(df_bad)}, "
          f"exact_groups={len(exact)}, canonical_groups={len(component_sizes)}, "
          f"tolerance_groups={int((component_sizes > 1).sum())}, "
          f"conflicting_CN_groups={int(labels.map(lambda values: values == {'C', 'N'}).sum())}, "
          f"atomic_fraction_tolerance={tolerance}")
    return df, df_bad

def group_by_composition_csv(
    metadata_csv: str,
    symmetry_csv: str,
    output_csv: str,
) -> pd.DataFrame:

    df_meta = pd.read_csv(metadata_csv)
    df_sym = pd.read_csv(symmetry_csv)

    print("Loaded metadata:", metadata_csv, "→", len(df_meta), "records")
    print("Loaded symmetry:", symmetry_csv, "→", len(df_sym), "records")

    df_grouped, df_bad = group_by_composition_and_fix_labels(df_meta, df_sym)

    df_grouped.to_csv(output_csv, index=False)
    print("\nSaved grouped dataset:", output_csv, "with", len(df_grouped), "records")

    if len(df_bad) > 0:
        bad_csv = output_csv.replace(".csv", "_bad.csv")
        df_bad.to_csv(bad_csv, index=False)
        print("Saved BAD grouped rows:", bad_csv, "with", len(df_bad), "records")

    print("\nSample of grouping columns:")
    print(
        df_grouped[
            [
                "filepath",
                "chem_formula_cif",
                "group_id_comp",
                "group_id_struct",
                "group_id",
                "centrosymmetry",
                "label_type",
            ]
        ].head()
    )

    return df_grouped


def add_conflict_flags(
    df: pd.DataFrame,
    formula_col: str = "chem_formula_cif",
    centrosymmetry_col: str = "centrosymmetry",
    point_group_col: str = "point_group",
) -> pd.DataFrame:
    """
    Add canonical_formula and conflict flags to a dataset.

    Args:
        df: Input DataFrame with chemical formulas and symmetry labels.
        formula_col: Column name with raw chemical formulas.
        centrosymmetry_col: Column name with centrosymmetry labels.
        point_group_col: Column name with point group labels.

    Returns:
        DataFrame with new columns:
            - canonical_formula: flattened sum formula (e.g. CaO2H2 instead of Ca(OH)2)
            - centrosymmetry_conflict: True if same formula has >1 centrosymmetry
            - point_group_conflict: True if same formula has >1 point group
    """

    # helper: flatten formula to canonical string
    def canonical_formula(formula: str) -> str:
        from collections import Counter
        import re

        def expand_parentheses(f):
            while "(" in f:
                f = re.sub(
                    r"\(([A-Za-z0-9]+)\)(\d+)",
                    lambda m: "".join(
                        el + str(int(num or 1) * int(m.group(2)))
                        for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", m.group(1))
                    ),
                    f,
                )
            return f

        f = expand_parentheses(str(formula))
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", f)
        counts = Counter({el: (int(num) if num else 1) for el, num in tokens})
        return "".join(f"{el}{counts[el]}" for el in sorted(counts))

    # add canonical formula
    df["canonical_formula"] = df[formula_col].apply(canonical_formula)

    # conflict flags
    if centrosymmetry_col in df.columns:
        df["centrosymmetry_conflict"] = (
            df.groupby("canonical_formula")[centrosymmetry_col].transform("nunique") > 1
        )

    if point_group_col in df.columns:
        df["point_group_conflict"] = (
            df.groupby("canonical_formula")[point_group_col].transform("nunique") > 1
        )

    return df


def summarize_group_stats(df_grouped: pd.DataFrame) -> None:
    """
    Print additional statistics at the composition-group level:
    - number of unique composition groups per label_type
    - cross-tab of label_type vs chem_class (row-level)
    - (optional) composition-level sets of chem_class

    Expects df_grouped to contain:
        - 'group_id_comp'
        - 'label_type'
        - 'chem_class'
    """
    print("=== Group-level statistics ===")

    # 1) Number of unique composition groups per label_type
    comp_label = (
        df_grouped.groupby("group_id_comp")["label_type"]
        .first()
        .value_counts()
    )
    print("\nUnique composition groups per label_type:")
    for lbl, cnt in comp_label.items():
        print(f"{lbl}: {cnt}")

    # 2) Row-level cross-tab: label_type vs chem_class
    class_map = {
        "N": "nonorganic",
        "O": "organic",
        "M": "metaloorganic",
        "U": "unknown",
    }

    print("\nRow-level cross-tab: label_type x chem_class (counts)")
    ctab = pd.crosstab(df_grouped["label_type"], df_grouped["chem_class"])
    print(ctab)

    print("\nRow-level cross-tab with mapped chem_class names:")
    # Rename chem_class columns to include full names
    ctab_named = ctab.copy()
    ctab_named.columns = [
        f"{col} ({class_map.get(col, 'unknown')})" for col in ctab_named.columns
    ]
    print(ctab_named)

    # 3) Optional: composition-level chem_class sets
    comp_classes = (
        df_grouped.groupby("group_id_comp")["chem_class"]
        .apply(lambda s: set(s))
        .reset_index(name="chem_class_set")
    )

    print("\nSample of composition-level chem_class sets:")
    print(comp_classes.head())
