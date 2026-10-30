"""Generate composition descriptors and padded elemental feature arrays."""

import warnings
from matminer.featurizers.composition import ElementProperty
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from tqdm import tqdm


def safe_composition(x):
    """Parse a chemical formula, returning None when parsing fails."""
    try:
        return Composition(str(x))
    except Exception:
        return None


def magpie_feature_generation(
    input_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Generate Magpie features for a dataset with chemical formulas and
    centrosymmetry labels.

    The function performs the following steps:
        1) Load the input CSV.
        2) Check key columns and drop rows with invalid or empty formulas.
        3) Optionally remove rows with centrosymmetry == 'U' (keep only 'C' and 'N').
        4) Parse chem_formula_cif into pymatgen Composition objects.
        5) Run the Magpie featurizer (ElementProperty.from_preset("magpie")).
        6) Drop rows that contain NaN in any Magpie feature.
        7) Save the cleaned feature table to output_csv.

    Args:
        input_csv: Path to the input CSV file. Must contain columns:
                   ["filepath", "chem_formula_cif", "centrosymmetry"].
        output_csv: Path to the output CSV where Magpie features will be saved.

    Returns:
        A pandas DataFrame with Magpie features and original columns for the
        rows that passed all filters.
    """

    # 1. Load the input dataset
    df = pd.read_csv(input_csv)
    print("Loaded:", input_csv, "rows:", len(df))

    # 2. Check key columns
    required_cols = ["filepath", "chem_formula_cif", "centrosymmetry"]
    missing = df[required_cols].isna().sum()

    print("\nMissing values in key columns:")
    print(missing)

    # Drop rows without a valid chemical formula
    df = df.dropna(subset=["chem_formula_cif"])
    df = df[df["chem_formula_cif"].astype(str).str.len() > 0]

    print("\nAfter dropping invalid formulas:", len(df))

    # Optionally drop rows with centrosymmetry == "U"
    # (Ambiguous compositions are usually not used for supervised ML)
    df = df[df["centrosymmetry"].isin(["C", "N"])].copy()
    print("After removing centrosymmetry = 'U':", len(df))

    # 3. Convert chemical formulas to Composition objects
    from tqdm import tqdm
    tqdm.pandas()

    df["composition"] = df["chem_formula_cif"].progress_apply(safe_composition)

    # Drop records where Composition parsing failed
    n_bad_comp = df["composition"].isna().sum()
    print(f"\nBad Composition entries: {n_bad_comp}")

    df = df.dropna(subset=["composition"])
    print("Remaining after Composition cleanup:", len(df))

    # 4. Define Magpie featurizer
    warnings.filterwarnings(
        "ignore",
        message=r".*MagpieData\(impute_nan=False\).*",
        category=UserWarning,
    )

    magpie_featurizer = ElementProperty.from_preset("magpie")

    # 5. Generate Magpie features

    print("\n===Running Magpie featurizer...")
    df_magpie = magpie_featurizer.featurize_dataframe(df, "composition", ignore_errors=True, pbar=True)

    print("Magpie columns:", len(df_magpie.columns))

    # 6. Check NaNs in resulting feature columns

    feature_cols = [c for c in df_magpie.columns if c.startswith("Magpie")]
    nan_counts = df_magpie[feature_cols].isna().sum()

    print("\nNaN count per Magpie column (non-zero only):")
    print(nan_counts[nan_counts > 0])

    # Drop rows that contain NaN in any Magpie feature
    df_magpie_clean = df_magpie.dropna(subset=feature_cols).copy()

    print("\nAfter removing rows with NaN in features:", len(df_magpie_clean))

    # 7. Save the final feature table

    df_magpie_clean.to_csv(output_csv, index=False)

    # Final report
    print("\n📊 Final report:")
    print(" - Input rows:", len(pd.read_csv(input_csv)))
    print(" - After dropping invalid formulas:", len(df))
    print(" - After Composition cleanup:", len(df))
    print(" - After NaN removal:", len(df_magpie_clean))
    print(" - Saved rows:", len(df_magpie_clean))
    print("\nSaved:", output_csv, "with:", df_magpie_clean.shape)

    return df_magpie_clean


def _safe_atomic_radius(e) -> float:
    """
    Return a stable atomic radius value without emitting pymatgen warnings.
    Preference order:
      1) atomic_radius_calculated
      2) atomic_radius
      3) 0.0
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*No data available for atomic_radius_calculated.*",
            category=UserWarning,
        )
        r_calc = getattr(e, "atomic_radius_calculated", None)

    r_emp = getattr(e, "atomic_radius", None)

    # Normalize None -> 0.0 and ensure float
    if r_calc:
        return float(r_calc)
    if r_emp:
        return float(r_emp)
    return 0.0


def simple_feature_generation_padded(input_csv: str, output_csv: str, max_atoms: int = 20) -> pd.DataFrame:
    """
    Generate simplified Magpie-like features with fixed-size tensors.
    Each structure -> tensor [max_atoms, num_features], padded with zeros if fewer atoms.
    Features per atom: Z, mass, covalent radius, electronegativity, num_atoms.
    """

    # 1. Load dataset
    df = pd.read_csv(input_csv)
    print("Loaded:", input_csv, "rows:", len(df))

    # 2. Drop invalid formulas
    df = df.dropna(subset=["chem_formula_cif"])
    df = df[df["chem_formula_cif"].astype(str).str.len() > 0]
    df = df[df["centrosymmetry"].isin(["C", "N"])].copy()
    print("Remaining after cleanup:", len(df))

    # 3. Parse formulas
    tqdm.pandas()
    df["composition"] = df["chem_formula_cif"].progress_apply(safe_composition)
    df = df.dropna(subset=["composition"])
    print("Remaining after Composition parsing:", len(df))
    print("\n===Running Simple featurizer...")
    # 4. Feature extraction
    def extract_tensor(comp: Composition, max_atoms: int) -> np.ndarray:
        """Return a flattened (max_atoms * num_features,) tensor for a single composition."""
        num_features = 5  # Z, mass, radius, electronegativity, num_atoms

        rows: list[list[float]] = []
        for el, amt in comp.items():
            e = Element(el)
            n = int(amt)
            if n <= 0:
                continue
            for _ in range(n):
                rows.append([
                    float(e.Z),
                    float(e.atomic_mass),
                    _safe_atomic_radius(e),
                    float(e.X or 0.0),
                    float(comp.num_atoms),
                ])

        atoms = np.array(rows, dtype=float)

        # Enforce a two-dimensional array with shape (n_atoms, num_features).
        if atoms.size == 0:
            atoms = np.zeros((0, num_features), dtype=float)
        elif atoms.ndim == 1:
            if atoms.shape[0] == num_features:
                atoms = atoms.reshape(1, num_features)
            else:
                atoms = np.zeros((0, num_features), dtype=float)

        # Pad or truncate
        if atoms.shape[0] < max_atoms:
            pad = np.zeros((max_atoms - atoms.shape[0], num_features), dtype=float)
            atoms = np.vstack([atoms, pad])
        else:
            atoms = atoms[:max_atoms, :]

        return atoms.flatten()  # shape: [max_atoms * num_features]

    feature_rows = []
    for comp in tqdm(df["composition"], desc="Featurizing"):
        feature_rows.append(extract_tensor(comp, max_atoms))

    colnames = [f"atom{i}_{feat}" for i in range(max_atoms)
                for feat in ["Z","Mass","Radius","Electronegativity","NumAtoms"]]

    df_feats = pd.DataFrame(feature_rows, index=df.index, columns=colnames)

    # 5. Merge with original
    df_out = pd.concat([df, df_feats], axis=1)

    # 6. Check NaNs in resulting feature columns
    feature_cols = [c for c in df_out.columns if c.startswith("atom")]

    nan_counts = df_out[feature_cols].isna().sum()
    print("\nNaN count per simple feature column (non-zero only):")
    print(nan_counts[nan_counts > 0])

    # Drop rows that contain NaN in any simple feature
    df_clean = df_out.dropna(subset=feature_cols).copy()
    print("\nAfter removing rows with NaN in features:", len(df_clean))

    # 7. Save the final feature table
    df_clean.to_csv(output_csv, index=False)
    print("Saved:", output_csv, "rows:", len(df_clean))

    return df_clean

def frac_count_feature_generation_padded(input_csv: str, output_csv: str, aggregate_feat: bool, max_atoms: int = 20) -> pd.DataFrame:
    """
    Generate full-shell FracCount features with fixed-size tensors.
    Each structure -> tensor [max_atoms, 44], padded with zeros if fewer atoms.
    Features per atom: for each subshell (1s..7f): electron count + fractional occupancy.
    """

    import pandas as pd
    import numpy as np
    from pymatgen.core import Composition, Element
    from tqdm import tqdm


    # 1. Load dataset

    df = pd.read_csv(input_csv)
    print("Loaded:", input_csv, "rows:", len(df))


    # 2. Drop invalid formulas

    df = df.dropna(subset=["chem_formula_cif"])
    df = df[df["chem_formula_cif"].astype(str).str.len() > 0]
    df = df[df["centrosymmetry"].isin(["C", "N"])].copy()
    print("Remaining after cleanup:", len(df))


    # 3. Parse formulas

    tqdm.pandas()
    df["composition"] = df["chem_formula_cif"].progress_apply(safe_composition)
    df = df.dropna(subset=["composition"])
    print("Remaining after Composition parsing:", len(df))
    print("\n===Running FracCount featurizer...")

    # 4. Full-shell layout

    subshells = [
        "1s",
        "2s","2p",
        "3s","3p","3d",
        "4s","4p","4d","4f",
        "5s","5p","5d","5f",
        "6s","6p","6d","6f",
        "7s","7p","7d","7f"
    ]
    capacities = {"s": 2, "p": 6, "d": 10, "f": 14}


    # 5. Parse electron counts and fractional subshell occupancies.

    def frac_count_vector(element: Element):

        # safe access to electronic structure
        try:
            es = element.electronic_structure.replace("[","").replace("]","")
        except Exception:
            # return zero vector of correct length
            return [0.0] * (len(subshells) * 2)

        parts = es.split(".")

        valence = {}
        for term in parts:
            term = term.strip()
            if not term or term[0].isalpha():
                continue
            i = 0
            while i < len(term) and term[i].isdigit():
                i += 1
            n = term[:i]
            l = term[i]
            ecount = int(term[i+1:])
            valence[f"{n}{l}"] = ecount

        # find first valence subshell (e.g. "4d")
        if valence:
            first_valence = min(valence.keys(), key=lambda x: (int(x[:-1]), "spdf".index(x[-1])))
        else:
            first_valence = None

        vec = []
        for subshell in subshells:
            n = int(subshell[:-1])
            l = subshell[-1]

            if first_valence is not None:
                fv_n = int(first_valence[:-1])
                fv_l = first_valence[-1]

                # if subshell is before first valence → full
                if (n < fv_n) or (n == fv_n and "spdf".index(l) < "spdf".index(fv_l)):
                    count = capacities[l]
                    frac = 1.0
                else:
                    count = valence.get(subshell, 0)
                    frac = count / capacities[l]
            else:
                count = 0
                frac = 0.0

            vec.extend([count, frac])

        return vec


    # 6. Extract tensor for composition

    def extract_tensor(comp: Composition, max_atoms: int, bad: list):
        """
        Convert a Composition into a padded atom feature tensor.
        Fully dynamic: no hardcoded feature counts.
        Handles invalid elements by skipping and logging them into `bad`.
        Returns a flattened vector of shape [max_atoms * num_features].
        """

        atoms = []

        for el, amt in comp.items():
            try:
                e = Element(el)
            except Exception:
                # log invalid element
                bad.append(el)
                print(f"[WARN] Skipping invalid element: {el}")
                continue

            # simple features
            simple = [
                e.Z,
                e.atomic_mass,
                _safe_atomic_radius(e),
                e.X or 0,
                comp.num_atoms,
            ]

            # frac-count features (dynamic)
            fc = frac_count_vector(e)

            vec = simple + fc

            # repeat according to stoichiometry
            for _ in range(int(amt)):
                atoms.append(vec)

        atoms = np.array(atoms, dtype=float)

        # Handle empty compositions
        if atoms.size == 0:
            num_features = len(simple) + len(fc)
            return np.zeros(max_atoms * num_features, dtype=float)

        # Ensure atoms is 2D
        if atoms.ndim == 1:
            atoms = atoms.reshape(1, -1)

        num_features = atoms.shape[1]

        # Pad or truncate
        if atoms.shape[0] < max_atoms:
            pad = np.zeros((max_atoms - atoms.shape[0], num_features))
            atoms = np.vstack([atoms, pad])
        else:
            atoms = atoms[:max_atoms]

        return atoms.flatten()


    def aggregate_features(atom_matrix):
        """
        Compute aggregated features (mean, std, min, max) for simple and frac-count features,
        plus global composition-level features.
        atom_matrix shape: [max_atoms, num_features]
        """

        feats = {}

        # indices of simple features (fixed order in extract_tensor)
        Z_index = 0
        Mass_index = 1
        NumAtoms_index = 4

        # Global composition-level features
        feats["Z_sum"] = atom_matrix[:, Z_index].sum()
        feats["Mass_sum"] = atom_matrix[:, Mass_index].sum()
        feats["NumAtoms_total"] = atom_matrix[:, NumAtoms_index].sum()

        # Aggregates for simple features
        simple_names = ["Z", "Mass", "Radius", "EN", "NumAtoms"]
        for i, name in enumerate(simple_names):
            col = atom_matrix[:, i]
            feats[f"{name}_mean"] = np.mean(col)
            feats[f"{name}_std"] = np.std(col)
            feats[f"{name}_min"] = np.min(col)
            feats[f"{name}_max"] = np.max(col)

        # Aggregates for frac-count features
        num_features = atom_matrix.shape[1]
        for j in range(5, num_features):
            col = atom_matrix[:, j]
            feats[f"FC_{j}_mean"] = np.mean(col)
            feats[f"FC_{j}_std"] = np.std(col)
            feats[f"FC_{j}_min"] = np.min(col)
            feats[f"FC_{j}_max"] = np.max(col)

        return feats


    # 7. Featurize all compositions

    bad = []  # collect invalid elements or compositions

    feature_rows = []

    for comp in tqdm(df["composition"], desc="FracCount featurizing"):
        atom_matrix = extract_tensor(comp, max_atoms, bad)  # flattened tensor

        if aggregate_feat:
            # reshape back to [max_atoms, num_features]
            atom_matrix = atom_matrix.reshape(max_atoms, -1)
            agg = aggregate_features(atom_matrix)
            feature_rows.append(agg)
        else:
            feature_rows.append(atom_matrix)


    # 7b. Build DataFrame

    if aggregate_feat:
        # aggregated features → dicts → pandas handles keys automatically
        df_feats = pd.DataFrame(feature_rows, index=df.index)
        df_feats = df_feats.add_prefix("atom_")

    else:
        # infer number of features per atom automatically
        num_features = atom_matrix.shape[1]

        # Validate the feature-vector dimensions.
        expected_len = max_atoms * num_features
        for row in feature_rows:
            if len(row) != expected_len:
                raise ValueError(f"Row length {len(row)} != expected {expected_len}")

        # build column names dynamically
        colnames = []
        simple_names = ["Z", "Mass", "Radius", "EN", "NumAtoms"]

        for i in range(max_atoms):
            # simple features
            for name in simple_names:
                colnames.append(f"atom{i}_{name}")

            # frac-count features
            for sub in subshells:
                colnames.append(f"atom{i}_FracCount_{sub}_e")
                colnames.append(f"atom{i}_FracCount_{sub}_frac")

        df_feats = pd.DataFrame(feature_rows, index=df.index, columns=colnames)


    # 8. Merge with original

    df_out = pd.concat([df, df_feats], axis=1)


    # 9. Drop rows with NaNs

    feature_cols = [c for c in df_out.columns if c.startswith("atom")]
    nan_counts = df_out[feature_cols].isna().sum()
    print("\nNaN count per FracCount feature column (non-zero only):")
    print(nan_counts[nan_counts > 0])

    df_clean = df_out.dropna(subset=feature_cols).copy()
    print("\nAfter removing rows with NaN in features:", len(df_clean))


    # 10. Save main dataset

    df_clean.to_csv(output_csv, index=False)
    print("Saved:", output_csv, "rows:", len(df_clean))


    # 11. Save BAD structures

    if len(bad) > 0:
        bad_csv = output_csv.replace(".csv", "_bad.csv")
        pd.DataFrame({"bad": bad}).to_csv(bad_csv, index=False)
        print("Saved BAD list:", bad_csv, "count:", len(bad))
    else:
        print("No BAD entries detected.")

    return df_clean
