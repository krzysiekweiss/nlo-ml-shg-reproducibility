"""Assign chemical domains from elemental composition using fixed heuristic rules."""

import re
import pandas as pd
from collections import Counter
from tqdm import tqdm

# Element groups
ALKALI = {"Li","Na","K","Rb","Cs","Fr"}
ALKALINE_EARTH = {"Be","Mg","Ca","Sr","Ba","Ra"}
D_BLOCK = {"Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
           "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
           "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg"}
P_BLOCK_METALS = {"Al","Ga","In","Tl","Sn","Pb","Bi"}
LANTHANIDES = {"La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}
ACTINIDES   = {"Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"}
METALS = ALKALI | ALKALINE_EARTH | D_BLOCK | P_BLOCK_METALS | LANTHANIDES | ACTINIDES
METALLOIDS = {"B","Si","Ge","As","Sb","Te"}
ORGANIC_ALLOWED = {"H","C","N","O","S","P","F","Cl","Br","I"}
HALOGENS = {"F","Cl","Br","I"}

# Formula parser
def parse_counts(formula: str) -> Counter:
    """
    Parse a chemical sum formula into element counts.
    Handles parentheses like AlP2H17C5(NO4)2 by expanding them.
    Returns a Counter {element: count}.
    """
    def expand_parentheses(f):
        while "(" in f:
            f = re.sub(r"\(([A-Za-z0-9]+)\)(\d+)",
                       lambda m: "".join(
                           el + str(int(num or 1)*int(m.group(2)))
                           for el,num in re.findall(r"([A-Z][a-z]?)(\d*)", m.group(1))
                       ), f)
        return f

    formula = expand_parentheses(formula)
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    return Counter({el: (int(num) if num else 1) for el, num in tokens})

# Classifier function
def chem_classify_formula(formula: str,
                         strict_carbonyl: bool = True,
                         strict_carbides: bool = True) -> tuple:
    """
    Classify a compound as Organic (O), Inorganic (N), or Metaloorganic (M)
    based on heuristic rules applied to its sum formula.

    Parameters:
    - formula: str, chemical sum formula (e.g. 'FeC5O5', 'AlP2H17C5(NO4)2')
    - strict_carbonyl: bool, if True classify metal carbonyls as inorganic
    - strict_carbides: bool, if True classify metal carbides as inorganic

    Returns:
    - tuple: (classification_code, subclass_reason)
      classification_code ∈ {"O","N","M"}
      subclass_reason: str explanation
    """
    counts = parse_counts(formula)
    elems = set(counts.keys())

    # Rule 1: No carbon → inorganic
    if "C" not in elems:
        return "N", "no carbon"

    # Rule 2: Pure carbon → inorganic
    if elems == {"C"}:
        return "N", "pure carbon"

    # Rule 3: Carbon oxides
    if elems.issubset({"C","O","H"}):
        o = counts.get("O",0)
        h = counts.get("H",0)
        if o >= 1 and h <= 2*counts["C"]:
            return "N", "carbon oxide"

    # Rule 4: Carbides
    if strict_carbides and (elems & METALS) and elems.issubset(METALS | {"C","H","O"}):
        return "N", "carbide"

    # Rule 5: Carbonates / bicarbonates
    if (elems & (ALKALI | ALKALINE_EARTH)) and "O" in elems:
        o = counts["O"]
        if o >= 3*counts["C"]:
            return "N", "carbonate/bicarbonate"

    # Rule 6: Cyanides / isocyanates / thiocyanates
    if (elems & (METALS | METALLOIDS)):
        if elems.issubset({"C","N"} | METALS | METALLOIDS):
            return "N", "cyanide"
        if elems.issubset({"C","N","O"} | METALS | METALLOIDS):
            return "N", "isocyanate"
        if elems.issubset({"C","N","S"} | METALS | METALLOIDS):
            return "N", "thiocyanate"

    # Rule 7: Organic-only elements
    if elems.issubset(ORGANIC_ALLOWED):
        return "O", "only organic elements"

    # Rule 8: Metal or metalloid with organic heteroatoms implies metal-organic.
    if (elems & (METALS | METALLOIDS)):
        return "M", "metal/metalloid present"

    # Rule 9: Fallback → organic
    return "O", "fallback"

# CSV reader with progress bar and summary
def chem_classify_dataframe(df, column="chem_formula_cif",
                            strict_carbonyl=True, strict_carbides=True,
                            report=True, save_path=None):
    """
    Classify formulas in a DataFrame.
    Shows progress bar and prints summary report at the end.

    Parameters:
    - df: pandas.DataFrame
    - column: str, column name containing formulas
    - strict_carbonyl: bool, if True classify metal carbonyls as inorganic
    - strict_carbides: bool, if True classify metal carbides as inorganic
    - report: bool, if True print summary reports
    - save_path: str or None, if provided save results to CSV

    Returns:
    - DataFrame with original formulas and two new columns:
      'chem_class' (O/N/M) and 'chem_subclass' (reason string)
    """


    print("▶ Starting classification...")

    classifications, subclasses = [], []
    for formula in tqdm(df[column], desc="Classifying formulas"):
        cls, reason = chem_classify_formula(str(formula),strict_carbonyl, strict_carbides)
        classifications.append(cls)
        subclasses.append(reason)

    df["chem_class"] = classifications
    df["chem_subclass"] = subclasses

    # Summary report
    if report:
        summary = df["chem_class"].value_counts()
        print("\n📊 Classification report:")
        print("\n - total: ", len(df))
        for cls, count in summary.items():
            label = {"O":"Organic","N":"Inorganic","M":"Metaloorganic"}[cls]
            print(f" - {label} ({cls}): {count}")
        # Subclass report
        print("\n📊 Subclass breakdown:")
        subclass_summary = df["chem_subclass"].value_counts()
        for reason, count in subclass_summary.items():
            print(f" - {reason}: {count}")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n💾 Saved classified DataFrame to {save_path}")

    return df

# CSV wrapper
def chem_classify_csv(path, **kwargs):
    """
    Wrapper: read CSV and classify formulas.
    Returns DataFrame with classification results.
    """
    df = pd.read_csv(path)
    return chem_classify_dataframe(df, **kwargs)
