"""Map chemical elements to atomic numbers and periodic-table categories."""

from __future__ import annotations

# Minimal atomic-number lookup for typical CIF element symbols.

ELEMENT_Z: dict[str, int] = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
    "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68,
    "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
    "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92,
}

# Periodic-table categories used in element-level reports.

def element_category(symbol: str) -> str:
    """Return a periodic-table category label for an element symbol."""
    s = str(symbol).strip().capitalize()
    if s == "H":
        return "Hydrogen"

    alkali = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
    alkaline_earth = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
    halogens = {"F", "Cl", "Br", "I", "At", "Ts"}
    noble_gases = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}

    # "Other nonmetals" (excluding halogens and noble gases)
    other_nonmetals = {"C", "N", "O", "P", "S", "Se"}

    metalloids = {"B", "Si", "Ge", "As", "Sb", "Te", "Po"}  # Po is assigned to the metalloid category.

    # Post-transition metals / "other metals"
    post_transition = {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"}

    lanthanides = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
    actinides = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}

    # Transition metals (d-block) + group 3 choices: Sc/Y included; La/Ac handled above
    transition = {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    }

    if s in alkali:
        return "Alkali metals"
    if s in alkaline_earth:
        return "Alkaline earth metals"
    if s in lanthanides:
        return "Lanthanides"
    if s in actinides:
        return "Actinides"
    if s in transition:
        return "Transition metals"
    if s in post_transition:
        return "Post-transition metals"
    if s in metalloids:
        return "Metalloids"
    if s in halogens:
        return "Halogens"
    if s in noble_gases:
        return "Noble gases"
    if s in other_nonmetals:
        return "Other nonmetals"

    # Assign unrecognized symbols to the fallback category.
    return "Other metals"


def category_color_map() -> dict[str, str]:
    """Color map for element categories (metric vs Z and similar plots)."""
    return {
        "Hydrogen": "#B3E5FC",           # light blue
        "Alkali metals": "#F1C40F",
        "Alkaline earth metals": "#E67E22",
        "Transition metals": "#2ECC71",
        "Other metals": "#8B4513",       # brown
        "Post-transition metals": "#2C3E50",
        "Metalloids": "#5DADE2",
        "Other nonmetals": "#8B4513",   # brown
        "Halogens": "#9B59B6",
        "Noble gases": "#3498DB",
        "Lanthanides": "#E53935",        # red
        "Actinides": "#8B4513",
    }


def periodic_block_from_Z(z: int) -> str:
    if z in {1, 3, 11, 19, 37, 55, 87}:
        return "s"
    if z in {4, 12, 20, 38, 56, 88}:
        return "s"
    if 21 <= z <= 30 or 39 <= z <= 48 or 72 <= z <= 80:
        return "d"
    if 57 <= z <= 71 or 89 <= z <= 103:
        return "f"
    return "p"
