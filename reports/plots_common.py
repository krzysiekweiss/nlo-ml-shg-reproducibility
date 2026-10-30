"""Define shared figure styles, labels and output paths."""

import matplotlib.pyplot as plt

from pathlib import Path
from typing import Iterable, Optional, Union
import pandas as pd

MODEL_DISPLAY = {
    "rf": "RandomForest",
    "lgbm": "LGBM",
}

# Style maps (model-level colors)
model_color = {
    "rf": "#4477AA",   # blue
    "lgbm": "#EE6677", # red
}


def domain_color_palette(
    project: str | None = None,
    *,
    palette: str | None = None,
) -> dict[str, str]:
    """Domain -> color map with per-project gradients.

    - MP-like (default): blue gradient I (dark) / O (medium) / M (light).
    - COD-like: red gradient I (dark) / O (medium) / M (light).
    - Green: green gradient I (dark) / O (medium) / M (light).

    Use palette="mp" | "cod" | "green" to force one; else inferred from project name.
    """
    base_mp = {
        "I": "#3182bd",  # dark blue
        "O": "#9ecae1",  # medium blue
        "M": "#deebf7",  # light blue
    }
    base_cod = {
        "I": "#de2d26",  # dark red
        "O": "#fc9272",  # medium red
        "M": "#fee0d2",  # light red
    }
    # Green palette: darkest I, then M, then O (lightest)
    base_green = {
        "I": "#31a354",  # dark green
        "M": "#a1d99b",  # medium green
        "O": "#e5f5e0",  # light green
    }

    if (palette or "").lower() == "green":
        base = base_green
    elif (palette or "").lower() == "cod":
        base = base_cod
    elif (palette or "").lower() == "mp":
        base = base_mp
    else:
        proj = (project or "").upper()
        if "COD" in proj:
            base = base_cod
        elif "GREEN" in proj:
            base = base_green
        else:
            base = base_mp

    out = dict(base)
    # Treat N as I for coloring
    out["N"] = out["I"]
    return out


def domain_line_color(model: str, project: str | None, domain: str) -> str:
    """Color for a line identified by (model, domain).

    Requirements:
    - RF should stay blue-ish, LGBM red-ish.
    - Within each model, use a gradient by domain (I, M, O).
    """
    m = (model or "").lower()
    d = (domain or "").upper()

    # Blue gradient for RF-like models
    if m == "rf":
        blue_grad = {
            "I": "#3182bd",  # dark blue
            "M": "#9ecae1",  # medium
            "O": "#deebf7",  # light
            "N": "#3182bd",
        }
        return blue_grad.get(d, model_color.get("rf", "#4477AA"))

    # Red gradient for LGBM-like models
    if m == "lgbm":
        red_grad = {
            "I": "#de2d26",  # dark red
            "M": "#fc9272",  # medium
            "O": "#fee0d2",  # light
            "N": "#de2d26",
        }
        return red_grad.get(d, model_color.get("lgbm", "#EE6677"))

    # Fallback: project-based domain palette or plain model color
    palette = domain_color_palette(project)
    return palette.get(d, model_color.get(m, "#333333"))

def save_figure(
    fig,
    *,
    project_dir: str | Path,
    run_id: str,
    name: str,
    subdir: str = "figures",
    ext: str = "png",
    dpi: int = 200,
    close: bool = True,
) -> Path:
    """
    Save matplotlib figure to:
      <project_dir>/results/<run_id>/<subdir>/<name>.<ext>
    """
    if project_dir is None:
        raise ValueError("save_figure requires project_dir (got None)")
    if run_id is None:
        raise ValueError("save_figure requires run_id (got None)")


    project_dir = Path(project_dir)
    out_dir = project_dir / "results" / run_id / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{name}.{ext}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved figure as: {out_path}")

    if close:
        plt.close(fig)

    return out_path

def build_legend_label(
    gg: pd.DataFrame,
    *,
    auc: float | None = None,
    model_col: str = "model",
    domain_col: str = "domain",
    show_domain: bool = False,
    display_map: dict[str, str] | None = None,
) -> str:
    """
    Build a compact legend label for a plot group.

    Assumption: typically one run_id corresponds to one model.
    If multiple models are detected in `gg`, the label falls back to "MultipleModels".
    """
    display_map = display_map or MODEL_DISPLAY

    def _norm(x) -> str:
        return str(x).strip()

    def _is_missing(x) -> bool:
        return x is None or pd.isna(x) or _norm(x) == ""

    parts: list[str] = []

    # Model
    model_name = None
    if model_col in gg.columns:
        models = [m for m in gg[model_col].dropna().unique().tolist() if not _is_missing(m)]
        if len(models) == 1:
            key = _norm(models[0]).lower()
            model_name = display_map.get(key, _norm(models[0]))
        elif len(models) > 1:
            model_name = "MultipleModels"

    if model_name:
        parts.append(model_name)


    if show_domain and domain_col in gg.columns:
        domains = [d for d in gg[domain_col].dropna().unique().tolist() if not _is_missing(d)]
        if len(domains) == 1:
            parts.append(_norm(domains[0]))
        elif len(domains) > 1:
            parts.append("multiple")

    # AUC suffix
    if auc is not None:
        parts.append(f"AUC={auc:.3f}")

    return " | ".join(parts) if parts else ("ALL" if auc is None else f"ALL | AUC={auc:.3f}")

def make_plot_name(
    *,
    kind: str,
    run_id: str | None = None,
    mode: str | None = None,
    models: Optional[Union[str, Iterable[str]]] = None,
    steps: Optional[Union[str, Iterable[str]]] = None,
    domains: Optional[Union[str, Iterable[str]]] = None,
) -> str:
    parts = [kind]

    if run_id:
        parts.append(f"run_{run_id}")
    if mode:
        parts.append(f"mode_{mode}")

    def _norm(x):
        if x is None:
            return None
        if isinstance(x, str):
            return x
        xs = list(x)
        if len(xs) == 1:
            return xs[0]
        return "MULTI"

    m = _norm(models)
    s = _norm(steps)
    d = _norm(domains)

    if m:
        parts.append(f"model_{m}")
    if s:
        parts.append(f"step_{s}")
    if d:
        parts.append(f"domain_{d}")

    return "__".join(parts)
