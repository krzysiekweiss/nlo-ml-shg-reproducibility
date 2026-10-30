"""Plot ranked composition-descriptor importance across chemical domains."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .plots_common import save_figure, model_color, domain_color_palette


def plot_feature_importance_for_cfg(
    cfg,
    *,
    run_id: str,
    model: str,
    domains: list[str],
    top_k: int = 5,
    normalize: bool = False,
    save: bool = False,
    title: str | None = None,
    project_dir: str | Path | None = None,   # kept for compatibility; not used if cfg.RESULTS_DIR exists
    name: str | None = None,
    verbose: bool = False,
    plot_other: bool = False,
    use_hatch: bool = True,
    baselines: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Plot feature importance for a single cfg (single dataset).
    Overlay domains on one axis (color = model, hatch = domain).

    The cfg.RESULTS_DIR attribute identifies the input directory. Importance
    tables are selected by run, model and domain. If the aggregate Other
    contribution dominates, it is displayed in a separate panel.
    Returns a tidy table used for plotting.
    """
    # resolve results directory (cfg is one dataset)
    if hasattr(cfg, "RESULTS_DIR") and getattr(cfg, "RESULTS_DIR"):
        results_dir = Path(getattr(cfg, "RESULTS_DIR"))
    else:

        proj_dir = Path(project_dir or cfg.PROJECT_DIR)
        results_dir = proj_dir / "results"

    model_key = str(model).lower()
    default_color = model_color.get(model_key, "#333333")

    # Domain colors:
    # - RF: always blue gradient by domain (I dark, O medium, M light)
    # - LGBM: red gradient
    # - other models: fall back to project-based palette
    if model_key == "rf":
        domain_color = domain_color_palette(palette="mp")
    elif model_key == "lgbm":
        domain_color = domain_color_palette(palette="cod")
    else:
        domain_color = domain_color_palette(getattr(cfg, "PROJECT_NAME", None))

    # Distinguish domains with optional hatch patterns.
    hatch_map = {"I": "", "M": "//", "O": "xx"}

    # Map displayed domain labels to the saved domain identifiers.
    DOMAIN_CANONICAL = {
        "N": "I",  # Saved identifier for the inorganic domain.
        "I": "I",
        "M": "M",
        "O": "O",
    }

    def _find_importance_csv(domain: str) -> Path | None:
        """Find per-domain importance CSV under RESULTS_DIR/run_id/tables."""
        tables_dir = results_dir / run_id / "tables"
        if not tables_dir.exists():
            print(f"[WARN] Missing importance tables dir: {tables_dir}")
            return None

        # Full tables only. A top15 file must not be selected in place of __all.
        matches = []
        for p in tables_dir.glob("feature_importance__magpie__domain_*__all.csv"):
            name = p.name
            try:
                dom_tag = name.split("domain_")[1].split("__")[0]
            except IndexError:
                continue
            if DOMAIN_CANONICAL.get(dom_tag) == domain:
                matches.append(p)

        if not matches:
            print(f"[WARN] No full importance CSV (__all) for domain={domain} in {tables_dir}")
            return None
        if len(matches) > 1:
            listed = ", ".join(str(p) for p in matches)
            raise RuntimeError(
                f"Multiple full importance files for domain={domain} in {tables_dir}: {listed}"
            )
        return matches[0]

    def _load_full(domain: str) -> pd.DataFrame | None:
        p = _find_importance_csv(domain)
        if p is None:
            return None
        df = pd.read_csv(p)

        if "feature" not in df.columns or "importance" not in df.columns:
            raise KeyError(f"CSV must contain columns ['feature','importance']: {p}")

        df = df[["feature", "importance"]].copy()
        df["feature"] = df["feature"].astype(str)
        df["importance"] = df["importance"].astype(float)
        if df["feature"].duplicated().any():
            dupes = df.loc[df["feature"].duplicated(), "feature"].unique().tolist()
            raise ValueError(f"Duplicate features in {p}: {dupes}")

        # Normalize the full vector before any top-k cut.
        if normalize:
            total = float(df["importance"].sum())
            if total > 0:
                df["importance"] = df["importance"] / total

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def _table_for_shared_features(full: pd.DataFrame, shared: list[str], domain: str) -> pd.DataFrame:
        """Importance of the shared top features, plus Other from the rest of the full table."""
        indexed = full.set_index("feature")["importance"]
        missing = [feature for feature in shared if feature not in indexed.index]
        if missing:
            raise KeyError(
                f"Importance table for domain={domain} is missing required features: {missing}"
            )
        other = float(indexed.drop(index=shared).sum())
        return pd.DataFrame({
            "feature": list(shared) + ["Other"],
            "importance": [float(indexed.loc[feature]) for feature in shared] + [other],
        })

    # Load full per-domain tables
    requested_domains = list(domains)
    full_tables = {}
    for d in requested_domains:
        loaded = _load_full(d)
        if loaded is None:
            continue
        full_tables[d] = loaded
    loaded_domains = [d for d in requested_domains if d in full_tables]
    if not loaded_domains:
        print(f"[WARN] No importance tables to plot for model={model_key} run_id={run_id}")
        return pd.DataFrame()

    # Top 15 selection and order come from the first requested domain that loaded.
    ref = loaded_domains[0]
    top_features = full_tables[ref]["feature"].head(int(top_k)).tolist()
    dom_tables = {
        d: _table_for_shared_features(full_tables[d], top_features, d)
        for d in loaded_domains
    }

    # Canonical domain ordering for plots/legend so bars match legend order.
    # Preferred order: I -> O -> M; any other domains are appended afterwards.
    preferred_order = ["I", "O", "M", "N"]
    domains = [d for d in preferred_order if d in dom_tables] + [d for d in loaded_domains if d not in preferred_order]

    features = top_features + ["Other"]

    # Optional baselines (e.g. Song, other models)
    baseline_tables: list[dict] = []
    if baselines:
        for spec in baselines:
            bpath = Path(spec["csv"])
            if not bpath.exists():
                print(f"[WARN] Baseline importance CSV not found: {bpath}")
                continue
            bdf = pd.read_csv(bpath)
            if not {"feature", "importance"}.issubset(bdf.columns):
                raise KeyError("Baseline CSV must contain columns ['feature','importance']")
            bdf = bdf[["feature", "importance"]].copy()
            bdf["feature"] = bdf["feature"].astype(str)
            bdf["importance"] = bdf["importance"].astype(float)
            if normalize:
                s_b = float(bdf["importance"].sum())
                if s_b > 0:
                    bdf["importance"] = bdf["importance"] / s_b
            baseline_tables.append(
                {
                    "label": spec.get("label") or "baseline",
                    "color": spec.get("color", "#B0B0B0"),
                    "df": bdf,
                }
            )

    # Prepare split: top features vs "Other"
    has_other = "Other" in features
    if (plot_other == False):
        has_other = False

    # Rank 1 of the reference domain stays at the top of the horizontal bars.
    features_top = list(reversed(top_features))

    def _max_without_other(df: pd.DataFrame) -> float:
        dd = df[df["feature"] != "Other"]
        return float(dd["importance"].max()) if not dd.empty else 0.0

    # y-scale for top-features panel (exclude Other so small bars are visible)
    y_max_top = max(_max_without_other(dom_tables[d]) for d in domains) if domains else 1.0
    y_max_top = y_max_top * 1.10 if y_max_top > 0 else 1.0

    # y-scale for "Other" panel (if present)
    y_max_other = 1.0
    if has_other:
        vals_other = [
            float(dom_tables[d].loc[dom_tables[d]["feature"] == "Other", "importance"].iloc[0])
            if (dom_tables[d]["feature"] == "Other").any() else 0.0
            for d in domains
        ]
        for b in baseline_tables:
            bdf = b["df"]
            if (bdf["feature"] == "Other").any():
                vals_other.append(
                    float(
                        bdf.loc[
                            bdf["feature"] == "Other", "importance"
                        ].iloc[0]
                    )
                )
        y_max_other = max(vals_other) if vals_other else 0.0
        y_max_other = y_max_other * 1.05 if y_max_other > 0 else 1.0

    # Plot: 1 axis if no "Other", otherwise 2 panels (Top + Other)
    if has_other:
        fig, (ax_top, ax_other) = plt.subplots(
            1, 2,
            figsize=(12, 4),
            gridspec_kw={"width_ratios": [4.5, 1.2]},
            sharey=False,
        )
    else:
        fig, ax_top = plt.subplots(figsize=(7, 5))
        ax_other = None

    # Bar geometry for overlay
    y = np.arange(len(features_top))
    width = 0.8 / max(1, len(domains))
    offsets = np.linspace(
        -width * (len(domains) - 1) / 2,
        width * (len(domains) - 1) / 2,
        len(domains),
    )

    # Top-features panel
    # Optional baseline bars (full-width, drawn behind domain bars)
    for b in baseline_tables:
        bdf = (
            b["df"][b["df"]["feature"] != "Other"]
            .set_index("feature")
            .reindex(features_top, fill_value=0.0)
            .reset_index()
        )
        ax_top.barh(
            y,
            bdf["importance"].to_numpy(),
            height=0.8,
            color=b["color"],
            alpha=0.35,
            label=b["label"],
            edgecolor="none",
            zorder=0,
        )

    # Domain overlays: reverse domains for bar placement so visual vertical
    # order matches legend (I top, then O, then M).
    for off, d in zip(offsets, reversed(domains)):
        df = dom_tables[d]
        df_top = df[df["feature"] != "Other"].set_index("feature").reindex(features_top).reset_index()
        if df_top["importance"].isna().any():
            missing = df_top.loc[df_top["importance"].isna(), "feature"].tolist()
            raise KeyError(f"Importance table for domain={d} is missing required features: {missing}")

        ax_top.barh(
            y + off,
            df_top["importance"].to_numpy(),
            height=width,
            color=domain_color.get(d, default_color),
            alpha=0.9,
            hatch=hatch_map.get(d, "") if use_hatch else "",
            label=d,
            edgecolor="black",
            linewidth=0.6,
        )

    def _pretty_feature_label(s: str) -> str:
        """Human-readable feature labels for plots (presentation only)."""
        s = s.replace("MagpieData", "").strip()
        s = s.replace("_", " ")

        # Statistic is a prefix in the raw name. mean and avg_dev move
        # behind the property name; other statistics keep their prefix.
        suffix_stats = {
            "mean": r"($\mu$)",
            "avg dev": r"($d$)",
        }
        prefix_stats = {
            "std dev": "σ",
            "std. dev.": "σ",
        }

        feat_map = {
            "Electronegativity": "Electronegativity",
            "CovalentRadius": "Covalent radius",
            "MeltingT": "Melting temperature",
            "SpaceGroupNumber": "Space-group number",
            "MendeleevNumber": "Mendeleev number",
            "GSvolume pa": "Volume per atom",
            "AtomicWeight": "Atomic weight",
            # Valence electrons (resolved)
            "NValence": "Total valence electrons",
            "NpValence": "p-valence electrons",
            "NsValence": "s-valence electrons",
            "NdValence": "d-valence electrons",
            "NfValence": "f-valence electrons",

            # Unfilled valence states
            "NUnfilled": "Unfilled valence states",
            "NpUnfilled": "Unfilled p states",
            "NsUnfilled": "Unfilled s states",
            "NdUnfilled": "Unfilled d states",
            "NfUnfilled": "Unfilled f states",

        }

        # normalize spacing
        s = " ".join(s.split())

        # Longer prefixes first so "std. dev." is not cut as a shorter token.
        for k, suffix in sorted(suffix_stats.items(), key=lambda kv: -len(kv[0])):
            if s.lower().startswith(k):
                rest = s[len(k):].strip()
                name = feat_map.get(rest, rest)
                return f"{name} {suffix}"
        for k, prefix in sorted(prefix_stats.items(), key=lambda kv: -len(kv[0])):
            if s.lower().startswith(k):
                rest = s[len(k):].strip()
                name = feat_map.get(rest, rest)
                return f"{prefix} {name}"

        # fallback: just feature mapping
        return feat_map.get(s, s)

    ax_top.set_yticks(y)
    ax_top.set_yticklabels([_pretty_feature_label(f) for f in features_top])
    ax_top.set_xlabel("Normalized importance" if normalize else "Feature importance")
    ax_top.set_xlim(0.0, y_max_top)
    # Other-only panel (if applicable)
    if ax_other is not None:
        yo = np.array([0])
        ax_other.set_yticks([0])
        ax_other.set_yticklabels(["Other"])
        # Optional baselines for "Other"
        for b in baseline_tables:
            bdf = b["df"]
            if (bdf["feature"] == "Other").any():
                val_b = float(
                    bdf.loc[bdf["feature"] == "Other", "importance"].iloc[0]
                )
                ax_other.barh(
                    yo,
                    [val_b],
                    height=0.8,
                    color=b["color"],
                    alpha=0.35,
                    label=b["label"],
                    edgecolor="none",
                    zorder=0,
                )

        for off, d in zip(offsets, reversed(domains)):
            df = dom_tables[d]
            val = float(df.loc[df["feature"] == "Other", "importance"].iloc[0]) if (df["feature"] == "Other").any() else 0.0

            ax_other.barh(
                yo + off,
                [val],
                height=width,
                color=domain_color.get(d, default_color),
                alpha=0.9,
                hatch=hatch_map.get(d, "") if use_hatch else "",
                edgecolor="black",
                linewidth=0.6,
            )

        ax_other.set_xticks([0])
        ax_other.set_xticklabels(["Other"])
        ax_other.set_xlim(0.0, y_max_other)

    # Title / legend
    if title is None:
        proj_name = getattr(cfg, "PROJECT_NAME", getattr(cfg, "project_name", ""))
        title = f"{proj_name} | {model_key} | run={run_id}"

    if has_other:
        ax_top.set_title(f"{title} | Top features")
        ax_other.set_title("Other")
    else:
        ax_top.set_title(title)

    if len(domains) > 1:
        # Build legend explicitly so order matches canonical domains (I, O, M).
        from matplotlib.patches import Patch

        handles = []
        labels = []
        for d in domains:
            handles.append(
                Patch(
                    facecolor=domain_color.get(d, default_color),
                    hatch=hatch_map.get(d, "") if use_hatch else "",
                    edgecolor="black",
                    linewidth=0.6,
                )
            )
            labels.append(d)

        ax_top.legend(
            handles,
            labels,
            title="Domain",
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(1.02, 0.0),
        )


    fig.tight_layout()

    # Build tidy output table
    out_rows = []
    for d in domains:
        t = dom_tables[d].copy()
        t.insert(0, "domain", d)
        t.insert(0, "run_id", run_id)
        t.insert(0, "model", model_key)
        out_rows.append(t)
    out_tbl = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()

    if save:
        out_name = name or f"feature_importance__{model_key}__run_{run_id}"
        if title == "" and not out_name.endswith("_notitle"):
            out_name = f"{out_name}_notitle"
        save_figure(fig, project_dir=cfg.PROJECT_DIR, run_id="paper", name=out_name, close=False)

    if verbose:
        print(out_tbl.head(30).to_string(index=False))

    return out_tbl
