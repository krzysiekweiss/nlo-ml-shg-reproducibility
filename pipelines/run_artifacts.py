"""Write run metadata, metrics, predictions and fitted-model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

import os, sys
sys.path.append(os.path.abspath(".."))


def assert_not_exists(path: str) -> None:
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")


def _write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def save_run_artifacts(
    *,
    cfg,
    run_id: str,
    mode: str,   # "cv" or "final"
    model: str,  # "rf" or "lgbm"
    step: str,   # "magpie" | "simple" | "frac"
    domain: str, # "N" | "O" | "M"
    out: dict,
) -> dict:
    """
    Save run artifacts to disk.

    This function relies on existing project utilities:
      - ml_utils.write_csv(df, path)
      - ml_utils.save_bundle(path, bundle)

    Expected output keys:
      CV:    metrics_df, oof_df, bundle, feature_cols, (optional) validation_report
      FINAL: test_metrics, test_pred_df, bundle, feature_cols, (optional) validation_report
    """
    import core.ml_utils as mlu

    results_dir = Path(cfg.RESULTS_DIR)
    models_dir = Path(cfg.MODELS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    prefix = run_id
    paths: dict = {}

    # metrics
    metrics_path = str(results_dir / f"metrics_{prefix}.csv")
    assert_not_exists(metrics_path)

    if mode == "cv":
        if "metrics_df" not in out:
            raise KeyError("Missing 'metrics_df' in CV output.")
        print(f"[WRITE] CV metrics -> {metrics_path}")
        mlu.write_csv(out["metrics_df"], metrics_path)

    elif mode == "final":
        if "test_metrics" not in out:
            raise KeyError("Missing 'test_metrics' in FINAL output.")
        print(f"[WRITE] FINAL metrics -> {metrics_path}")
        mlu.write_csv(pd.DataFrame([out["test_metrics"]]), metrics_path)

    else:
        raise ValueError("mode must be 'cv' or 'final'")

    paths["metrics"] = metrics_path

    # predictions
    if mode == "cv":
        if "oof_df" not in out:
            raise KeyError("Missing 'oof_df' in CV output.")
        oof_path = str(results_dir / f"oof_{prefix}.csv")
        assert_not_exists(oof_path)
        print(f"[WRITE] CV oof -> {oof_path}")
        mlu.write_csv(out["oof_df"], oof_path)
        paths["oof"] = oof_path

    if mode == "final":
        if "test_pred_df" not in out:
            raise KeyError("Missing 'test_pred_df' in FINAL output.")
        holdout_path = str(results_dir / f"holdout_{prefix}.csv")
        assert_not_exists(holdout_path)
        print(f"[WRITE] Holdout preds -> {holdout_path}")
        mlu.write_csv(out["test_pred_df"], holdout_path)
        paths["holdout"] = holdout_path

    # per-element report (FINAL only)
    if mode == "final" and "test_pred_df" in out:
        try:
            tbl_elem = mlu.report_by_element(
                out["test_pred_df"],
                y_true_col=out["y_col"],
                elements_col="elements_cif",
                min_n=30,
            )

            elem_path = str(results_dir / f"per_element_{prefix}.csv")
            assert_not_exists(elem_path)
            print(f"[WRITE] Per-element report -> {elem_path}")
            tbl_elem.to_csv(elem_path, index=False)

            paths["per_element"] = elem_path

            if getattr(cfg, "VERBOSE", False):
                print("\n=== Per-element report on TEST (top 25) ===")
                print(tbl_elem.head(25).to_string(index=False))

        except Exception as e:
            print(f"[WARN] Per-element report failed: {e}")


    # model bundle
    if "bundle" not in out:
        raise KeyError("Missing 'bundle' in output.")

    model_path = str(models_dir / f"model_{prefix}.pkl")
    assert_not_exists(model_path)
    print(f"[WRITE] Model bundle -> {model_path}")
    mlu.save_bundle(model_path, out["bundle"])
    paths["model"] = model_path

    # feature columns
    if "feature_cols" in out and out["feature_cols"] is not None:
        feat_path = str(results_dir / f"feature_cols_{prefix}.txt")
        assert_not_exists(feat_path)
        print(f"[WRITE] Feature cols -> {feat_path}")
        _write_text(feat_path, "\n".join(map(str, out["feature_cols"])))
        paths["feature_cols"] = feat_path

    # validation report (optional)
    if "validation_report" in out and out["validation_report"] is not None:
        rep_path = str(results_dir / f"report_{prefix}.txt")
        assert_not_exists(rep_path)
        print(f"[WRITE] Validation report -> {rep_path}")
        _write_text(rep_path, str(out["validation_report"]))
        paths["validation_report"] = rep_path

    # metadata
    meta = {
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "step": step,
        "domain": domain,
        "project": getattr(cfg, "PROJECT_NAME", None),
        "created": datetime.now().isoformat(timespec="seconds"),
        "paths": paths,
        "y_col": out.get("y_col"),
        "fold_col": out.get("fold_col"),
    }
    meta_path = str(results_dir / f"meta_{prefix}.json")
    assert_not_exists(meta_path)
    print(f"[WRITE] Meta -> {meta_path}")
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    paths["meta"] = meta_path

    return paths
