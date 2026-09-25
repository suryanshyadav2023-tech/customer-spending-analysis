"""Step 9: load the saved models from models/ (no retraining) and check that they reproduce the test metrics.

Checks, per feature set: segmentation (all users), single-stage RF classifier, single-stage RF regressor and the
hurdle model (stage 1 x stage 2). Compared (1) at full precision with outputs/tables/*.csv and the stored test
predictions, and (2) as printed in RESULTS.md. Writes outputs/tables/model_verification.csv; exits 1 on a mismatch.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from common import (CLUSTERS, FEATURES, FS_LABEL, MODELS, PREDICTIONS, ROOT, SEGMENTATION_MODEL, TABLES,
                    add_segment_dummies)
from modeling import clf_metrics, prf_at, prior_correct, reg_metrics

TOL = 1e-9  # full-precision tolerance (RF predict_proba sums trees across threads, so last bits can differ)


def main():
    man = json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))
    seg_model = joblib.load(SEGMENTATION_MODEL)
    feats = pd.read_parquet(FEATURES)
    stored_seg = pd.read_parquet(CLUSTERS)["segment"]
    pr = pd.read_parquet(PREDICTIONS)
    results_md = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    rows = []

    def add(check, file, metric, expected, got, printed_ok=None):
        diff = abs(float(expected) - float(got))
        rows.append({"check": check, "file": file, "metric": metric, "expected": expected, "reproduced": got,
                     "abs_diff": diff, "full_precision_match": diff <= TOL,
                     "matches_RESULTS_md": "" if printed_ok is None else bool(printed_ok)})

    # ---- segmentation ----
    X = np.log1p(feats[seg_model["input_columns"]].to_numpy(dtype=float))
    cid = seg_model["kmeans"].predict(seg_model["scaler"].transform(X))
    seg = pd.Series(cid, index=feats.index).map(seg_model["cluster_to_segment"])
    add("segmentation: users assigned to the stored segment", "segmentation.joblib", "mismatched users",
        0, int((seg != stored_seg.astype(str)).sum()))

    df = add_segment_dummies(feats, stored_seg)
    te = df.loc[pr.index]
    yc, yr = te["will_purchase"].to_numpy(), te["future_spend"].to_numpy()
    buy = yr > 0
    cm = pd.read_csv(TABLES / "classification_metrics.csv").set_index(["feature_set", "model"])
    rm = pd.read_csv(TABLES / "regression_metrics.csv").set_index(["feature_set", "model"])
    hm = pd.read_csv(TABLES / "hurdle_metrics.csv")
    tune = pd.read_csv(TABLES / "threshold_tuning.csv").set_index(["feature_set", "model"])
    n1, n0 = man["prior_correction"]["n1_train_positives"], man["prior_correction"]["n0_train_negatives"]

    for fs, spec in man["feature_sets"].items():
        cols = spec["columns"]
        # single-stage RF classifier
        f = spec["single_stage_rf_classifier"]["file"]
        p = joblib.load(MODELS / f).predict_proba(te[cols])[:, 1]
        add(f"{fs}: RF classifier test probabilities", f, "max |diff| vs stored", 0,
            np.max(np.abs(p - pr[f"clf|{fs}|RandomForest"].to_numpy())))
        met = clf_metrics(yc, p)
        printed = f"| {FS_LABEL[fs]} | Random Forest | {met['roc_auc']:.4f} | {met['pr_auc']:.4f} | {met['f1']:.4f} |"
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            add(f"{fs}: RF classifier", f, k, cm.loc[(fs, "RandomForest"), k], met[k], printed in results_md)
        ft = prf_at(yc, p, tune.loc[(fs, "RandomForest"), "tuned_threshold_from_validation"])[2]
        add(f"{fs}: RF classifier", f, "f1_at_tuned_threshold", tune.loc[(fs, "RandomForest"), "test_f1_at_tuned"], ft,
            f"| {ft:.4f} |" in results_md)
        # single-stage RF regressor
        f = spec["single_stage_rf_regressor"]["file"]
        q = joblib.load(MODELS / f).predict(te[cols])
        add(f"{fs}: RF regressor test predictions", f, "max |diff| vs stored", 0,
            np.max(np.abs(q - pr[f"reg|{fs}|RandomForest"].to_numpy())))
        met = reg_metrics(yr, q)
        for k in ["MAE", "RMSE", "R2"]:
            add(f"{fs}: RF regressor (single-stage)", f, k, rm.loc[(fs, "RandomForest"), k], met[k],
                f"| {FS_LABEL[fs]} | RandomForest (single-stage) | {met['MAE']:,.3f} [" in results_md)
        # hurdle
        s1, s2 = spec["hurdle_stage1_classifier"]["file"], spec["hurdle_stage2_regressor"]["file"]
        p_raw = joblib.load(MODELS / s1).predict_proba(te[cols])[:, 1]
        h = prior_correct(p_raw, n1, n0) * joblib.load(MODELS / s2).predict(te[cols])
        add(f"{fs}: hurdle test predictions", f"{s1} + {s2}", "max |diff| vs stored", 0,
            np.max(np.abs(h - pr[f"hurdle|{fs}"].to_numpy())))
        exp = hm[(hm["feature_set"] == fs) & hm["model"].str.contains("prior-corrected")].set_index("subset")
        for subset, mask in [("all_test_users", np.ones_like(buy)), ("test_buyers_only", buy)]:
            met = reg_metrics(yr[mask], h[mask])
            printed = None if subset == "test_buyers_only" else \
                f"| {FS_LABEL[fs]} | Hurdle (prior-corrected) | {met['MAE']:,.3f} [" in results_md
            for k in ["MAE", "RMSE", "R2"]:
                add(f"{fs}: hurdle ({subset})", f"{s1} + {s2}", k, exp.loc[subset, k], met[k], printed)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "model_verification.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["check", "metric", "expected", "reproduced", "abs_diff", "full_precision_match",
               "matches_RESULTS_md"]].to_string())
    bad = out[~out["full_precision_match"] | (out["matches_RESULTS_md"] == False)]  # noqa: E712
    print(f"\n{len(out)} checks; max abs diff {out['abs_diff'].max():.3e}; failures: {len(bad)}")
    if len(bad):
        sys.exit(1)


if __name__ == "__main__":
    main()
