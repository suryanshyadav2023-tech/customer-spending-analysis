"""Step 4: regression + classification + ablation (a/b/c) + threshold tuning + hurdle model + leakage audit.

All models are fit on the train split and scored on the same held-out test users.
Scalers (for the linear models) live inside Pipelines, so they are fit on train only.
Model choice (hurdle stage 1) and F1 thresholds use an inner validation split of TRAIN, never the test set.
Outputs (outputs/tables): classification_metrics, regression_metrics, classifier_validation, split_sizes,
threshold_tuning, ablation_summary, ablation_bootstrap, classification_ci, hurdle_metrics, hurdle_calibration,
hurdle_bootstrap, regression_ci, regression_error_concentration, leakage_audit.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score, \
    root_mean_squared_error

from common import (BEHAVIORAL, CLUSTERS, FEATURES, FEATURES_LEAKY, MODELS, PREDICTIONS, RFM, SEED, TABLES,
                    add_segment_dummies, feature_sets, fit_segments)
from modeling import (CLF, N_BOOT, REG, STAGE2, fit_all, inner_split, prf_at, prior_correct, reg_metrics,
                      validate_classifiers)


def ci_row(arr: np.ndarray) -> dict:
    return {"ci95_low": np.percentile(arr, 2.5), "ci95_high": np.percentile(arr, 97.5)}


def main():
    df = pd.read_parquet(FEATURES)
    seg = pd.read_parquet(CLUSTERS)["segment"]
    df = add_segment_dummies(df, seg)
    fsets = feature_sets(df)
    print(json.dumps(fsets, indent=1))
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    print(f"train={len(tr):,} test={len(te):,} test positives={int(te['will_purchase'].sum()):,}")

    # ---------------- honest models + ablation ----------------
    honest, preds, fitted = fit_all(df, fsets, "honest")
    honest[honest["task"] == "classification"].drop(columns=["MAE", "RMSE", "R2"], errors="ignore") \
        .dropna(axis=1, how="all").to_csv(TABLES / "classification_metrics.csv", index=False)
    honest[honest["task"] == "regression"].dropna(axis=1, how="all") \
        .to_csv(TABLES / "regression_metrics.csv", index=False)
    for fs in fsets:
        joblib.dump(fitted[f"clf|{fs}|RandomForest"], MODELS / f"rf_clf_{fs}.joblib")
        joblib.dump(fitted[f"reg|{fs}|RandomForest"], MODELS / f"rf_reg_{fs}.joblib")

    val = validate_classifiers(df, fsets)
    val.to_csv(TABLES / "classifier_validation.csv", index=False)
    _, itr, iva = inner_split(df)
    pd.DataFrame([
        {"part": "train (all)", "users": len(tr), "buyers_oct25_31": int(tr["will_purchase"].sum())},
        {"part": "inner train (model choice / thresholds)", "users": len(itr),
         "buyers_oct25_31": int(tr.loc[itr, "will_purchase"].sum())},
        {"part": "validation (part of train)", "users": len(iva), "buyers_oct25_31": int(tr.loc[iva, "will_purchase"].sum())},
        {"part": "test", "users": len(te), "buyers_oct25_31": int(te["will_purchase"].sum())},
    ]).to_csv(TABLES / "split_sizes.csv", index=False)

    # ---------------- F1 threshold tuned on validation, applied to test ----------------
    yc, yr = te["will_purchase"].to_numpy(), te["future_spend"].to_numpy()
    thr = val.set_index(["feature_set", "model"])["val_best_f1_threshold"]
    trows = []
    for fs in fsets:
        for m in CLF:
            p = preds[f"clf|{fs}|{m}"]
            p5, r5, f5 = prf_at(yc, p, 0.5)
            t = thr.loc[(fs, m)]
            pt, rt, ft = prf_at(yc, p, t)
            trows.append({"feature_set": fs, "model": m, "test_precision_at_0.5": p5, "test_recall_at_0.5": r5,
                          "test_f1_at_0.5": f5, "tuned_threshold_from_validation": t,
                          "val_f1_at_tuned_threshold": val.set_index(["feature_set", "model"]).loc[
                              (fs, m), "val_f1_at_best_threshold"],
                          "test_precision_at_tuned": pt, "test_recall_at_tuned": rt, "test_f1_at_tuned": ft})
    tune = pd.DataFrame(trows)
    tune.to_csv(TABLES / "threshold_tuning.csv", index=False)
    print(tune.round(4).to_string())

    # ---------------- hurdle model ----------------
    n1 = int(tr["will_purchase"].sum())
    n0 = len(tr) - n1
    y_te = yr
    buy_te = y_te > 0
    tr_buy = tr[tr["future_spend"] > 0]
    print(f"stage-2 training buyers (future_spend>0): {len(tr_buy):,}; test buyers: {int(buy_te.sum()):,}")
    hrows, crows = [], []
    for fs, cols in fsets.items():
        best = val[val["feature_set"] == fs].sort_values("val_roc_auc", ascending=False).iloc[0]["model"]
        p_raw = preds[f"clf|{fs}|{best}"]
        p_cal = prior_correct(p_raw, n1, n0)
        stage2 = STAGE2().fit(tr_buy[cols], tr_buy["future_spend"])
        e_spend = stage2.predict(te[cols])
        joblib.dump(stage2, MODELS / f"hurdle_stage2_{fs}.joblib")
        candidates = {
            "LinearRegression (single-stage)": preds[f"reg|{fs}|LinearRegression"],
            "RandomForest (single-stage)": preds[f"reg|{fs}|RandomForest"],
            f"Hurdle: P[{best}, prior-corrected] x E[spend|buy]": p_cal * e_spend,
            f"Hurdle: P[{best}, raw balanced] x E[spend|buy]": p_raw * e_spend,
        }
        preds[f"hurdle|{fs}"] = p_cal * e_spend
        preds[f"hurdle_raw|{fs}"] = p_raw * e_spend
        preds[f"stage2|{fs}"] = e_spend
        preds[f"pcal|{fs}"] = p_cal
        for subset, mask in [("all_test_users", np.ones_like(buy_te)), ("test_buyers_only", buy_te)]:
            for name, pred in candidates.items():
                hrows.append({"feature_set": fs, "subset": subset, "n": int(mask.sum()), "model": name,
                              "stage1_classifier": best, **reg_metrics(y_te[mask], pred[mask])})
            if subset == "test_buyers_only":
                hrows.append({"feature_set": fs, "subset": subset, "n": int(mask.sum()),
                              "model": "Stage-2 RF alone: E[spend|buy]", "stage1_classifier": "",
                              **reg_metrics(y_te[mask], e_spend[mask])})
        crows.append({"feature_set": fs, "stage1_classifier": best, "stage2_train_buyers": len(tr_buy),
                      "train_positives_n1": n1, "train_negatives_n0": n0,
                      "test_purchase_rate": yc.mean(),
                      "mean_p_raw_balanced": p_raw.mean(), "mean_p_prior_corrected": p_cal.mean(),
                      "brier_raw": np.mean((p_raw - yc) ** 2), "brier_prior_corrected": np.mean((p_cal - yc) ** 2),
                      "sum_pred_spend_hurdle": float((p_cal * e_spend).sum()), "sum_actual_spend": float(y_te.sum())})
    hurdle = pd.DataFrame(hrows)
    hurdle.to_csv(TABLES / "hurdle_metrics.csv", index=False)
    pd.DataFrame(crows).to_csv(TABLES / "hurdle_calibration.csv", index=False)

    # ---------------- persist final models + manifest (read by 09_verify_models.py and app.py) ----------------
    rf_val = val[val["model"] == "RandomForest"].sort_values("val_roc_auc", ascending=False)
    demo_fs = rf_val.iloc[0]["feature_set"]
    manifest = {"note": "All models fit on the train split (seed 42). Feature values are raw obs-window features "
                        "(Oct 1-24); linear models include their StandardScaler inside the Pipeline.",
                "segmentation": {"file": "segmentation.joblib"},
                "prior_correction": {"n1_train_positives": n1, "n0_train_negatives": n0,
                                     "formula": "odds = p/(1-p) * n1/n0; p_corrected = odds/(1+odds)"},
                "final": {"feature_set": demo_fs,
                          "rule": "feature set of the Random Forest classifier with the highest validation ROC-AUC "
                                  "(also the SHAP model); used by the demo and the worked example"},
                "feature_sets": {}}
    for c in crows:
        fs = c["feature_set"]
        best = c["stage1_classifier"]
        stage1_file = f"rf_clf_{fs}.joblib" if best == "RandomForest" else f"hurdle_stage1_{best}_{fs}.joblib"
        if best != "RandomForest":
            joblib.dump(fitted[f"clf|{fs}|{best}"], MODELS / stage1_file)
        manifest["feature_sets"][fs] = {
            "columns": fsets[fs],
            "hurdle_stage1_classifier": {"model": best, "file": stage1_file,
                                         "rule": "highest validation ROC-AUC among the three classifiers"},
            "hurdle_stage2_regressor": {"file": f"hurdle_stage2_{fs}.joblib",
                                        "trained_on": f"{len(tr_buy)} train users with future_spend > 0"},
            "single_stage_rf_classifier": {"file": f"rf_clf_{fs}.joblib"},
            "single_stage_rf_regressor": {"file": f"rf_reg_{fs}.joblib"}}
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(hurdle.round(4).to_string())
    print(pd.DataFrame(crows).round(4).to_string())

    # ---------------- ablation summary ----------------
    key_metrics = {"classification": ["roc_auc", "pr_auc", "f1"], "regression": ["RMSE", "MAE", "R2"]}
    summ = []
    for task, mets in key_metrics.items():
        sub = honest[honest["task"] == task]
        for met in mets:
            pv = sub.pivot(index="model", columns="feature_set", values=met)
            better_low = met in ("RMSE", "MAE")
            pv["best_feature_set"] = pv[list(fsets)].idxmin(axis=1) if better_low else pv[list(fsets)].idxmax(axis=1)
            pv.insert(0, "metric", met)
            pv.insert(0, "task", task)
            summ.append(pv.reset_index())
    pv = tune.pivot(index="model", columns="feature_set", values="test_f1_at_tuned")
    pv["best_feature_set"] = pv[list(fsets)].idxmax(axis=1)
    pv.insert(0, "metric", "f1_at_tuned_threshold")
    pv.insert(0, "task", "classification")
    summ.append(pv.reset_index())
    for met in ["RMSE", "MAE", "R2"]:
        pv = hurdle[(hurdle["subset"] == "all_test_users") & hurdle["model"].str.contains("prior-corrected")] \
            .set_index("feature_set")[met]
        row = {"task": "regression", "metric": met, "model": "Hurdle (prior-corrected)", **pv.to_dict()}
        row["best_feature_set"] = pv.idxmin() if met != "R2" else pv.idxmax()
        summ.append(pd.DataFrame([row]))
    ablation = pd.concat(summ, ignore_index=True)
    ablation.to_csv(TABLES / "ablation_summary.csv", index=False)
    print(ablation.round(4).to_string())

    # ---------------- paired bootstrap #1: classification metrics + single-stage RMSE ----------------
    rng = np.random.default_rng(SEED)
    n = len(te)
    fs_names = list(fsets)
    pairs = [(fs_names[1], fs_names[0]), (fs_names[2], fs_names[1]), (fs_names[2], fs_names[0])]
    clf_mets = ["roc_auc", "pr_auc", "f1_at_0.5", "f1_at_tuned", "precision_at_tuned", "recall_at_tuned"]
    bc = {(fs, m): {k: np.empty(N_BOOT) for k in clf_mets} for fs in fs_names for m in CLF}
    reg_series = {m: {fs: preds[f"reg|{fs}|{m}"] for fs in fs_names} for m in REG}
    reg_series["Hurdle"] = {fs: preds[f"hurdle|{fs}"] for fs in fs_names}
    br = {m: {fs: np.empty(N_BOOT) for fs in fs_names} for m in reg_series}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        y = yc[idx]
        for (fs, m), store in bc.items():
            p = preds[f"clf|{fs}|{m}"][idx]
            store["roc_auc"][b] = roc_auc_score(y, p)
            store["pr_auc"][b] = average_precision_score(y, p)
            store["f1_at_0.5"][b] = prf_at(y, p, 0.5)[2]
            pt, rt, ft = prf_at(y, p, thr.loc[(fs, m)])
            store["precision_at_tuned"][b], store["recall_at_tuned"][b], store["f1_at_tuned"][b] = pt, rt, ft
        for m, per_fs in reg_series.items():
            for fs, p in per_fs.items():
                br[m][fs][b] = root_mean_squared_error(yr[idx], p[idx])
    brows = []
    for m in CLF:
        for hi, lo in pairs:
            d = bc[(hi, m)]["roc_auc"] - bc[(lo, m)]["roc_auc"]
            brows.append({"task": "classification", "metric": "roc_auc", "model": m, "comparison": f"{hi} minus {lo}",
                          "mean_diff": d.mean(), **ci_row(d), "share_of_resamples_improved": float(np.mean(d > 0))})
    for m, per_fs in br.items():
        for hi, lo in pairs:
            d = per_fs[hi] - per_fs[lo]
            brows.append({"task": "regression", "metric": "RMSE", "model": m, "comparison": f"{hi} minus {lo}",
                          "mean_diff": d.mean(), **ci_row(d), "share_of_resamples_improved": float(np.mean(d < 0))})
    for m in CLF:
        for hi, lo in pairs:
            d = bc[(hi, m)]["f1_at_tuned"] - bc[(lo, m)]["f1_at_tuned"]
            brows.append({"task": "classification", "metric": "f1_at_tuned", "model": m,
                          "comparison": f"{hi} minus {lo}", "mean_diff": d.mean(), **ci_row(d),
                          "share_of_resamples_improved": float(np.mean(d > 0))})
    bdf = pd.DataFrame(brows)
    bdf.to_csv(TABLES / "ablation_bootstrap.csv", index=False)
    print(bdf.round(5).to_string())

    point = {}
    for fs in fs_names:
        for m in CLF:
            p = preds[f"clf|{fs}|{m}"]
            pt, rt, ft = prf_at(yc, p, thr.loc[(fs, m)])
            point[(fs, m)] = {"roc_auc": roc_auc_score(yc, p), "pr_auc": average_precision_score(yc, p),
                              "f1_at_0.5": prf_at(yc, p, 0.5)[2], "f1_at_tuned": ft,
                              "precision_at_tuned": pt, "recall_at_tuned": rt}
    pd.DataFrame([{"feature_set": fs, "model": m, "metric": k, "value": point[(fs, m)][k], **ci_row(arr)}
                  for (fs, m), store in bc.items() for k, arr in store.items()]) \
        .to_csv(TABLES / "classification_ci.csv", index=False)

    # ---------------- paired bootstrap #2: hurdle vs single-stage + per-model spend metrics ----------------
    rng2 = np.random.default_rng(SEED)
    comps = [(fs, base) for fs in fs_names for base in ["RandomForest", "LinearRegression"]]
    diffs = {c: {"RMSE": np.empty(N_BOOT), "MAE": np.empty(N_BOOT)} for c in comps}
    spend_models = {fs: {"LinearRegression (single-stage)": preds[f"reg|{fs}|LinearRegression"],
                         "DecisionTree (single-stage)": preds[f"reg|{fs}|DecisionTree"],
                         "RandomForest (single-stage)": preds[f"reg|{fs}|RandomForest"],
                         "Hurdle (prior-corrected)": preds[f"hurdle|{fs}"]} for fs in fs_names}
    subsets = ["all_test_users", "test_buyers_only"]
    sm = {(fs, name, s, met): np.empty(N_BOOT) for fs in fs_names for name in spend_models[fs] for s in subsets
          for met in ["MAE", "RMSE", "R2"]}
    for b in range(N_BOOT):
        idx = rng2.integers(0, n, n)
        for fs, base in comps:
            h, s = preds[f"hurdle|{fs}"][idx], preds[f"reg|{fs}|{base}"][idx]
            diffs[(fs, base)]["RMSE"][b] = root_mean_squared_error(yr[idx], h) - root_mean_squared_error(yr[idx], s)
            diffs[(fs, base)]["MAE"][b] = mean_absolute_error(yr[idx], h) - mean_absolute_error(yr[idx], s)
        buy_idx = idx[yr[idx] > 0]
        for fs, models in spend_models.items():
            for name, p in models.items():
                for s, ii in (("all_test_users", idx), ("test_buyers_only", buy_idx)):
                    y, q = yr[ii], p[ii]
                    sm[(fs, name, s, "MAE")][b] = np.mean(np.abs(q - y))
                    sm[(fs, name, s, "RMSE")][b] = np.sqrt(np.mean((q - y) ** 2))
                    sm[(fs, name, s, "R2")][b] = 1 - np.sum((q - y) ** 2) / np.sum((y - y.mean()) ** 2)
    hb = pd.DataFrame([{"feature_set": fs, "comparison": f"Hurdle (prior-corrected) minus {base} (single-stage)",
                        "metric": met, "mean_diff": d.mean(), "ci95_low": np.percentile(d, 2.5),
                        "ci95_high": np.percentile(d, 97.5), "share_of_resamples_hurdle_better": float(np.mean(d < 0))}
                       for (fs, base), per in diffs.items() for met, d in per.items()])
    hb.to_csv(TABLES / "hurdle_bootstrap.csv", index=False)
    print(hb.round(4).to_string())
    rrows = []
    for (fs, name, s, met), arr in sm.items():
        mask = np.ones(n, bool) if s == "all_test_users" else yr > 0
        val_pt = reg_metrics(yr[mask], spend_models[fs][name][mask])[met]
        rrows.append({"feature_set": fs, "model": name, "subset": s, "n": int(mask.sum()), "metric": met,
                      "value": val_pt, **ci_row(arr)})
    pd.DataFrame(rrows).to_csv(TABLES / "regression_ci.csv", index=False)

    # How concentrated are the regression errors? (heavy-tailed target -> a few users dominate RMSE/R2)
    erows = []
    for key in [k for k in preds if k.startswith(("reg|", "hurdle|"))]:
        p = preds[key]
        se = (p - yr) ** 2
        erows.append({"prediction": key, "share_of_SSE_from_top10_users": np.sort(se)[-10:].sum() / se.sum(),
                      "share_of_SSE_from_top1_user": se.max() / se.sum(), "max_prediction": p.max(),
                      "share_negative_predictions": float(np.mean(p < 0))})
    err = pd.DataFrame(erows)
    err.to_csv(TABLES / "regression_error_concentration.csv", index=False)
    print(err.round(4).to_string())

    # ---------------- save test predictions ----------------
    out = pd.DataFrame({"will_purchase": yc, "future_spend": yr, "segment": seg.loc[te.index].astype(str).to_numpy()},
                       index=te.index)
    out = pd.concat([out, pd.DataFrame(preds, index=te.index)], axis=1)
    out.to_parquet(PREDICTIONS)

    # ---------------- leakage audit (deliberately flawed setup) ----------------
    leaky = pd.read_parquet(FEATURES_LEAKY)
    k = int(pd.read_csv(TABLES / "k_selection.csv")["chosen_k"].iloc[0])
    clus_cols = [c for c in RFM + BEHAVIORAL if c in fsets["b_rfm_behav"]]
    _, leaky_seg, _, _, _ = fit_segments(leaky, clus_cols, k)
    print("leaky segments:", leaky_seg.value_counts().to_dict())
    leaky = add_segment_dummies(leaky, leaky_seg)
    leaky_fsets = feature_sets(leaky)
    leaky_res, _, _ = fit_all(leaky, leaky_fsets, "LEAKY_full_month")
    audit_mets = {"classification": ["roc_auc", "pr_auc", "f1", "recall", "precision"],
                  "regression": ["R2", "RMSE", "MAE"]}
    arows = []
    for (task, fs, model), h in honest.set_index(["task", "feature_set", "model"]).iterrows():
        lk = leaky_res.set_index(["task", "feature_set", "model"]).loc[(task, fs, model)]
        for met in audit_mets[task]:
            arows.append({"setup_note": "LEAKY = features from Oct 1-31 (overlaps target window); deliberately flawed",
                          "task": task, "feature_set": fs, "model": model, "metric": met,
                          "honest": h[met], "leaky": lk[met], "inflation_leaky_minus_honest": lk[met] - h[met]})
    audit = pd.DataFrame(arows)
    audit.to_csv(TABLES / "leakage_audit.csv", index=False)
    print(audit.drop(columns="setup_note").round(4).to_string())


if __name__ == "__main__":
    main()
