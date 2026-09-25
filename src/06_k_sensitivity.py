"""Step 6: K sensitivity. Main results use K=3; here feature set (c) and the segment profiles are rerun with
K=2 and K=4 (same features, split, models, validation rule, hurdle and bootstrap settings).

Outputs (outputs/tables): k_sensitivity_metrics.csv, k_sensitivity_profiles.csv, k_sensitivity_conclusions.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score, root_mean_squared_error

from common import (CLUSTERS, FEATURES, PREDICTIONS, SEED, TABLES, add_segment_dummies, feature_sets, fit_segments)
from modeling import CLF, N_BOOT, STAGE2, fit_all, prior_correct, reg_metrics, validate_classifiers

K_ALT = [2, 4]
FS_B, FS_C = "b_rfm_behav", "c_rfm_behav_cluster"


def profile(df: pd.DataFrame, seg: pd.Series, k: int) -> pd.DataFrame:
    p = df[["has_purchased", "monetary", "frequency", "n_views", "n_sessions", "days_active", "will_purchase",
            "future_spend"]].groupby(seg, observed=True).mean()
    p.insert(0, "n_users", seg.value_counts())
    p.insert(1, "share", p["n_users"] / len(df))
    p = p.rename(columns={"will_purchase": "pred_window_purchase_rate", "future_spend": "pred_window_mean_spend"})
    return p.reset_index(names="segment").assign(k=k)


def fmt_ci(d: np.ndarray) -> str:
    return f"{d.mean():+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"


def main():
    df = pd.read_parquet(FEATURES)
    pm = pd.read_parquet(PREDICTIONS)
    k_main = int(pd.read_csv(TABLES / "k_selection.csv")["chosen_k"].iloc[0])
    cols_b = feature_sets(df)[FS_B]
    tr = df[df["split"] == "train"]
    te = df.loc[pm.index]
    yc, yr = te["will_purchase"].to_numpy(), te["future_spend"].to_numpy()
    n1 = int(tr["will_purchase"].sum())
    n0 = len(tr) - n1
    val_main = pd.read_csv(TABLES / "classifier_validation.csv")

    runs = {}   # k -> dict(preds, val, seg, metric rows)
    main_seg = pd.read_parquet(CLUSTERS)["segment"]
    runs[k_main] = {
        "seg": main_seg,
        "preds": {**{f"clf|{m}": pm[f"clf|{FS_C}|{m}"].to_numpy() for m in CLF},
                  **{f"reg|{m}": pm[f"reg|{FS_C}|{m}"].to_numpy() for m in ["LinearRegression", "DecisionTree",
                                                                            "RandomForest"]},
                  "hurdle": pm[f"hurdle|{FS_C}"].to_numpy()},
        "val": val_main[val_main["feature_set"] == FS_C],
    }
    for k in K_ALT:
        _, seg, _, _, mapping = fit_segments(df, cols_b, k)
        print(f"K={k}: {mapping}", flush=True)
        dk = add_segment_dummies(df, seg)
        cols_c = feature_sets(dk)[FS_C]
        _, preds, _ = fit_all(dk, {FS_C: cols_c}, f"K={k}")
        val = validate_classifiers(dk, {FS_C: cols_c})
        best = val.sort_values("val_roc_auc", ascending=False).iloc[0]["model"]
        trk = dk[dk["split"] == "train"]
        buyers = trk[trk["future_spend"] > 0]
        e = STAGE2().fit(buyers[cols_c], buyers["future_spend"]).predict(dk.loc[pm.index, cols_c])
        p_cal = prior_correct(preds[f"clf|{FS_C}|{best}"], n1, n0)
        runs[k] = {"seg": seg, "val": val,
                   "preds": {**{f"clf|{m}": preds[f"clf|{FS_C}|{m}"] for m in CLF},
                             **{f"reg|{m}": preds[f"reg|{FS_C}|{m}"] for m in ["LinearRegression", "DecisionTree",
                                                                               "RandomForest"]},
                             "hurdle": p_cal * e}}
    ks = sorted(runs)

    # ---- point metrics for (c) at each K ----
    mrows = []
    for k in ks:
        P = runs[k]["preds"]
        for m in CLF:
            mrows.append({"k": k, "task": "classification", "model": m, "roc_auc": roc_auc_score(yc, P[f"clf|{m}"])})
        for m in ["LinearRegression", "DecisionTree", "RandomForest"]:
            mrows.append({"k": k, "task": "regression", "model": f"{m} (single-stage)", **reg_metrics(yr, P[f"reg|{m}"])})
        best = runs[k]["val"].sort_values("val_roc_auc", ascending=False).iloc[0]["model"]
        mrows.append({"k": k, "task": "regression", "model": f"Hurdle (stage 1 = {best})",
                      **reg_metrics(yr, P["hurdle"])})
    metrics = pd.DataFrame(mrows)
    metrics.insert(1, "feature_set", FS_C)
    metrics.to_csv(TABLES / "k_sensitivity_metrics.csv", index=False)
    print(metrics.round(4).to_string())

    profiles = pd.concat([profile(df, runs[k]["seg"], k) for k in ks], ignore_index=True)
    profiles = profiles[["k"] + [c for c in profiles.columns if c != "k"]]
    profiles.to_csv(TABLES / "k_sensitivity_profiles.csv", index=False)
    print(profiles.round(3).to_string())

    # ---- paired bootstrap (same resamples as 04: default_rng(SEED), one draw per resample) ----
    rng = np.random.default_rng(SEED)
    n = len(te)
    auc = {(k, m): np.empty(N_BOOT) for k in ks for m in CLF}
    auc_b = {m: np.empty(N_BOOT) for m in CLF}
    d_rmse = {k: np.empty(N_BOOT) for k in ks}
    d_mae = {k: np.empty(N_BOOT) for k in ks}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for m in CLF:
            auc_b[m][b] = roc_auc_score(yc[idx], pm[f"clf|{FS_B}|{m}"].to_numpy()[idx])
            for k in ks:
                auc[(k, m)][b] = roc_auc_score(yc[idx], runs[k]["preds"][f"clf|{m}"][idx])
        for k in ks:
            h, s = runs[k]["preds"]["hurdle"][idx], runs[k]["preds"]["reg|RandomForest"][idx]
            d_rmse[k][b] = root_mean_squared_error(yr[idx], h) - root_mean_squared_error(yr[idx], s)
            d_mae[k][b] = mean_absolute_error(yr[idx], h) - mean_absolute_error(yr[idx], s)

    # ---- conclusions: does anything change relative to K=3? ----
    hurdle_b_rmse = root_mean_squared_error(yr, pm[f"hurdle|{FS_B}"])
    rf_val_b = val_main[(val_main["feature_set"] == FS_B) & (val_main["model"] == "RandomForest")]["val_roc_auc"].item()
    checks = []
    for m in CLF:
        checks.append((f"(c) minus (b) test ROC-AUC, {m} [95% CI]", "(c) significantly better: CI low > 0",
                       {k: (fmt_ci(auc[(k, m)] - auc_b[m]), np.percentile(auc[(k, m)] - auc_b[m], 2.5) > 0)
                        for k in ks}))
    checks.append(("Hurdle (c) minus single-stage RF (c), RMSE [95% CI]", "hurdle better: CI high < 0",
                   {k: (fmt_ci(d_rmse[k]), np.percentile(d_rmse[k], 97.5) < 0) for k in ks}))
    checks.append(("Hurdle (c) minus single-stage RF (c), MAE [95% CI]", "hurdle better: CI high < 0",
                   {k: (fmt_ci(d_mae[k]), np.percentile(d_mae[k], 97.5) < 0) for k in ks}))
    checks.append(("Hurdle test RMSE, (c) vs (b)", "(c) lower than (b)",
                   {k: (f"c={root_mean_squared_error(yr, runs[k]['preds']['hurdle']):.3f} vs b={hurdle_b_rmse:.3f}",
                        root_mean_squared_error(yr, runs[k]["preds"]["hurdle"]) < hurdle_b_rmse) for k in ks}))
    checks.append(("Stage-1 classifier chosen for (c) by validation ROC-AUC", "same model as K=3",
                   {k: (runs[k]["val"].sort_values("val_roc_auc", ascending=False).iloc[0]["model"], None) for k in ks}))
    checks.append(("RF validation ROC-AUC, (c) vs (b)", "(c) > (b) (decides which RF is explained by SHAP)",
                   {k: (f"c={runs[k]['val'].set_index('model').loc['RandomForest', 'val_roc_auc']:.4f} vs "
                        f"b={rf_val_b:.4f}",
                        runs[k]["val"].set_index("model").loc["RandomForest", "val_roc_auc"] > rf_val_b) for k in ks}))
    for k in ks:
        pk = profiles[profiles["k"] == k].set_index("segment")
        runs[k]["names"] = " | ".join(pk.index)
        buyer_segs = pk.index[pk["has_purchased"] >= 0.5]
        runs[k]["buyers_top"] = bool(len(buyer_segs)) and pk.loc[buyer_segs, "pred_window_purchase_rate"].min() > \
            pk.drop(index=buyer_segs)["pred_window_purchase_rate"].max()
        runs[k]["engaged"] = any(s.startswith("Engaged browsers") for s in pk.index)
    checks.append(("Segments found", "same segment types as K=3", {k: (runs[k]["names"], None) for k in ks}))
    checks.append(("Buyer segment(s) have the highest Oct 25-31 purchase rate", "yes",
                   {k: ("yes" if runs[k]["buyers_top"] else "no", runs[k]["buyers_top"]) for k in ks}))
    checks.append(("An 'Engaged browsers' segment exists", "yes",
                   {k: ("yes" if runs[k]["engaged"] else "no", runs[k]["engaged"]) for k in ks}))

    crow = []
    for name, crit, per_k in checks:
        row = {"check": name, "criterion": crit}
        for k in ks:
            v, ok = per_k[k]
            row[f"K={k} value"] = v
            row[f"K={k} result"] = "" if ok is None else ("holds" if ok else "does not hold")
        base_v, base_ok = per_k[k_main]
        for k in ks:
            if k == k_main:
                continue
            v, ok = per_k[k]
            if ok is None:
                same = v == base_v if name != "Segments found" else \
                    sorted({s.split(",")[0].rstrip(" 0123456789") for s in v.split(" | ")}) == \
                    sorted({s.split(",")[0].rstrip(" 0123456789") for s in base_v.split(" | ")})
            else:
                same = ok == base_ok
            row[f"K={k}: same conclusion as K={k_main}"] = "yes" if same else "NO"
        crow.append(row)
    conc = pd.DataFrame(crow)
    conc.to_csv(TABLES / "k_sensitivity_conclusions.csv", index=False)
    print(conc.to_string())


if __name__ == "__main__":
    main()
