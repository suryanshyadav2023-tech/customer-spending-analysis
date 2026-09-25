"""Step 7: report package in outputs/report/ (tables T1-T6 as CSV + markdown, architecture diagram, figures at
300 dpi + figures_list.md, methods_facts.md, worked_example.md). Every number is read from pipeline outputs.

`python 07_report.py --methods-only` rewrites methods_facts.md only (run_all.py calls this at the end so the
runtime section includes every step).
"""
import os
import platform
import shutil
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd

from common import (BEHAVIORAL, BUYER_SHARE_MIN, CLUSTERS, FEATURES, FIGURES, FS_LABEL, OBS_START, PRED_END,
                    PRED_START, PREDICTIONS, RAW_CSV, RECENCY_CAP_DAYS, REPORT, RFM, ROOT, SAMPLE_MOD, SEED, SERIES,
                    SURFACE, TABLES, TEXT, TEXT_2, WORKED_EXAMPLE_USERS, feature_sets, label, set_style, slug)
from modeling import CLF, LOGREG_KW, N_BOOT, REG, RF_KW, STAGE2, TEST_SIZE, TREE_KW, VAL_SIZE

RT = REPORT / "tables"
RF_FIG = REPORT / "figures"
MODEL_LABEL = {"LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree",
               "RandomForest": "Random Forest", "LinearRegression": "Linear Regression"}
METRIC_LABEL = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 @0.5", "recall": "Recall @0.5",
                "precision": "Precision @0.5", "R2": "R²", "RMSE": "RMSE", "MAE": "MAE"}


def t(name):
    return pd.read_csv(TABLES / name)


def kv(name):
    return pd.read_csv(TABLES / name).set_index("stat")["value"]


def md_table(df: pd.DataFrame) -> str:
    esc = lambda v: str(v).replace("|", "\\|")
    head = "| " + " | ".join(map(esc, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(map(esc, r)) + " |" for r in df.itertuples(index=False)])


def save(name: str, csv_df: pd.DataFrame, md_df: pd.DataFrame, title: str, notes=()):
    csv_df.to_csv(RT / f"{name}.csv", index=False)
    body = [f"# {title}", "", md_table(md_df), ""] + [f"- {n}" for n in notes]
    (RT / f"{name}.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def f4(x):
    return f"{x:.4f}"


def f3(x):
    return f"{x:,.3f}"


def ci(lo, hi, f=f4):
    return f"[{f(lo)}, {f(hi)}]"


# ------------------------------------------------------------------------------------------------ tables
def t1():
    ss, ps, sp = kv("sample_stats.csv"), kv("population_stats.csv"), t("split_sizes.csv").set_index("part")
    rows = [
        ("Source file", "2019-Oct.csv (Kaggle: mkechinov/ecommerce-behavior-data-from-multi-category-store)"),
        ("Raw events", f"{int(ss['raw_rows']):,}"),
        ("Raw users", f"{int(ss['raw_users']):,}"),
        ("Sampling rule", f"user_id % {SAMPLE_MOD} == 0"),
        ("Sampled events", f"{int(ss['sample_rows']):,}"),
        ("Sampled users", f"{int(ss['sample_users']):,}"),
        ("Sample fraction (users)", f4(float(ss["sample_user_share"]))),
        ("Observation window (features)", f"{OBS_START:%Y-%m-%d} to {PRED_START - pd.Timedelta(seconds=1):%Y-%m-%d %H:%M:%S} UTC"),
        ("Events in observation window", f"{int(ps['obs_window_events']):,}"),
        ("Prediction window (targets)", f"{PRED_START:%Y-%m-%d} to {PRED_END - pd.Timedelta(seconds=1):%Y-%m-%d %H:%M:%S} UTC"),
        ("Events in prediction window", f"{int(ps['pred_window_events']):,}"),
        ("Population (users with >= 1 event in observation window)", f"{int(ps['population_users']):,}"),
        ("Buyers in prediction window (population)", f"{int(ps['will_purchase_positive']):,}"),
        ("Buy rate in prediction window (all / train / test)",
         f"{f4(float(ps['will_purchase_rate']))} / {f4(float(ps['will_purchase_rate_train']))} / "
         f"{f4(float(ps['will_purchase_rate_test']))}"),
        ("Mean spend among prediction-window buyers", f3(float(ps["future_spend_mean_buyers"]))),
        ("Median spend among prediction-window buyers", f3(float(ps["future_spend_median_buyers"]))),
        ("Train users (all)", f"{int(sp.loc['train (all)', 'users']):,}"),
        ("  of which inner-train users", f"{int(sp.loc['inner train (model choice / thresholds)', 'users']):,}"),
        ("  of which validation users", f"{int(sp.loc['validation (part of train)', 'users']):,}"),
        ("Test users", f"{int(sp.loc['test', 'users']):,}"),
        ("Test buyers (prediction window)", f"{int(sp.loc['test', 'buyers_oct25_31']):,}"),
    ]
    d = pd.DataFrame(rows, columns=["Item", "Value"])
    save("T1_dataset_summary", d, d, "T1. Dataset summary")


def t2():
    ci_df = t("classification_ci.csv").set_index(["feature_set", "model", "metric"])
    cm = t("classification_metrics.csv").set_index(["feature_set", "model"])
    tune = t("threshold_tuning.csv").set_index(["feature_set", "model"])
    csv_rows, md_rows = [], []
    for fs in FS_LABEL:
        for m in CLF:
            g = lambda k: ci_df.loc[(fs, m, k)]
            r = {"Feature set": FS_LABEL[fs], "Model": MODEL_LABEL[m]}
            for k, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1_at_0.5", "F1 @0.5")]:
                r[name], r[f"{name} CI low"], r[f"{name} CI high"] = g(k)["value"], g(k)["ci95_low"], g(k)["ci95_high"]
            r["Precision @0.5"], r["Recall @0.5"] = cm.loc[(fs, m), "precision"], cm.loc[(fs, m), "recall"]
            r["Tuned threshold (validation)"] = tune.loc[(fs, m), "tuned_threshold_from_validation"]
            k = "f1_at_tuned"
            r["F1 @tuned"], r["F1 @tuned CI low"], r["F1 @tuned CI high"] = g(k)["value"], g(k)["ci95_low"], g(k)["ci95_high"]
            r["Precision @tuned"], r["Recall @tuned"] = g("precision_at_tuned")["value"], g("recall_at_tuned")["value"]
            csv_rows.append(r)
            md_rows.append({"Feature set": r["Feature set"], "Model": r["Model"],
                            "ROC-AUC [95% CI]": f"{f4(r['ROC-AUC'])} {ci(r['ROC-AUC CI low'], r['ROC-AUC CI high'])}",
                            "PR-AUC [95% CI]": f"{f4(r['PR-AUC'])} {ci(r['PR-AUC CI low'], r['PR-AUC CI high'])}",
                            "F1 @0.5 [95% CI]": f"{f4(r['F1 @0.5'])} {ci(r['F1 @0.5 CI low'], r['F1 @0.5 CI high'])}",
                            "Precision / Recall @0.5": f"{f4(r['Precision @0.5'])} / {f4(r['Recall @0.5'])}",
                            "Tuned threshold": f4(r["Tuned threshold (validation)"]),
                            "F1 @tuned [95% CI]": f"{f4(r['F1 @tuned'])} {ci(r['F1 @tuned CI low'], r['F1 @tuned CI high'])}",
                            "Precision / Recall @tuned": f"{f4(r['Precision @tuned'])} / {f4(r['Recall @tuned'])}"})
    c = pd.DataFrame(csv_rows)
    save("T2_classification", c.round(4), pd.DataFrame(md_rows), "T2. Classification: will_purchase (test set)",
         [f"Test set: {int(t('split_sizes.csv').set_index('part').loc['test', 'users']):,} users. "
          f"95% CI = percentile interval over {N_BOOT} paired bootstrap resamples of test users (seed {SEED}).",
          "Tuned threshold = threshold maximising F1 on the validation split (part of train); never chosen on test.",
          "All classifiers use class_weight='balanced'. PR-AUC = average precision."])
    b = t("ablation_bootstrap.csv")
    b = b[b["task"] == "classification"].copy()
    lab = {"b_rfm_behav minus a_rfm": "(b) - (a)", "c_rfm_behav_cluster minus b_rfm_behav": "(c) - (b)",
           "c_rfm_behav_cluster minus a_rfm": "(c) - (a)"}
    d = pd.DataFrame({"Metric": b["metric"].map({"roc_auc": "ROC-AUC", "f1_at_tuned": "F1 @tuned"}),
                      "Model": b["model"].map(MODEL_LABEL), "Comparison": b["comparison"].map(lab),
                      "Mean difference": b["mean_diff"], "CI low": b["ci95_low"], "CI high": b["ci95_high"],
                      "Share of resamples improved": b["share_of_resamples_improved"]})
    dm = d.assign(**{"Mean difference": d["Mean difference"].map(lambda v: f"{v:+.4f}"),
                     "95% CI": [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(d["CI low"], d["CI high"])],
                     "Share of resamples improved": d["Share of resamples improved"].map(f4)}) \
        .drop(columns=["CI low", "CI high"])
    save("T2b_classification_feature_set_differences", d.round(4), dm,
         "T2b. Classification: paired differences between feature sets (test set)",
         [f"Paired bootstrap, {N_BOOT} resamples of test users; positive = later feature set better."])


def t3():
    r = t("regression_ci.csv")
    order = ["LinearRegression (single-stage)", "DecisionTree (single-stage)", "RandomForest (single-stage)",
             "Hurdle (prior-corrected)"]
    mlab = {m: MODEL_LABEL[m.split(" ")[0]] + " (single-stage)" for m in order[:3]}
    mlab["Hurdle (prior-corrected)"] = "Hurdle: P(buy) x E[spend | buy]"
    cal = t("hurdle_calibration.csv").set_index("feature_set")
    csv_rows, md_rows = [], []
    for fs in FS_LABEL:
        for subset, slab in [("all_test_users", "All test users"), ("test_buyers_only", "Test buyers only")]:
            for m in order:
                g = r[(r["feature_set"] == fs) & (r["model"] == m) & (r["subset"] == subset)].set_index("metric")
                row = {"Feature set": FS_LABEL[fs], "Evaluated on": slab, "n": int(g["n"].iloc[0]), "Model": mlab[m]}
                mdr = dict(row)
                for met, name in [("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²")]:
                    row[name], row[f"{name} CI low"], row[f"{name} CI high"] = \
                        g.loc[met, "value"], g.loc[met, "ci95_low"], g.loc[met, "ci95_high"]
                    fmt = f4 if met == "R2" else f3
                    mdr[f"{name} [95% CI]"] = f"{fmt(row[name])} {ci(row[f'{name} CI low'], row[f'{name} CI high'], fmt)}"
                csv_rows.append(row)
                md_rows.append({**mdr, "n": f"{mdr['n']:,}"})
    save("T3_spend", pd.DataFrame(csv_rows).round(4), pd.DataFrame(md_rows),
         "T3. Spend prediction: single-stage vs hurdle (future_spend, Oct 25-31, test set)",
         [f"Hurdle stage 1 = classifier with the highest validation ROC-AUC per feature set "
          f"({', '.join(f'{FS_LABEL[fs]}: {MODEL_LABEL[cal.loc[fs, 'stage1_classifier']]}' for fs in FS_LABEL)}), "
          "probabilities prior-corrected for class_weight='balanced'.",
          f"Hurdle stage 2 = Random Forest regressor trained on {int(cal['stage2_train_buyers'].iloc[0]):,} train users "
          "with future_spend > 0.",
          f"95% CI = percentile interval over {N_BOOT} bootstrap resamples of test users; the buyers-only intervals "
          "use the buyers inside each resample."])
    h = t("hurdle_bootstrap.csv")
    d = pd.DataFrame({"Feature set": h["feature_set"].map(FS_LABEL),
                      "Comparison": h["comparison"].str.replace("RandomForest", "Random Forest")
                      .str.replace("LinearRegression", "Linear Regression"),
                      "Metric": h["metric"], "Mean difference": h["mean_diff"], "CI low": h["ci95_low"],
                      "CI high": h["ci95_high"], "Share of resamples hurdle better": h["share_of_resamples_hurdle_better"]})
    dm = d.assign(**{"Mean difference": d["Mean difference"].map(lambda v: f"{v:+.3f}"),
                     "95% CI": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(d["CI low"], d["CI high"])],
                     "Share of resamples hurdle better": d["Share of resamples hurdle better"].map(f4)}) \
        .drop(columns=["CI low", "CI high"])
    save("T3b_hurdle_vs_single_stage", d.round(4), dm,
         "T3b. Hurdle minus single-stage, all test users (negative = hurdle better)",
         [f"Paired bootstrap, {N_BOOT} resamples of test users."])


def t4():
    a = t("leakage_audit.csv")
    a = a[a["metric"].isin(["roc_auc", "pr_auc", "f1", "R2", "RMSE", "MAE"])]
    d = pd.DataFrame({"Task": a["task"].str.capitalize(), "Feature set": a["feature_set"].map(FS_LABEL),
                      "Model": a["model"].map(MODEL_LABEL), "Metric": a["metric"].map(METRIC_LABEL),
                      "Honest (features Oct 1-24)": a["honest"], "Leaky (features Oct 1-31)": a["leaky"],
                      "Inflation (leaky - honest)": a["inflation_leaky_minus_honest"]})
    dm = d.copy()
    for c in ["Honest (features Oct 1-24)", "Leaky (features Oct 1-31)"]:
        dm[c] = [f3(v) if m in ("RMSE", "MAE") else f4(v) for v, m in zip(d[c], d["Metric"])]
    dm["Inflation (leaky - honest)"] = [f"{v:+,.3f}" if m in ("RMSE", "MAE") else f"{v:+.4f}"
                                       for v, m in zip(d["Inflation (leaky - honest)"], d["Metric"])]
    save("T4_leakage_audit", d.round(4), dm, "T4. Leakage audit: honest vs deliberately leaky features (test set)",
         ["LEAKY setup is deliberately flawed and used only for comparison: features are rebuilt from Oct 1-31, "
          "which overlaps the Oct 25-31 target window. Same users, targets, split, models and K-means K.",
          "For RMSE and MAE a negative inflation means the leaky model looks better."])


def t5():
    p = t("cluster_profiles.csv")
    cols = {"segment": "Segment", "n_users": "Users", "share": "Share of users", "n_train": "Train users",
            "has_purchased": "Purchased Oct 1-24 (share)", "monetary": "Mean monetary", "frequency": "Mean frequency",
            "recency_days": "Mean recency (days)", "n_sessions": "Mean sessions", "n_views": "Mean product views",
            "n_carts": "Mean add-to-cart events", "days_active": "Mean days active",
            "avg_session_duration": "Mean session duration (min)", "n_categories": "Mean categories viewed",
            "days_since_last_event": "Mean days since last event", "cart_to_view_ratio": "Mean cart-to-view ratio",
            "purchase_to_cart_ratio": "Mean purchase-to-cart ratio",
            "pred_window_purchase_rate": "Oct 25-31 purchase rate", "pred_window_mean_spend": "Oct 25-31 mean spend"}
    d = p[list(cols)].rename(columns=cols)
    dm = d.copy()
    for c in d.columns[1:]:
        dm[c] = d[c].map(lambda v: f"{int(v):,}") if c in ("Users", "Train users") else \
            d[c].map(f4 if c in ("Share of users", "Purchased Oct 1-24 (share)", "Oct 25-31 purchase rate") else f3)
    save("T5_cluster_profiles", d.round(4), dm, "T5. Segment profiles (K-means, K=3; means over all population users)",
         ["Features are from Oct 1-24. The two Oct 25-31 columns are descriptive only and were not used for "
          "clustering or naming.", "Naming rule and the numbers behind each name: outputs/report/cluster_naming.md."])


def t6():
    o = t("shap_importance.csv")
    s = t("shap_top5_per_segment.csv")
    val = t("classifier_validation.csv")
    best_fs = val[val["model"] == "RandomForest"].sort_values("val_roc_auc", ascending=False).iloc[0]["feature_set"]
    rows = [{"Scope": "Overall", "Ranking": "all features", "n users": 2000, "Rank": i + 1,
             "Feature": r["label"], "Mean |SHAP|": r["mean_abs_shap"]} for i, r in o.head(10).iterrows()]
    for _, r in s.iterrows():
        rows.append({"Scope": f"Segment: {r['segment']}", "Ranking": r["ranking"].replace("_", " "),
                     "n users": int(r["n_shap_rows"]), "Rank": int(r["rank"]), "Feature": r["label"],
                     "Mean |SHAP|": r["mean_abs_shap"]})
    d = pd.DataFrame(rows)
    dm = d.assign(**{"Mean |SHAP|": d["Mean |SHAP|"].map(lambda v: f"{v:.4f}"),
                     "n users": d["n users"].map(lambda v: f"{v:,}")})
    save("T6_shap", d.round(5), dm, "T6. Top SHAP features, overall (top 10) and per segment (top 5)",
         [f"Model: Random Forest classifier for will_purchase on {FS_LABEL[best_fs]} (highest validation ROC-AUC "
          "among Random Forest classifiers). TreeExplainer, positive class.",
          "Segment indicators are constant within a segment, so their per-segment SHAP reflects the gap to the "
          "overall baseline; the 'excluding seg dummies' ranking leaves them out."])


# ------------------------------------------------------------------------------------------------ diagram
def architecture():
    from matplotlib.patches import FancyBboxPatch
    plt = set_style()
    ss, ps, sp = kv("sample_stats.csv"), kv("population_stats.csv"), t("split_sizes.csv").set_index("part")
    prof = t("cluster_profiles.csv")
    cal = t("hurdle_calibration.csv").set_index("feature_set")
    k = int(t("k_selection.csv")["chosen_k"].iloc[0])
    fig, ax = plt.subplots(figsize=(9, 11.5))
    ax.set_xlim(0, 10)
    ax.axis("off")
    styles = {"data": ("#eef4fc", SERIES[0]), "model": ("#fdf0ea", SERIES[1]), "eval": ("#e8f6f0", SERIES[2])}

    LINE = 0.27  # body line height in data units (8.4 pt text, linespacing 1.3, this figure size)

    def box(x, top, w, title, body, kind, pad_lines=0):
        """Box whose height fits its text; returns (x, top, bottom)."""
        n = (body.count("\n") + 1 if body else 0) + pad_lines
        h = 0.62 + n * LINE + 0.18
        fill, edge = styles[kind]
        ax.add_patch(FancyBboxPatch((x - w / 2, top - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fill, edgecolor=edge, linewidth=1.4))
        ax.text(x, top - 0.18, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=TEXT)
        if body:
            ax.text(x, top - 0.58, body, ha="center", va="top", fontsize=8.4, color=TEXT_2, linespacing=1.3)
        return x, top, top - h

    def arrow(a, b, xa=None, xb=None):
        ax.annotate("", xy=(b[0] if xb is None else xb, b[1] + 0.02),
                    xytext=(a[0] if xa is None else xa, a[2] - 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=TEXT_2, lw=1.2, shrinkA=0, shrinkB=0))

    GAP = 0.42
    raw = box(5, 15.0, 7.4, "Raw events", f"2019-Oct.csv (REES46): view / cart / purchase events\n"
              f"{int(ss['raw_rows']):,} events, {int(ss['raw_users']):,} users", "data")
    smp = box(5, raw[2] - GAP, 7.4, "User sampling", f"chunked read; keep users with user_id % {SAMPLE_MOD} == 0\n"
              f"{int(ss['sample_rows']):,} events, {int(ss['sample_users']):,} users", "data")
    top = smp[2] - GAP
    obs = box(2.65, top, 4.5, "Observation window", "Oct 1-24  ->  features\n"
              f"population: {int(ps['population_users']):,} users\n(>= 1 event in the window)", "data")
    prd = box(7.45, top, 4.3, "Prediction window", "Oct 25-31  ->  targets\nwill_purchase (0/1)\n"
              "future_spend (sum of purchase prices)", "data")
    top = min(obs[2], prd[2]) - GAP
    fea = box(2.65, top, 4.5, "RFM + behavioral features", "RFM: recency, has_purchased,\nfrequency, monetary\n"
              "Behavioral: sessions, views, carts, session\nduration, ratios, days active, categories", "data")
    spl = box(7.45, top, 4.3, "Split (seed 42)", f"train {int(sp.loc['train (all)', 'users']):,} / "
              f"test {int(sp.loc['test', 'users']):,} (80/20,\nstratified on will_purchase);\n"
              f"validation = {int(sp.loc['validation (part of train)', 'users']):,} train users\n"
              "(model choice + F1 thresholds)", "data")
    km = box(5, min(fea[2], spl[2]) - GAP, 9.3, f"K-means segmentation (K={k}, fit on train)",
             "log1p + StandardScaler;  segments: " + ", ".join(prof["segment"]) +
             "\nfeature sets: (a) RFM  |  (b) RFM + behavioral  |  (c) RFM + behavioral + segment", "model")
    top = km[2] - GAP
    sgl = box(2.6, top, 4.7, "Single-stage models", "classification (will_purchase):\nLogistic Regression, "
              "Decision Tree, Random Forest\n(class_weight = balanced)\nregression (future_spend):\n"
              "Linear Regression, Decision Tree, Random Forest", "model")
    hrd = box(7.5, top, 4.7, "Hurdle model", "Stage 1: P(buy) from the classifier with the\nhighest validation "
              "ROC-AUC, prior-corrected\nx\nStage 2: E[spend | buy], Random Forest\ntrained on "
              f"{int(cal['stage2_train_buyers'].iloc[0]):,} train buyers only", "model")
    ev_top = min(sgl[2], hrd[2]) - GAP
    ev = box(5, ev_top, 9.3, "Evaluation (held-out test set)", "", "eval", pad_lines=6)
    for x, title, body in [(1.53, "Metrics", "ROC-AUC, PR-AUC,\nF1 @0.5 and @tuned;\nMAE, RMSE, R²"),
                           (3.84, "Bootstrap CIs", f"{N_BOOT} paired resamples\nof test users,\n95% percentile CI"),
                           (6.16, "Leakage audit", "features rebuilt on\nOct 1-31 (deliberately\nflawed) vs honest"),
                           (8.47, "SHAP", "TreeExplainer on the\nbest Random Forest,\noverall + per segment")]:
        box(x, ev_top - 0.62, 2.15, title, body, "eval")
    arrow(raw, smp)
    arrow(smp, obs, xa=3.6)
    arrow(smp, prd, xa=6.4)
    arrow(obs, fea)
    arrow(prd, spl)
    arrow(fea, km)
    arrow(spl, km, xb=spl[0])
    arrow(km, sgl, xa=3.4)
    arrow(km, hrd, xa=6.6)
    arrow(sgl, ev, xb=sgl[0])
    arrow(hrd, ev, xb=hrd[0])
    ax.set_ylim(ev[2] - 0.15, 15.35)
    ax.set_title("Pipeline: customer spending behaviour (REES46, October 2019)", fontsize=12.5, pad=4)
    fig.savefig(RF_FIG / "architecture.png", facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figures list
def figures_list():
    k = int(t("k_selection.csv")["chosen_k"].iloc[0])
    val = t("classifier_validation.csv")
    best_fs = val[val["model"] == "RandomForest"].sort_values("val_roc_auc", ascending=False).iloc[0]["feature_set"]
    segs = list(pd.read_parquet(CLUSTERS)["segment"].cat.categories)
    caps = [
        ("architecture.png", "Block diagram of the pipeline: raw events, sampling, windows, features, split, K-means "
                             "segmentation, single-stage and hurdle models, evaluation."),
        ("elbow_plot.png", "K-means inertia (within-cluster sum of squares) for K = 2..8 on train users; the "
                           "detected elbow is marked."),
        ("silhouette_plot.png", "Silhouette score for K = 2..8 on a 20,000-user train sample; the maximum is marked."),
        ("cluster_profiles.png", f"Mean of eight Oct 1-24 features per segment (K={k}), one panel per feature."),
        ("roc_curves.png", "Test-set ROC curves for will_purchase, one panel per classifier, one line per feature "
                           "set, with ROC-AUC in the legend."),
        ("actual_vs_predicted.png", f"Actual vs predicted Oct 25-31 spend for test users (symlog axes), feature set "
                                    f"{FS_LABEL[best_fs]}: Linear Regression, Random Forest and hurdle model."),
        ("rf_feature_importance.png", f"Impurity-based feature importance of the Random Forest classifier, "
                                      f"{FS_LABEL[best_fs]}."),
        ("shap_summary.png", f"SHAP beeswarm for the Random Forest classifier ({FS_LABEL[best_fs]}) on 2,000 random "
                             "test users; colour = feature value."),
    ] + [(f"shap_summary_segment_{slug(s)}.png",
          f"SHAP beeswarm for the same model on up to 1,000 test users in segment '{s}'.") for s in segs]
    for f, _ in caps:
        assert (RF_FIG / f).exists(), f"missing figure {f}"
    L = ["# Figures", "", "All figures are PNG at 300 dpi in `outputs/report/figures/` (copies of `outputs/figures/` "
         "plus the architecture diagram).", "", "| File | Caption |", "|---|---|"]
    L += [f"| {f} | {c} |" for f, c in caps]
    (REPORT / "figures_list.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ methods facts
FEATURE_DEFS = {
    "recency_days": f"days from the user's last purchase to 2019-10-25 00:00 UTC; {RECENCY_CAP_DAYS:.0f} if no purchase",
    "has_purchased": "1 if the user made >= 1 purchase in the window, else 0",
    "frequency": "number of purchase events",
    "monetary": "sum of price over purchase events",
    "n_sessions": "number of distinct user_session ids",
    "n_views": "number of view events",
    "n_carts": "number of cart (add-to-cart) events",
    "n_remove_from_cart": "number of remove_from_cart events (DROPPED, see below)",
    "avg_session_duration": "mean over the user's sessions of (last event time - first event time), in minutes",
    "cart_to_view_ratio": "n_carts / n_views (0 if n_views = 0)",
    "purchase_to_cart_ratio": "frequency / n_carts (0 if n_carts = 0)",
    "days_active": "number of distinct calendar days (UTC) with >= 1 event",
    "n_categories": "number of distinct category_id values among view events",
    "days_since_last_event": "days from the user's last event of any type to 2019-10-25 00:00 UTC",
}


def params_text(model) -> str:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    p = est.get_params()
    keep = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "criterion",
            "class_weight", "bootstrap", "C", "penalty", "l1_ratio", "solver", "max_iter", "fit_intercept",
            "random_state"]
    s = ", ".join(f"{k}={p[k]!r}" for k in keep if k in p and p[k] != "deprecated")
    return ("StandardScaler -> " if hasattr(model, "steps") else "") + f"{type(est).__name__}({s})"


def methods_facts():
    ss, ps, sp = kv("sample_stats.csv"), kv("population_stats.csv"), t("split_sizes.csv").set_index("part")
    cal = t("hurdle_calibration.csv").set_index("feature_set")
    ksel = t("k_selection.csv")
    df = pd.read_parquet(FEATURES)
    fsets = feature_sets(df.assign(**{"seg_placeholder": 0.0}))
    size = RAW_CSV.stat().st_size if RAW_CSV.exists() else None
    L = ["# Methods facts", "", "Generated by `src/07_report.py` from code constants and pipeline outputs.", ""]
    w = L.append
    w("## Data")
    w(f"- File: `data/raw/2019-Oct.csv` from Kaggle dataset `mkechinov/ecommerce-behavior-data-from-multi-category-store` "
      f"(only the October file). Size: {size:,} bytes ({size / 1e9:.2f} GB)." if size else "- File: not found on disk.")
    w(f"- Raw events: {int(ss['raw_rows']):,}; raw users: {int(ss['raw_users']):,}. Event counts in the 10% user "
      f"sample: view {int(ss['events_view']):,}, cart {int(ss['events_cart']):,}, purchase "
      f"{int(ss['events_purchase']):,}, remove_from_cart {int(ss['events_remove_from_cart']):,} (event types were "
      "counted in the sample only, not in the full file).")
    w("- Columns read: event_time, event_type, product_id, category_id, price, user_id, user_session "
      "(category_code and brand not read). Timestamps are UTC.")
    w("")
    w("## Sampling")
    w(f"- pandas read_csv in chunks of 5,000,000 rows; keep users with user_id % {SAMPLE_MOD} == 0.")
    w(f"- Result: {int(ss['sample_rows']):,} events, {int(ss['sample_users']):,} users "
      f"({float(ss['sample_user_share']):.4f} of users). Saved to data/processed/sample.parquet.")
    w("- user_session UUIDs converted to 64-bit integers with pandas.util.hash_array (deterministic).")
    w(f"- Exact duplicate rows kept as-is: {int(ss['exact_duplicate_rows_in_sample']):,} in the sample.")
    w("")
    w("## Windows, population, targets")
    w(f"- Observation (feature) window: [{OBS_START}, {PRED_START}) UTC. Prediction (target) window: "
      f"[{PRED_START}, {PRED_END}) UTC. Reference time for recency and days_since_last_event: {PRED_START} UTC.")
    w(f"- Population: users with >= 1 event in the observation window: {int(ps['population_users']):,}.")
    w("- will_purchase = 1 if the user has >= 1 purchase event in the prediction window; future_spend = sum of "
      "price over those purchase events (0 otherwise).")
    w("- Assertion in code: every feature event is before the reference time (no target-window data in features).")
    w("")
    w("## Features (observation window only)")
    for f in RFM + BEHAVIORAL:
        w(f"- `{f}` ({'RFM' if f in RFM else 'behavioral'}): {FEATURE_DEFS[f]}.")
    w(f"- remove_from_cart: the sample ({int(ss['sample_users']):,} users) contains "
      f"{int(ss['events_remove_from_cart'])} remove_from_cart events, so n_remove_from_cart is 0 for every user; "
      "columns with a single value on train are dropped automatically (common.feature_sets), so it is in no model.")
    w(f"- Recency cap: {RECENCY_CAP_DAYS:.0f} days for users without a purchase (observation window is 24 days), "
      "together with the has_purchased flag.")
    w(f"- Feature sets: (a) {len(fsets['a_rfm'])} features {fsets['a_rfm']}; (b) {len(fsets['b_rfm_behav'])} features; "
      "(c) = (b) + one-hot segment indicators (one column per K-means segment).")
    w("")
    w("## Split")
    w(f"- Outer split: train {int(sp.loc['train (all)', 'users']):,} / test {int(sp.loc['test', 'users']):,} "
      f"(test_size={TEST_SIZE}), stratified on will_purchase, random_state={SEED}.")
    w(f"- Inner validation: {VAL_SIZE:.0%} of train, stratified, random_state={SEED}: inner train "
      f"{int(sp.loc['inner train (model choice / thresholds)', 'users']):,}, validation "
      f"{int(sp.loc['validation (part of train)', 'users']):,}. Used only for choosing the hurdle stage-1 classifier, "
      "the Random Forest explained by SHAP, and F1 thresholds.")
    w("- Scalers are fit on train only (inside sklearn Pipelines); K-means scaler and centroids are fit on train only.")
    w("")
    w("## Segmentation")
    w("- Inputs: the (b) features, log1p transform, StandardScaler (fit on train).")
    w(f"- KMeans(n_init=10, random_state={SEED}); K tried 2..8; silhouette on a 20,000-user train sample "
      f"(random_state={SEED}); elbow = K with the largest distance below the chord of normalised inertia.")
    w(f"- Decision rule: use K if elbow and silhouette agree, else K=3. Result: {ksel['decision'].iloc[0]}.")
    w(f"- Naming: rule in outputs/report/cluster_naming.md (buyer if >= {BUYER_SHARE_MIN} of train users purchased in "
      "Oct 1-24; non-buyers split by mean views and sessions vs train means). Target-window data not used.")
    w("- K sensitivity: feature set (c) and segment profiles rerun with K=2 and K=4 (outputs/tables/k_sensitivity_*.csv).")
    w("")
    w("## Models and hyperparameters (scikit-learn; fixed, not tuned)")
    for n_, mk in CLF.items():
        w(f"- Classifier {n_}: {params_text(mk())}")
    for n_, mk in REG.items():
        w(f"- Regressor {n_}: {params_text(mk())}")
    w(f"- Hurdle stage 2: {params_text(STAGE2())}, trained on {int(cal['stage2_train_buyers'].iloc[0]):,} train users "
      "with future_spend > 0.")
    w("")
    w("## Hurdle model")
    w("- predicted_spend = P(purchase) x E[spend | purchase].")
    w("- Stage 1 classifier per feature set = highest validation ROC-AUC: " +
      "; ".join(f"{FS_LABEL[fs]}: {cal.loc[fs, 'stage1_classifier']}" for fs in cal.index) + ".")
    n1, n0 = int(cal["train_positives_n1"].iloc[0]), int(cal["train_negatives_n0"].iloc[0])
    w(f"- Probability correction (undo class_weight='balanced'): odds = p/(1-p) x n1/n0, p_corrected = odds/(1+odds), "
      f"with train positives n1 = {n1:,} and negatives n0 = {n0:,} (n1/n0 = {n1 / n0:.6f}).")
    w("")
    w("## Evaluation")
    w("- Classification metrics on test: accuracy, precision, recall, F1 at threshold 0.5 and at the tuned threshold, "
      "ROC-AUC, PR-AUC (average precision).")
    w("- Threshold tuning: for each classifier x feature set, the threshold maximising F1 on the validation split "
      "(model fit on inner train; sklearn precision_recall_curve, predict positive if p >= threshold); applied "
      "unchanged to the test predictions of the model fit on all of train.")
    w("- Regression metrics on test: MAE, RMSE, R², on all test users and on test buyers only.")
    w(f"- Bootstrap: {N_BOOT} paired resamples of the test users with replacement (numpy default_rng({SEED})); "
      "95% CI = 2.5th and 97.5th percentiles; buyers-only intervals use the buyers inside each resample.")
    w("- Leakage audit (deliberately flawed): features rebuilt from [2019-10-01, 2019-11-01) with reference time "
      "2019-11-01; same users, targets, split, models; K-means refit on those features with the same K.")
    w("- SHAP: shap.TreeExplainer (default settings) on the Random Forest classifier with the highest validation "
      f"ROC-AUC; 2,000 random test users (random_state={SEED}); per segment up to 1,000 test users.")
    w("")
    w("## Software and hardware")
    w(f"- Python {sys.version.split()[0]} on {platform.platform()}; {os.cpu_count()} logical CPUs; CPU only.")
    w("- " + ", ".join(f"{p} {version(p)}" for p in ["numpy", "pandas", "scikit-learn", "shap", "matplotlib",
                                                    "pyarrow", "joblib", "kagglehub"]))
    w("")
    w("## Runtime")
    rt_path = TABLES / "runtime.csv"
    if rt_path.exists():
        rt = pd.read_csv(rt_path)
        w("| Step | Seconds | Note |")
        w("|---|---|---|")
        for _, r in rt.iterrows():
            w(f"| {r['step']} | {'' if pd.isna(r['seconds']) else f'{r.seconds:.0f}'} | "
              f"{'' if pd.isna(r['note']) else r['note']} |")
    else:
        w("- runtime.csv not written yet (run `python run_all.py`).")
    (REPORT / "methods_facts.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ worked example
def worked_example():
    fs = "c_rfm_behav_cluster"
    pr = pd.read_parquet(PREDICTIONS)
    feats = pd.read_parquet(FEATURES)
    cal = t("hurdle_calibration.csv").set_index("feature_set")
    clf_name = cal.loc[fs, "stage1_classifier"]
    n1, n0 = int(cal.loc[fs, "train_positives_n1"]), int(cal.loc[fs, "train_negatives_n0"])
    picks = [("User A", "Recent buyers", True, "segment 'Recent buyers' and bought in Oct 25-31"),
             ("User B", "Engaged browsers", False, "segment 'Engaged browsers' and did not buy in Oct 25-31"),
             ("User C", "Casual visitors", None, "segment 'Casual visitors'")]
    L = ["# Worked example: hurdle prediction for three real test users", "",
         f"Model: hurdle on feature set {FS_LABEL[fs]}. Stage 1 = {MODEL_LABEL[clf_name]} classifier (highest "
         f"validation ROC-AUC), stage 2 = Random Forest regressor trained on train buyers only.",
         "Selection rule (deterministic): within each group, the test user whose hurdle prediction is closest to the "
         "group's median hurdle prediction (ties: lowest user_id). Users are identified only as User A/B/C.",
         "Values are computed at full precision and shown rounded.", ""]
    shown = RFM + [c for c in BEHAVIORAL if c != "n_remove_from_cart"]
    chosen = {}
    for name, segname, bought, desc in picks:
        g = pr[pr["segment"] == segname]
        if bought is True:
            g = g[g["future_spend"] > 0]
        elif bought is False:
            g = g[g["future_spend"] == 0]
        h = g[f"hurdle|{fs}"]
        d = (h - h.median()).abs()
        uid = d[d == d.min()].index.min()
        chosen[name] = (uid, desc, len(g), float(h.median()))
    # positions in the test set (row order of test_predictions.parquet), used by app.py; no user_id is stored
    pd.DataFrame([{"label": n_, "test_index": int(pr.index.get_loc(u)), "group": d_}
                  for n_, (u, d_, *_) in chosen.items()]).to_csv(WORKED_EXAMPLE_USERS, index=False)
    L.append("## Feature values (Oct 1-24)")
    L.append("")
    L.append("| Feature | " + " | ".join(chosen) + " |")
    L.append("|---|" + "---|" * len(chosen))
    for f in shown:
        vals = []
        for uid, *_ in chosen.values():
            v = feats.loc[uid, f]
            vals.append(f"{v:,.0f}" if f in ("has_purchased", "frequency", "n_sessions", "n_views", "n_carts",
                                            "days_active", "n_categories") else f"{v:,.3f}")
        L.append(f"| {label(f)} | " + " | ".join(vals) + " |")
    L.append("| Segment (K-means) | " + " | ".join(pr.loc[u, "segment"] for u, *_ in chosen.values()) + " |")
    L.append("")
    for name, (uid, desc, n_group, med) in chosen.items():
        p_raw = float(pr.loc[uid, f"clf|{fs}|{clf_name}"])
        odds_raw = p_raw / (1 - p_raw)
        odds = odds_raw * n1 / n0
        p = odds / (1 + odds)
        e = float(pr.loc[uid, f"stage2|{fs}"])
        hurdle = p * e
        assert np.isclose(p, pr.loc[uid, f"pcal|{fs}"]) and np.isclose(hurdle, pr.loc[uid, f"hurdle|{fs}"])
        actual = float(pr.loc[uid, "future_spend"])
        rf_single = float(pr.loc[uid, f"reg|{fs}|RandomForest"])
        L += [f"## {name}: {desc}", "",
              f"- Group size: {n_group:,} test users; group median hurdle prediction {med:,.4f}.",
              f"- Stage 1 raw probability ({MODEL_LABEL[clf_name]}, balanced class weights): p_raw = {p_raw:.6f}",
              f"- Raw odds: p_raw / (1 - p_raw) = {p_raw:.6f} / {1 - p_raw:.6f} = {odds_raw:.6f}",
              f"- Prior correction: odds = {odds_raw:.6f} x n1/n0 = {odds_raw:.6f} x {n1:,}/{n0:,} = {odds:.6f}",
              f"- Corrected probability: p = odds / (1 + odds) = {odds:.6f} / {1 + odds:.6f} = {p:.6f}",
              f"- Stage 2 estimate: E[spend | buy] = {e:,.4f}",
              f"- Hurdle prediction: p x E = {p:.6f} x {e:,.4f} = {hurdle:,.4f}",
              f"- Actual spend Oct 25-31: {actual:,.2f}",
              f"- Error (prediction - actual): {hurdle - actual:+,.4f}",
              f"- For comparison, single-stage Random Forest prediction: {rf_single:,.4f}", ""]
    (REPORT / "worked_example.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    if "--methods-only" in sys.argv:
        methods_facts()
        print("methods_facts.md refreshed")
        return
    RT.mkdir(parents=True, exist_ok=True)
    if RF_FIG.exists():
        shutil.rmtree(RF_FIG)
    RF_FIG.mkdir(parents=True)
    for f in FIGURES.glob("*.png"):
        shutil.copy2(f, RF_FIG / f.name)
    t1(); t2(); t3(); t4(); t5(); t6()
    architecture()
    figures_list()
    methods_facts()
    worked_example()
    print("report package written to", REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
