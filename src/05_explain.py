"""Step 5: explainability + figures.

- Best Random Forest = RF classifier (will_purchase) whose feature set has the highest VALIDATION ROC-AUC
  (classifier_validation.csv, inner split of train), so the test set is not used to pick it.
- SHAP TreeExplainer on 2,000 random test users; per-segment SHAP on up to 1,000 test users per K-means segment.
- Figures: ROC curves, actual-vs-predicted, RF feature importance, SHAP summary (overall + per segment).
"""
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, roc_curve

from common import (CLUSTERS, FEATURES, FIGURES, FS_LABEL, GRID, MODELS, PREDICTIONS, SEED, SERIES, TABLES, TEXT_2,
                    add_segment_dummies, feature_sets, label, set_style, slug)

N_SHAP = 2000
N_SHAP_SEGMENT = 1000


def positive_class_shap(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return sv[1]
    return sv[:, :, 1] if sv.ndim == 3 else sv


def main():
    plt = set_style()
    seg = pd.read_parquet(CLUSTERS)["segment"]
    segments = list(seg.cat.categories)
    df = add_segment_dummies(pd.read_parquet(FEATURES), seg)
    fsets = feature_sets(df)
    preds = pd.read_parquet(PREDICTIONS)
    val = pd.read_csv(TABLES / "classifier_validation.csv")
    rf_val = val[val["model"] == "RandomForest"].sort_values("val_roc_auc", ascending=False)
    best_fs = rf_val.iloc[0]["feature_set"]
    cols = fsets[best_fs]
    names = [label(c, segments) for c in cols]
    print("best RF classifier feature set (validation ROC-AUC):", best_fs)
    print(rf_val.to_string(index=False))
    rf = joblib.load(MODELS / f"rf_clf_{best_fs}.joblib")
    te = df.loc[preds.index]
    for old in FIGURES.glob("shap_summary_segment_*.png"):
        old.unlink()

    # ---------------- ROC curves ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, model in zip(axes, ["LogisticRegression", "DecisionTree", "RandomForest"]):
        for color, fs in zip(SERIES, fsets):
            p = preds[f"clf|{fs}|{model}"]
            fpr, tpr, _ = roc_curve(preds["will_purchase"], p)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{FS_LABEL[fs]}  AUC={roc_auc_score(preds['will_purchase'], p):.3f}")
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        ax.set_title({"LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree",
                      "RandomForest": "Random Forest"}[model])
        ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", fontsize=8.5)
    axes[0].set_ylabel("True positive rate")
    fig.suptitle("ROC curves for will_purchase (Oct 25-31), test set", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curves.png")
    plt.close(fig)

    # ---------------- actual vs predicted (future_spend) ----------------
    y = preds["future_spend"].to_numpy()
    panels = [(f"reg|{best_fs}|LinearRegression", "Linear Regression (single-stage)"),
              (f"reg|{best_fs}|RandomForest", "Random Forest (single-stage)"),
              (f"hurdle|{best_fs}", "Hurdle: P(buy) x E[spend | buy]")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True, sharey=True)
    lim = max(y.max(), max(preds[k].max() for k, _ in panels)) * 1.2
    for ax, (key, title) in zip(axes, panels):
        p = np.clip(preds[key].to_numpy(), 0, None)
        ax.scatter(y[y == 0], p[y == 0], s=4, alpha=0.15, color=SERIES[0], linewidths=0,
                   label="did not buy Oct 25-31")
        ax.scatter(y[y > 0], p[y > 0], s=6, alpha=0.5, color=SERIES[1], linewidths=0, label="bought Oct 25-31")
        ax.plot([0, lim], [0, lim], color=TEXT_2, lw=1)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Actual spend, Oct 25-31")
    axes[0].set_ylabel("Predicted spend (negatives clipped to 0)")
    axes[0].legend(loc="upper left", markerscale=3)
    fig.suptitle(f"Actual vs predicted spend, test set, feature set {FS_LABEL[best_fs]} (symlog axes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "actual_vs_predicted.png")
    plt.close(fig)

    # ---------------- RF feature importance ----------------
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values()
    imp.sort_values(ascending=False).rename("impurity_importance").to_frame() \
        .assign(label=lambda d: [label(c, segments) for c in d.index]) \
        .to_csv(TABLES / "rf_feature_importance.csv", index_label="feature")
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(imp) + 1.3))
    ax.barh([label(c, segments) for c in imp.index], imp.values, color=SERIES[0], height=0.62)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Random Forest classifier feature importance\n{FS_LABEL[best_fs]}")
    fig.savefig(FIGURES / "rf_feature_importance.png")
    plt.close(fig)

    # ---------------- SHAP (overall) ----------------
    explainer = shap.TreeExplainer(rf)
    X_all = te[cols].sample(n=min(N_SHAP, len(te)), random_state=SEED)
    sv = positive_class_shap(explainer, X_all)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    mean_abs.rename("mean_abs_shap").to_frame().assign(label=lambda d: [label(c, segments) for c in d.index]) \
        .to_csv(TABLES / "shap_importance.csv", index_label="feature")
    print("mean |SHAP| (overall):\n", mean_abs.round(5).to_string())
    shap.summary_plot(sv, X_all, feature_names=names, show=False, max_display=len(cols), plot_size=(9, 8))
    plt.title(f"SHAP summary, Random Forest classifier {FS_LABEL[best_fs]}\n(n={len(X_all):,} test users)",
              fontsize=11)
    plt.savefig(FIGURES / "shap_summary.png")
    plt.close("all")

    # ---------------- SHAP per K-means segment ----------------
    rows = []
    te_seg = seg.loc[te.index]
    for s in segments:
        Xs = te.loc[te_seg == s, cols]
        Xs = Xs.sample(n=min(N_SHAP_SEGMENT, len(Xs)), random_state=SEED)
        svs = positive_class_shap(explainer, Xs)
        ms = pd.Series(np.abs(svs).mean(axis=0), index=cols).sort_values(ascending=False)
        # seg_* dummies are constant inside a segment; their SHAP only measures the gap to the global
        # baseline, so a second ranking without them is also saved.
        for variant, ranked in [("all_features", ms), ("excluding_seg_dummies", ms[~ms.index.str.startswith("seg_")])]:
            for rank, (f, v) in enumerate(ranked.head(5).items(), 1):
                rows.append({"segment": s, "ranking": variant, "n_shap_rows": len(Xs), "rank": rank, "feature": f,
                             "label": label(f, segments), "mean_abs_shap": v,
                             "mean_shap_signed": float(svs[:, cols.index(f)].mean())})
        shap.summary_plot(svs, Xs, feature_names=names, show=False, max_display=len(cols), plot_size=(9, 8))
        plt.title(f"SHAP summary, segment: {s}\n(n={len(Xs):,} test users)", fontsize=11)
        plt.savefig(FIGURES / f"shap_summary_segment_{slug(s)}.png")
        plt.close("all")
        print(f"segment {s}: top5 = {list(ms.head(5).index)}")
    pd.DataFrame(rows).to_csv(TABLES / "shap_top5_per_segment.csv", index=False)
    print("saved figures and tables")


if __name__ == "__main__":
    main()
