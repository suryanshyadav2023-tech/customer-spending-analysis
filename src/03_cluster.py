"""Step 3: K selection (elbow + silhouette, K=2..8), K-means segmentation, cluster profiles and names.

Clustering inputs: obs-window RFM + behavioral features, log1p then StandardScaler.
Scaler and K-means are fit on TRAIN users only; test users are assigned with predict().
Segments are named from their obs-window profile (common.name_segments); the rule and the numbers
behind each name are written to outputs/report/cluster_naming.md.
Outputs: clusters.parquet, k_selection.csv, cluster_profiles.csv, elbow/silhouette/profile figures.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import (BEHAVIORAL, BUYER_SHARE_MIN, CLUSTERS, FEATURES, FIGURES, REPORT, RFM, SEED, SEGMENTATION_MODEL,
                    SERIES, TABLES, TEXT_2, cluster_matrix, feature_sets, fit_segments, label, name_segments,
                    set_style, tier_colors)

K_RANGE = range(2, 9)
DEFAULT_K = 3
SIL_SAMPLE = 20_000


def elbow_k(ks, inertia) -> int:
    """Knee = point with max distance below the chord from first to last (normalised) inertia."""
    x = (np.asarray(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.asarray(inertia) - min(inertia)) / (max(inertia) - min(inertia))
    dist = (1 - x) - y  # chord from (0,1) to (1,0) is y = 1 - x
    return int(ks[int(np.argmax(dist))])


def cluster_profile(df: pd.DataFrame, seg: pd.Series) -> pd.DataFrame:
    """Per-segment means over ALL population users. Target columns are descriptive only."""
    train = df["split"] == "train"
    prof = df[RFM + BEHAVIORAL + ["will_purchase", "future_spend"]].groupby(seg, observed=True).mean()
    prof.insert(0, "n_users", seg.value_counts())
    prof.insert(1, "share", prof["n_users"] / len(df))
    prof.insert(2, "n_train", seg[train].value_counts())
    return prof.rename(columns={"will_purchase": "pred_window_purchase_rate",
                                "future_spend": "pred_window_mean_spend"})


def write_naming_md(df, cid, mapping, k):
    train = df["split"] == "train"
    _, order, rule_prof, overall = name_segments(df, cid, train)
    old_rank = df.loc[train, "monetary"].groupby(cid[train]).mean().sort_values().index.tolist()
    old_names = dict(zip(old_rank, ["Low", "Medium", "High"] if k == 3 else [f"rank{i}" for i in range(k)]))
    extra = df.loc[train, ["frequency", "recency_days", "days_active", "will_purchase"]].groupby(cid[train]).mean()
    rows = []
    for c in sorted(mapping, key=lambda c: order.index(mapping[c])):
        r, e = rule_prof.loc[c], extra.loc[c]
        if r["has_purchased"] >= BUYER_SHARE_MIN:
            why = (f"purchase share {r['has_purchased']:.3f} >= {BUYER_SHARE_MIN} -> buyer cluster; "
                   f"mean recency {e['recency_days']:.2f} days (all purchases fall in Oct 1-24)")
        else:
            why = (f"purchase share {r['has_purchased']:.3f} < {BUYER_SHARE_MIN} -> non-buyer; views "
                   f"{r['n_views']:.2f} {'>=' if r['n_views'] >= overall['n_views'] else '<'} {overall['n_views']:.2f} "
                   f"and sessions {r['n_sessions']:.2f} {'>=' if r['n_sessions'] >= overall['n_sessions'] else '<'} "
                   f"{overall['n_sessions']:.2f}")
        rows.append((mapping[c], old_names[c], int((train & (cid == c)).sum()), r["has_purchased"], r["monetary"],
                     e["frequency"], e["recency_days"], r["n_views"], r["n_sessions"], e["days_active"],
                     e["will_purchase"], why))
    L = ["# Cluster naming", "",
         f"K = {k}. Names are assigned by code (`common.name_segments`) from TRAIN-user means of Oct 1-24 features.",
         "", "## Rule", "",
         f"1. Buyer cluster: share of users with a purchase in Oct 1-24 >= {BUYER_SHARE_MIN}. One buyer cluster -> "
         "\"Recent buyers\"; several -> \"Recent buyers, low/mid/high spend\" by mean monetary.",
         "2. Non-buyer cluster: \"Engaged browsers\" if mean product views AND mean sessions are both >= the "
         f"train-population means (views {overall['n_views']:.2f}, sessions {overall['n_sessions']:.2f}); "
         "otherwise \"Casual visitors\".",
         "3. The Oct 25-31 purchase rate is NOT used by the rule; it is listed only as a descriptive check.",
         "4. Previous names (by mean monetary only) are listed for traceability.", "",
         "## Profile numbers behind each name (train users)", "",
         "| Name | Previous name | Train users | Purchased Oct 1-24 (share) | Mean monetary | Mean frequency | "
         "Mean recency (days) | Mean product views | Mean sessions | Mean days active | "
         "Oct 25-31 purchase rate (not used) | Why |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]:,} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.2f} | {r[7]:.2f} | "
                 f"{r[8]:.2f} | {r[9]:.2f} | {r[10]:.4f} | {r[11]} |")
    (REPORT / "cluster_naming.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    plt = set_style()
    df = pd.read_parquet(FEATURES)
    cols = feature_sets(df)["b_rfm_behav"]
    print("clustering features:", cols)
    train = (df["split"] == "train").to_numpy()
    X = cluster_matrix(df, cols)
    Xtr = StandardScaler().fit(X[train]).transform(X[train])

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xtr)
        sil = silhouette_score(Xtr, km.labels_, sample_size=SIL_SAMPLE, random_state=SEED)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"k={k}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}", flush=True)
    ksel = pd.DataFrame(rows)
    ks = ksel["k"].tolist()
    k_elb = elbow_k(ks, ksel["inertia"].tolist())
    k_sil = int(ksel.loc[ksel["silhouette"].idxmax(), "k"])
    if k_elb == k_sil:
        k, reason = k_elb, "elbow and silhouette agree"
    else:
        k, reason = DEFAULT_K, f"ambiguous (elbow={k_elb}, silhouette={k_sil}) -> default K={DEFAULT_K}"
    print(f"elbow K={k_elb}, best-silhouette K={k_sil} => chosen K={k} ({reason})")
    ksel["elbow_k"] = k_elb
    ksel["best_silhouette_k"] = k_sil
    ksel["chosen_k"] = k
    ksel["decision"] = reason
    ksel.to_csv(TABLES / "k_selection.csv", index=False)

    cid, seg, scaler, km, mapping = fit_segments(df, cols, k)
    pd.DataFrame({"cluster": cid, "segment": seg}).to_parquet(CLUSTERS)
    joblib.dump({"scaler": scaler, "kmeans": km, "cluster_to_segment": {int(c): s for c, s in mapping.items()},
                 "segment_order": list(seg.cat.categories), "input_columns": cols, "transform": "log1p",
                 "k": k}, SEGMENTATION_MODEL)
    print("cluster -> segment:", mapping)
    write_naming_md(df, cid, mapping, k)

    prof = cluster_profile(df, seg)
    prof.to_csv(TABLES / "cluster_profiles.csv", index_label="segment")
    print(prof.T.round(3).to_string())

    # ---- figures ----
    for col, fname, ylabel, mark_k, tag in [
            ("inertia", "elbow_plot.png", "Inertia (within-cluster sum of squares)", k_elb, "detected elbow"),
            ("silhouette", "silhouette_plot.png", "Silhouette score", k_sil, "best silhouette")]:
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        ax.plot(ksel["k"], ksel[col], color=SERIES[0], marker="o", markersize=6)
        y = ksel.loc[ksel["k"] == mark_k, col].item()
        ax.plot([mark_k], [y], marker="o", markersize=10, color=SERIES[1], zorder=3)
        ax.annotate(f"{tag}: K={mark_k}", (mark_k, y), xytext=(10, -4), textcoords="offset points", va="top",
                    color=TEXT_2, fontsize=10)
        ax.set_xlabel("Number of clusters K")
        ax.set_ylabel(ylabel)
        ax.set_title(("Elbow method" if col == "inertia" else f"Silhouette (sample of {SIL_SAMPLE:,} train users)")
                     + f"  |  chosen K={k}")
        ax.set_xticks(list(K_RANGE))
        if col == "inertia":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
        fig.savefig(FIGURES / fname)
        plt.close(fig)

    show = ["monetary", "frequency", "recency_days", "n_sessions", "n_views", "n_carts",
            "days_active", "avg_session_duration"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    colors = tier_colors(len(prof))
    ticks = [s.replace(" ", "\n", 1) for s in prof.index]
    for ax, f in zip(axes.ravel(), show):
        ax.bar(ticks, prof[f], color=colors, width=0.62)
        ax.set_title(label(f), fontsize=10.5)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=9.5)
    fig.suptitle(f"Segment profiles: mean per segment, K={k}, features from Oct 1-24", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "cluster_profiles.png")
    plt.close(fig)
    print("saved figures and tables")


if __name__ == "__main__":
    main()
