"""Shared paths, constants and helpers for the pipeline."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "2019-Oct.csv"
PROCESSED = ROOT / "data" / "processed"
SAMPLE = PROCESSED / "sample.parquet"
FEATURES = PROCESSED / "features.parquet"            # honest: obs-window features + targets + split
FEATURES_LEAKY = PROCESSED / "features_leaky.parquet"  # deliberately flawed: full-month features
CLUSTERS = PROCESSED / "clusters.parquet"
PREDICTIONS = PROCESSED / "test_predictions.parquet"
MODELS = ROOT / "models"                               # final fitted models (joblib) + manifest.json
SEGMENTATION_MODEL = MODELS / "segmentation.joblib"   # scaler + K-means + segment names (written by 03)
WORKED_EXAMPLE_USERS = PROCESSED / "worked_example_users.csv"  # test-set positions of Users A/B/C (07)
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
REPORT = ROOT / "outputs" / "report"

for _d in (PROCESSED, MODELS, TABLES, FIGURES, REPORT):
    _d.mkdir(parents=True, exist_ok=True)

# Windows (event_time is UTC in the raw data; end bounds are exclusive)
OBS_START = pd.Timestamp("2019-10-01 00:00:00")
PRED_START = pd.Timestamp("2019-10-25 00:00:00")   # also the reference date for recency
PRED_END = pd.Timestamp("2019-11-01 00:00:00")

SAMPLE_MOD = 10  # keep users with user_id % SAMPLE_MOD == 0 (~10% of users)
RECENCY_CAP_DAYS = 60.0  # recency for users with no purchase in the feature window

RFM = ["recency_days", "has_purchased", "frequency", "monetary"]
BEHAVIORAL = [
    "n_sessions", "n_views", "n_carts", "n_remove_from_cart",
    "avg_session_duration", "cart_to_view_ratio", "purchase_to_cart_ratio",
    "days_active", "n_categories", "days_since_last_event",
]
TARGETS = ["will_purchase", "future_spend"]

FEATURE_LABELS = {
    "recency_days": "Recency (days since last purchase)",
    "has_purchased": "Has purchased (0/1)",
    "frequency": "Frequency (purchases)",
    "monetary": "Monetary (total purchase value)",
    "n_sessions": "Sessions",
    "n_views": "Product views",
    "n_carts": "Add-to-cart events",
    "n_remove_from_cart": "Remove-from-cart events",
    "avg_session_duration": "Avg session duration (min)",
    "cart_to_view_ratio": "Cart-to-view ratio",
    "purchase_to_cart_ratio": "Purchase-to-cart ratio",
    "days_active": "Days active",
    "n_categories": "Distinct categories viewed",
    "days_since_last_event": "Days since last event",
}
FS_LABEL = {"a_rfm": "(a) RFM", "b_rfm_behav": "(b) RFM + behavioral",
            "c_rfm_behav_cluster": "(c) RFM + behavioral + segment"}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def label(feature: str, segments=None) -> str:
    """Human-readable feature name; segment dummies become 'Segment: <name>'."""
    if feature.startswith("seg_"):
        for s in (segments or []):
            if f"seg_{slug(s)}" == feature:
                return f"Segment: {s}"
        return "Segment: " + feature[4:].replace("_", " ")
    return FEATURE_LABELS.get(feature, feature)


def feature_sets(df: pd.DataFrame) -> dict:
    """Ablation feature sets. Constant columns (zero variance on train) are dropped."""
    train = df[df["split"] == "train"]
    keep = lambda cols: [c for c in cols if c in df and train[c].nunique() > 1]
    seg = [c for c in df.columns if c.startswith("seg_")]
    return {
        "a_rfm": keep(RFM),
        "b_rfm_behav": keep(RFM + BEHAVIORAL),
        "c_rfm_behav_cluster": keep(RFM + BEHAVIORAL) + seg,
    }


# ---------------- clustering (shared by 03, the leakage audit in 04 and 06) ----------------
BUYER_SHARE_MIN = 0.5  # a cluster is a "buyer" cluster if >= 50% of its train users purchased Oct 1-24


def cluster_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    """log1p then (caller) standard-scale; all clustering inputs are non-negative."""
    return np.log1p(df[cols].to_numpy(dtype=float))


def name_segments(df: pd.DataFrame, cid: pd.Series, train: pd.Series):
    """Name clusters from their obs-window profile on TRAIN users (no target-window data is used).

    Rule:
      1. buyer cluster: share of users with a purchase in Oct 1-24 >= BUYER_SHARE_MIN.
         One buyer cluster -> "Recent buyers"; several -> "Recent buyers, <low|mid|high> spend" by mean monetary.
      2. non-buyer cluster: "Engaged browsers" if its mean product views AND mean sessions are both >= the
         train-population means, otherwise "Casual visitors". Duplicate names get " 1", " 2" by mean views.
    Returns (id->name dict, display order low->high tier, profile DataFrame used by the rule, overall means).
    """
    prof = df.loc[train, ["has_purchased", "monetary", "n_views", "n_sessions"]].groupby(cid[train]).mean()
    overall = df.loc[train, ["n_views", "n_sessions"]].mean()
    buyers = prof[prof["has_purchased"] >= BUYER_SHARE_MIN].sort_values("monetary")
    names = {}
    if len(buyers) == 1:
        names[buyers.index[0]] = "Recent buyers"
    elif len(buyers) > 1:
        tags = {2: ["low", "high"], 3: ["low", "mid", "high"]}.get(
            len(buyers), [f"rank {i + 1}" for i in range(len(buyers))])
        for c, tag in zip(buyers.index, tags):
            names[c] = f"Recent buyers, {tag} spend"
    non = prof.drop(index=buyers.index).sort_values("n_views")
    for c, row in non.iterrows():
        engaged = row["n_views"] >= overall["n_views"] and row["n_sessions"] >= overall["n_sessions"]
        names[c] = "Engaged browsers" if engaged else "Casual visitors"
    for base in set(names.values()):
        dup = [c for c in non.index if names.get(c) == base]
        if len(dup) > 1:
            for i, c in enumerate(dup, 1):
                names[c] = f"{base} {i}"
    tier = prof.assign(is_buyer=prof["has_purchased"] >= BUYER_SHARE_MIN,
                       key=np.where(prof["has_purchased"] >= BUYER_SHARE_MIN, prof["monetary"], prof["n_views"]))
    order = [names[c] for c in tier.sort_values(["is_buyer", "key"]).index]
    return names, order, prof, overall


def fit_segments(df: pd.DataFrame, cols: list, k: int):
    """Fit scaler + K-means on TRAIN users only, assign every user, name segments by `name_segments`.

    Returns (cluster_id Series, segment Series [ordered categorical, low->high tier], scaler, kmeans, id->name).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    train = df["split"] == "train"
    X = cluster_matrix(df, cols)
    scaler = StandardScaler().fit(X[train.to_numpy()])
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs[train.to_numpy()])
    cid = pd.Series(km.predict(Xs), index=df.index, name="cluster")
    mapping, order, _, _ = name_segments(df, cid, train)
    seg = pd.Series(pd.Categorical(cid.map(mapping), categories=order, ordered=True), index=df.index, name="segment")
    return cid, seg, scaler, km, mapping


def add_segment_dummies(df: pd.DataFrame, segment: pd.Series) -> pd.DataFrame:
    """One-hot segment columns seg_<slug>, ordered highest tier first."""
    cats = list(segment.cat.categories) if isinstance(segment.dtype, pd.CategoricalDtype) \
        else sorted(segment.unique())
    d = pd.DataFrame({f"seg_{slug(s)}": (segment == s).astype(float) for s in reversed(cats)}, index=segment.index)
    return df.drop(columns=[c for c in df.columns if c.startswith("seg_")]).join(d)


# ---------------- plotting (light-mode reference palette, validated) ----------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # categorical slots 1-3 (all-pairs safe)
ORDINAL_RAMP = ["#86b6ef", "#2a78d6", "#104281"]      # ordinal blue ramp, light -> dark (validated --ordinal)
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def tier_colors(n: int) -> list:
    return ORDINAL_RAMP if n == 3 else [ORDINAL_RAMP[round(i * 2 / max(n - 1, 1))] for i in range(n)]


def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": TEXT_2,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlecolor": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 10.5,
        "legend.frameon": False, "legend.fontsize": 9.5, "lines.linewidth": 2,
        "font.family": "sans-serif",
    })
    return plt
