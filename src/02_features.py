"""Step 2: observation/prediction windows, RFM + behavioral features, targets, train/test split.

Honest features:  events in [Oct 1, Oct 25)  -> features.parquet
Targets:          events in [Oct 25, Nov 1)
Leaky features:   events in [Oct 1, Nov 1)   -> features_leaky.parquet (leakage audit only)
Population:       users with >= 1 event in the observation window.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (BEHAVIORAL, FEATURES, FEATURES_LEAKY, OBS_START, PRED_END, PRED_START,
                    RECENCY_CAP_DAYS, RFM, SAMPLE, SEED, TABLES)

DAY = 86400.0


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).fillna(0.0)


def build_features(ev: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """Per-user features from the events in `ev` (all assumed to be before `ref`)."""
    assert ev["event_time"].max() < ref, "feature events must be before the reference date"
    by_user = ev.groupby("user_id", sort=True)
    out = pd.DataFrame(index=by_user.size().index)

    # --- RFM ---
    purch = ev[ev["event_type"] == "purchase"]
    pg = purch.groupby("user_id")
    out["frequency"] = pg.size().reindex(out.index, fill_value=0).astype(float)
    out["monetary"] = pg["price"].sum().reindex(out.index, fill_value=0.0)
    last_purchase = pg["event_time"].max().reindex(out.index)
    out["recency_days"] = ((ref - last_purchase).dt.total_seconds() / DAY).fillna(RECENCY_CAP_DAYS)
    out["has_purchased"] = (out["frequency"] > 0).astype(float)

    # --- behavioral ---
    counts = ev.groupby(["user_id", "event_type"], observed=False).size().unstack(fill_value=0)
    for et, col in [("view", "n_views"), ("cart", "n_carts"), ("remove_from_cart", "n_remove_from_cart")]:
        out[col] = (counts[et] if et in counts else pd.Series(0, index=counts.index)).reindex(
            out.index, fill_value=0).astype(float)

    sess = ev.groupby(["user_id", "session_id"])["event_time"].agg(["min", "max"])
    sess["dur_min"] = (sess["max"] - sess["min"]).dt.total_seconds() / 60.0
    sg = sess.groupby(level="user_id")["dur_min"]
    out["n_sessions"] = sg.size().reindex(out.index).astype(float)
    out["avg_session_duration"] = sg.mean().reindex(out.index)

    out["cart_to_view_ratio"] = safe_div(out["n_carts"], out["n_views"])
    out["purchase_to_cart_ratio"] = safe_div(out["frequency"], out["n_carts"])
    out["days_active"] = ev.assign(day=ev["event_time"].dt.floor("D")).groupby("user_id")["day"].nunique() \
        .reindex(out.index).astype(float)
    views = ev[ev["event_type"] == "view"]
    out["n_categories"] = views.groupby("user_id")["category_id"].nunique().reindex(out.index, fill_value=0) \
        .astype(float)
    out["days_since_last_event"] = (ref - by_user["event_time"].max()).dt.total_seconds() / DAY
    return out[RFM + BEHAVIORAL]


def build_targets(ev_pred: pd.DataFrame, users: pd.Index) -> pd.DataFrame:
    purch = ev_pred[ev_pred["event_type"] == "purchase"]
    spend = purch.groupby("user_id")["price"].sum().reindex(users, fill_value=0.0)
    return pd.DataFrame({"will_purchase": (spend.index.isin(purch["user_id"].unique())).astype(int),
                         "future_spend": spend.values}, index=users)


def main():
    ev = pd.read_parquet(SAMPLE)
    obs = ev[(ev["event_time"] >= OBS_START) & (ev["event_time"] < PRED_START)]
    pred = ev[(ev["event_time"] >= PRED_START) & (ev["event_time"] < PRED_END)]
    full = ev[(ev["event_time"] >= OBS_START) & (ev["event_time"] < PRED_END)]
    print(f"events: obs={len(obs):,}  pred={len(pred):,}  full={len(full):,}  outside={len(ev) - len(full):,}")

    feats = build_features(obs, PRED_START)
    users = feats.index
    y = build_targets(pred, users)
    assert np.all(feats.notna().to_numpy()), "unexpected NaN in features"

    train_idx, test_idx = train_test_split(users, test_size=0.2, stratify=y["will_purchase"], random_state=SEED)
    split = pd.Series("train", index=users)
    split.loc[test_idx] = "test"

    honest = feats.join(y).assign(split=split)
    honest.index.name = "user_id"
    honest.to_parquet(FEATURES)

    # Deliberately flawed features for the leakage audit: full month, same users, same targets, same split.
    leaky = build_features(full[full["user_id"].isin(users)], PRED_END).reindex(users)
    assert np.all(leaky.notna().to_numpy())
    leaky = leaky.join(y).assign(split=split)
    leaky.index.name = "user_id"
    leaky.to_parquet(FEATURES_LEAKY)

    buyers = honest["future_spend"] > 0
    stats = {
        "population_users": len(honest),
        "train_users": int((split == "train").sum()),
        "test_users": int((split == "test").sum()),
        "obs_window_events": len(obs),
        "pred_window_events": len(pred),
        "users_with_obs_purchase": int(honest["has_purchased"].sum()),
        "will_purchase_positive": int(honest["will_purchase"].sum()),
        "will_purchase_rate": honest["will_purchase"].mean(),
        "will_purchase_rate_train": honest.loc[split == "train", "will_purchase"].mean(),
        "will_purchase_rate_test": honest.loc[split == "test", "will_purchase"].mean(),
        "future_spend_mean_all": honest["future_spend"].mean(),
        "future_spend_mean_buyers": honest.loc[buyers, "future_spend"].mean(),
        "future_spend_median_buyers": honest.loc[buyers, "future_spend"].median(),
        "future_spend_max": honest["future_spend"].max(),
        "buyers_with_zero_spend": int(((honest["will_purchase"] == 1) & ~buyers).sum()),
        "pred_window_purchasers_not_in_population": int(
            pred.loc[pred["event_type"] == "purchase", "user_id"].drop_duplicates().isin(users).__invert__().sum()),
        "users_with_avg_session_over_24h": int((honest["avg_session_duration"] > 1440).sum()),
        "max_obs_purchases_one_user": int(honest["frequency"].max()),
    }
    pd.Series(stats, name="value").to_csv(TABLES / "population_stats.csv", index_label="stat")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("\nfeature summary (honest):")
    print(honest[RFM + BEHAVIORAL].describe().T.round(3).to_string())
    honest[RFM + BEHAVIORAL].describe().T.to_csv(TABLES / "feature_summary.csv")


if __name__ == "__main__":
    main()
