"""Step 1: chunked read of 2019-Oct.csv, keep ~10% of users (user_id % 10 == 0).

Output: data/processed/sample.parquet, outputs/tables/sample_stats.csv
"""
import time

import numpy as np
import pandas as pd

from common import RAW_CSV, SAMPLE, SAMPLE_MOD, TABLES

CHUNKSIZE = 5_000_000
USECOLS = ["event_time", "event_type", "product_id", "category_id", "price", "user_id", "user_session"]
DTYPES = {
    "event_type": "category",
    "product_id": "int32",
    "category_id": "int64",
    "price": "float64",
    "user_id": "int32",
    "user_session": "object",
}


def session_to_int(s: pd.Series) -> np.ndarray:
    """Deterministic 64-bit hash of the session UUID (missing sessions -> one 'missing' key)."""
    return pd.util.hash_array(s.fillna("missing").to_numpy(dtype=object)).astype("int64")


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found")
    t0 = time.time()
    total_rows, missing_sessions = 0, 0
    all_users = set()
    parts = []
    reader = pd.read_csv(RAW_CSV, usecols=USECOLS, dtype=DTYPES, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        all_users.update(chunk["user_id"].unique().tolist())
        chunk = chunk[chunk["user_id"] % SAMPLE_MOD == 0].copy()
        missing_sessions += int(chunk["user_session"].isna().sum())
        chunk["event_time"] = pd.to_datetime(chunk["event_time"], format="%Y-%m-%d %H:%M:%S UTC")
        chunk["session_id"] = session_to_int(chunk["user_session"])
        chunk = chunk.drop(columns="user_session")
        parts.append(chunk)
        print(f"chunk {i}: {total_rows:,} rows read, {sum(len(p) for p in parts):,} kept "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df["event_type"] = df["event_type"].astype("category")
    df = df.sort_values(["user_id", "event_time"], kind="stable").reset_index(drop=True)
    df.to_parquet(SAMPLE, index=False)

    n_dupes = int(df.duplicated().sum())
    counts = df["event_type"].value_counts()
    stats = {
        "raw_rows": total_rows,
        "raw_users": len(all_users),
        "sample_rows": len(df),
        "sample_users": int(df["user_id"].nunique()),
        "sample_user_share": df["user_id"].nunique() / len(all_users),
        "min_event_time": df["event_time"].min(),
        "max_event_time": df["event_time"].max(),
        "exact_duplicate_rows_in_sample": n_dupes,
        "missing_session_rows_in_sample": missing_sessions,
        "purchase_rows_with_nonpositive_price": int(((df["event_type"] == "purchase") & (df["price"] <= 0)).sum()),
    }
    for et in ["view", "cart", "remove_from_cart", "purchase"]:
        stats[f"events_{et}"] = int(counts.get(et, 0))
    pd.Series(stats, name="value").to_csv(TABLES / "sample_stats.csv", index_label="stat")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"saved {SAMPLE} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
