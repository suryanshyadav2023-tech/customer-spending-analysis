"""Model definitions, metrics and validation helpers shared by 04, 06 and the methods-facts report."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             precision_recall_curve, precision_score, r2_score, recall_score, roc_auc_score,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common import SEED

N_BOOT = 500
VAL_SIZE = 0.2   # inner validation share of TRAIN (stratified), used for model choice and threshold tuning
TEST_SIZE = 0.2  # outer test share (set in 02_features.py)
RF_KW = dict(n_estimators=200, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=SEED)
TREE_KW = dict(max_depth=8, min_samples_leaf=50, random_state=SEED)
LOGREG_KW = dict(class_weight="balanced", max_iter=5000)

CLF = {
    "LogisticRegression": lambda: make_pipeline(StandardScaler(), LogisticRegression(**LOGREG_KW)),
    "DecisionTree": lambda: DecisionTreeClassifier(class_weight="balanced", **TREE_KW),
    "RandomForest": lambda: RandomForestClassifier(class_weight="balanced", **RF_KW),
}
REG = {
    "LinearRegression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "DecisionTree": lambda: DecisionTreeRegressor(**TREE_KW),
    "RandomForest": lambda: RandomForestRegressor(**RF_KW),
}
STAGE2 = lambda: RandomForestRegressor(**RF_KW)  # hurdle stage 2 (buyers only)


def clf_metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p)}


def reg_metrics(y, pred) -> dict:
    return {"MAE": mean_absolute_error(y, pred), "RMSE": root_mean_squared_error(y, pred), "R2": r2_score(y, pred)}


def prf_at(y: np.ndarray, p: np.ndarray, thr: float):
    """Fast precision, recall, F1 at a threshold (predict positive if p >= thr)."""
    yhat = p >= thr
    tp = np.sum(yhat & (y == 1))
    fp = np.sum(yhat) - tp
    fn = np.sum(y == 1) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def best_f1_threshold(y: np.ndarray, p: np.ndarray):
    """Threshold maximising F1 on (y, p); ties -> lowest threshold. Returns (threshold, f1)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.clip(prec + rec, 1e-12, None), 0.0)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def prior_correct(p: np.ndarray, n1: int, n0: int) -> np.ndarray:
    """Undo class_weight='balanced': true odds = weighted odds * n1/n0."""
    odds = p / np.clip(1 - p, 1e-12, None) * (n1 / n0)
    return odds / (1 + odds)


def fit_all(df: pd.DataFrame, fsets: dict, label: str):
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    rows, preds, fitted = [], {}, {}
    for fs, cols in fsets.items():
        for task, models, target in [("classification", CLF, "will_purchase"), ("regression", REG, "future_spend")]:
            for name, make in models.items():
                t0 = time.time()
                m = make().fit(tr[cols], tr[target])
                secs = time.time() - t0
                if task == "classification":
                    p = m.predict_proba(te[cols])[:, 1]
                    met = clf_metrics(te[target], p)
                else:
                    p = m.predict(te[cols])
                    met = reg_metrics(te[target], p)
                key = f"{'clf' if task == 'classification' else 'reg'}|{fs}|{name}"
                preds[key], fitted[key] = p, m
                rows.append({"setup": label, "task": task, "feature_set": fs, "model": name,
                             "n_features": len(cols), **met, "fit_seconds": secs})
                print(f"[{label}] {task:14s} {fs:20s} {name:18s} " +
                      "  ".join(f"{k}={v:.4f}" for k, v in met.items()) + f"  ({secs:.1f}s)", flush=True)
    return pd.DataFrame(rows), preds, fitted


def inner_split(df: pd.DataFrame):
    tr = df[df["split"] == "train"]
    itr, iva = train_test_split(tr.index, test_size=VAL_SIZE, stratify=tr["will_purchase"], random_state=SEED)
    return tr, itr, iva


def validate_classifiers(df: pd.DataFrame, fsets: dict) -> pd.DataFrame:
    """Fit each classifier on the inner-train part of TRAIN and score the validation part.

    Used (never the test set) to pick the hurdle's stage-1 classifier and each classifier's F1-optimal threshold.
    """
    tr, itr, iva = inner_split(df)
    yv = tr.loc[iva, "will_purchase"].to_numpy()
    rows = []
    for fs, cols in fsets.items():
        for name, make in CLF.items():
            m = make().fit(tr.loc[itr, cols], tr.loc[itr, "will_purchase"])
            pv = m.predict_proba(tr.loc[iva, cols])[:, 1]
            thr, f1v = best_f1_threshold(yv, pv)
            rows.append({"feature_set": fs, "model": name, "val_roc_auc": roc_auc_score(yv, pv),
                         "val_f1_at_0.5": prf_at(yv, pv, 0.5)[2], "val_best_f1_threshold": thr,
                         "val_f1_at_best_threshold": f1v})
            print(f"[validation] {fs:20s} {name:18s} val_roc_auc={rows[-1]['val_roc_auc']:.4f} "
                  f"best_thr={thr:.4f} val_f1={f1v:.4f}", flush=True)
    return pd.DataFrame(rows)
