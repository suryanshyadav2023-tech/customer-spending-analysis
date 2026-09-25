"""Offline demo of the customer-spending pipeline.

Launch:  streamlit run app.py      (or: python -m streamlit run app.py)
Reads only saved outputs: models/ (joblib + manifest.json), data/processed/, outputs/report/, outputs/tables/.
Nothing is retrained. Test users are addressed by their position in the test set, never by user_id.
"""
import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from common import (BEHAVIORAL, CLUSTERS, FEATURES, FS_LABEL, MODELS, PREDICTIONS, REPORT, RFM,  # noqa: E402
                    SEGMENTATION_MODEL, TABLES, WORKED_EXAMPLE_USERS, label, slug)
from modeling import prior_correct  # noqa: E402

RT = REPORT / "tables"
FIG = REPORT / "figures"

st.set_page_config(page_title="Customer Spending Demo", page_icon=":bar_chart:", layout="wide")


# ------------------------------------------------------------------------------------------------ loading
@st.cache_resource
def load_models():
    man = json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))
    fs = man["final"]["feature_set"]
    spec = man["feature_sets"][fs]
    stage1 = joblib.load(MODELS / spec["hurdle_stage1_classifier"]["file"])
    stage2 = joblib.load(MODELS / spec["hurdle_stage2_regressor"]["file"])
    rf_reg = joblib.load(MODELS / spec["single_stage_rf_regressor"]["file"])
    seg = joblib.load(SEGMENTATION_MODEL)
    explainer = shap.TreeExplainer(stage1) if hasattr(stage1, "estimators_") else None
    return man, fs, spec, stage1, stage2, rf_reg, seg, explainer


@st.cache_data
def load_data():
    pr = pd.read_parquet(PREDICTIONS)
    feats = pd.read_parquet(FEATURES).loc[pr.index]          # test users, in test-set order
    stored_seg = pd.read_parquet(CLUSTERS)["segment"].loc[pr.index].astype(str)
    ex = pd.read_csv(WORKED_EXAMPLE_USERS)
    return feats.reset_index(drop=True), pr.reset_index(drop=True), stored_seg.reset_index(drop=True), ex


def md_section(name: str) -> str:
    """Markdown table file without its '# title' line."""
    text = (RT / f"{name}.md").read_text(encoding="utf-8")
    return re.sub(r"^# .*\n", "", text, count=1)


def md_filter(name: str, keep) -> str:
    """Keep the header, separator and the table rows for which keep(row_text) is true."""
    lines = (RT / f"{name}.md").read_text(encoding="utf-8").splitlines()
    table = [l for l in lines if l.startswith("|")]
    return "\n".join(table[:2] + [l for l in table[2:] if keep(l)])


# ------------------------------------------------------------------------------------------------ pages
def page_overview():
    st.title("Customer spending behaviour: REES46, October 2019")
    st.caption("Offline demo. All numbers are read from saved pipeline outputs; nothing is retrained.")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.subheader("T1. Dataset summary")
        st.dataframe(pd.read_csv(RT / "T1_dataset_summary.csv"), hide_index=True, height=770)
    with c2:
        st.subheader("Pipeline")
        st.image(str(FIG / "architecture.png"), width="stretch")

    st.header("Headline results (test set)")
    st.subheader("Feature-set comparison: will_purchase")
    st.markdown(md_filter("T2_classification", lambda l: True))
    st.markdown("Paired differences between feature sets (ROC-AUC):")
    st.markdown(md_filter("T2b_classification_feature_set_differences", lambda l: l.startswith("| ROC-AUC")))
    st.subheader("Hurdle vs single-stage: future_spend, all test users")
    st.markdown(md_filter("T3_spend", lambda l: "All test users" in l))
    st.markdown("Hurdle minus single-stage (negative = hurdle better):")
    st.markdown(md_filter("T3b_hurdle_vs_single_stage", lambda l: True))
    st.subheader("Leakage audit (deliberately flawed full-month features vs honest)")
    st.markdown(md_filter("T4_leakage_audit", lambda l: "| ROC-AUC |" in l or "| R² |" in l))


def page_customer():
    man, fs, spec, stage1, stage2, rf_reg, segm, explainer = load_models()
    feats, pr, stored_seg, ex = load_data()
    cols = spec["columns"]
    st.title("Customer view")
    st.caption(f"Hurdle model on feature set {FS_LABEL[fs]}: stage 1 = {spec['hurdle_stage1_classifier']['model']} "
               "classifier, stage 2 = Random Forest regressor trained on train buyers only. Models loaded from models/.")

    mode = st.radio("Choose a test user", ["Worked example (User A/B/C)", "Test-set index"], horizontal=True,
                    key="mode")
    if mode.startswith("Worked"):
        pick = st.selectbox("User", ex["label"].tolist(),
                            format_func=lambda s: f"{s}: {ex.set_index('label').loc[s, 'group']}", key="user")
        idx = int(ex.set_index("label").loc[pick, "test_index"])
        st.caption(f"{pick} is test-set index {idx} (same users as outputs/report/worked_example.md).")
    else:
        idx = int(st.number_input(f"Test-set index (0 to {len(feats) - 1:,})", min_value=0,
                                  max_value=len(feats) - 1, value=0, step=1, key="test_index"))

    row = feats.iloc[idx]
    # segment from the saved scaler + K-means
    x_seg = np.log1p(row[segm["input_columns"]].to_numpy(dtype=float)).reshape(1, -1)
    segment = segm["cluster_to_segment"][int(segm["kmeans"].predict(segm["scaler"].transform(x_seg))[0])]
    X = pd.DataFrame([{c: (float(c == f"seg_{slug(segment)}") if c.startswith("seg_") else float(row[c]))
                       for c in cols}])
    n1, n0 = man["prior_correction"]["n1_train_positives"], man["prior_correction"]["n0_train_negatives"]
    p_raw = float(stage1.predict_proba(X)[0, 1])
    p = float(prior_correct(np.array([p_raw]), n1, n0)[0])
    e = float(stage2.predict(X)[0])
    hurdle = p * e
    single = float(rf_reg.predict(X)[0])
    actual = float(pr.loc[idx, "future_spend"])

    m = st.columns(5)
    m[0].metric("Segment (K-means)", segment)
    m[1].metric("Stage 1: P(buy), corrected", f"{p:.4f}")
    m[2].metric("Stage 2: E[spend | buy]", f"{e:,.2f}")
    m[3].metric("Hurdle prediction", f"{hurdle:,.2f}")
    m[4].metric("Actual spend Oct 25-31", f"{actual:,.2f}")

    with st.expander("Arithmetic", expanded=True):
        odds_raw = p_raw / (1 - p_raw)
        odds = odds_raw * n1 / n0
        st.markdown(
            f"- Stage 1 raw probability (class-weighted): p_raw = {p_raw:.6f}\n"
            f"- Prior correction: odds = p_raw/(1-p_raw) x n1/n0 = {odds_raw:.6f} x {n1:,}/{n0:,} = {odds:.6f}; "
            f"p = odds/(1+odds) = {p:.6f}\n"
            f"- Hurdle: p x E[spend | buy] = {p:.6f} x {e:,.4f} = {hurdle:,.4f}\n"
            f"- Single-stage Random Forest prediction (same features): {single:,.4f}\n"
            f"- Actual spend Oct 25-31: {actual:,.2f}")
        diff = max(abs(hurdle - pr.loc[idx, f"hurdle|{fs}"]), abs(p - pr.loc[idx, f"pcal|{fs}"]))
        st.caption(f"Stored pipeline segment: {stored_seg.iloc[idx]}; max |difference| to the stored pipeline "
                   f"predictions for this user: {diff:.2e}.")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Features (Oct 1-24)")
        fv = [(label(c), "RFM" if c in RFM else "Behavioral", row[c]) for c in RFM + BEHAVIORAL if c in cols]
        st.dataframe(pd.DataFrame(fv, columns=["Feature", "Group", "Value"]).style.format({"Value": "{:,.3f}"}),
                     hide_index=True, height=520)
    with c2:
        st.subheader("SHAP waterfall (stage-1 classifier)")
        if explainer is None:
            st.info("Stage-1 classifier is not tree-based; no TreeExplainer waterfall.")
        else:
            sv = explainer.shap_values(X)
            sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
            base = np.ravel(explainer.expected_value)
            base = float(base[1] if base.size > 1 else base[0])
            expl = shap.Explanation(values=sv[0], base_values=base, data=X.iloc[0].to_numpy(),
                                    feature_names=[label(c, segm["segment_order"]) for c in cols])
            plt.figure()
            shap.plots.waterfall(expl, max_display=12, show=False)
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            st.pyplot(fig, width="stretch")
            plt.close("all")
            st.caption("Explains the stage-1 Random Forest output p_raw (class-weighted probability, before prior "
                       "correction): E[f(x)] is the model's average output, f(x) = p_raw for this user.")


def page_segments():
    st.title("Segments")
    st.subheader("T5. Segment profiles (K-means, K=3)")
    st.markdown(md_section("T5_cluster_profiles"))
    st.image(str(FIG / "cluster_profiles.png"), width="stretch")
    with st.expander("How the segments were named (outputs/report/cluster_naming.md)"):
        st.markdown(re.sub(r"^# .*\n", "", (REPORT / "cluster_naming.md").read_text(encoding="utf-8"), count=1))
    st.subheader("T6. Top SHAP features per segment")
    t6 = pd.read_csv(RT / "T6_shap.csv")
    segs = [s.replace("Segment: ", "") for s in t6["Scope"].unique() if s.startswith("Segment: ")]
    s = st.selectbox("Segment", segs, key="segment")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        for ranking in ["all features", "excluding seg dummies"]:
            sub = t6[(t6["Scope"] == f"Segment: {s}") & (t6["Ranking"] == ranking)]
            st.markdown(f"**Ranking: {ranking}** (n = {int(sub['n users'].iloc[0]):,} test users)")
            st.dataframe(sub[["Rank", "Feature", "Mean |SHAP|"]].style.format({"Mean |SHAP|": "{:.4f}"}),
                         hide_index=True)
        st.markdown("**Overall top 10** (2,000 test users)")
        st.dataframe(t6[t6["Scope"] == "Overall"][["Rank", "Feature", "Mean |SHAP|"]]
                     .style.format({"Mean |SHAP|": "{:.4f}"}), hide_index=True)
    with c2:
        st.image(str(FIG / f"shap_summary_segment_{slug(s)}.png"), width="stretch")


PAGES = {"Overview": page_overview, "Customer": page_customer, "Segments": page_segments}
choice = st.sidebar.radio("Page", list(PAGES), key="page")
PAGES[choice]()
