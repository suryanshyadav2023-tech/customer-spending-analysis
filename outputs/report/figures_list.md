# Figures

All figures are PNG at 300 dpi in `outputs/report/figures/` (copies of `outputs/figures/` plus the architecture diagram).

| File | Caption |
|---|---|
| architecture.png | Block diagram of the pipeline: raw events, sampling, windows, features, split, K-means segmentation, single-stage and hurdle models, evaluation. |
| elbow_plot.png | K-means inertia (within-cluster sum of squares) for K = 2..8 on train users; the detected elbow is marked. |
| silhouette_plot.png | Silhouette score for K = 2..8 on a 20,000-user train sample; the maximum is marked. |
| cluster_profiles.png | Mean of eight Oct 1-24 features per segment (K=3), one panel per feature. |
| roc_curves.png | Test-set ROC curves for will_purchase, one panel per classifier, one line per feature set, with ROC-AUC in the legend. |
| actual_vs_predicted.png | Actual vs predicted Oct 25-31 spend for test users (symlog axes), feature set (c) RFM + behavioral + segment: Linear Regression, Random Forest and hurdle model. |
| rf_feature_importance.png | Impurity-based feature importance of the Random Forest classifier, (c) RFM + behavioral + segment. |
| shap_summary.png | SHAP beeswarm for the Random Forest classifier ((c) RFM + behavioral + segment) on 2,000 random test users; colour = feature value. |
| shap_summary_segment_casual_visitors.png | SHAP beeswarm for the same model on up to 1,000 test users in segment 'Casual visitors'. |
| shap_summary_segment_engaged_browsers.png | SHAP beeswarm for the same model on up to 1,000 test users in segment 'Engaged browsers'. |
| shap_summary_segment_recent_buyers.png | SHAP beeswarm for the same model on up to 1,000 test users in segment 'Recent buyers'. |
