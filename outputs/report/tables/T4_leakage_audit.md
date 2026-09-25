# T4. Leakage audit: honest vs deliberately leaky features (test set)

| Task | Feature set | Model | Metric | Honest (features Oct 1-24) | Leaky (features Oct 1-31) | Inflation (leaky - honest) |
|---|---|---|---|---|---|---|
| Classification | (a) RFM | Logistic Regression | ROC-AUC | 0.7031 | 1.0000 | +0.2969 |
| Classification | (a) RFM | Logistic Regression | PR-AUC | 0.1602 | 1.0000 | +0.8398 |
| Classification | (a) RFM | Logistic Regression | F1 @0.5 | 0.1785 | 0.9101 | +0.7317 |
| Classification | (a) RFM | Decision Tree | ROC-AUC | 0.6936 | 1.0000 | +0.3064 |
| Classification | (a) RFM | Decision Tree | PR-AUC | 0.1621 | 1.0000 | +0.8379 |
| Classification | (a) RFM | Decision Tree | F1 @0.5 | 0.1790 | 1.0000 | +0.8210 |
| Classification | (a) RFM | Random Forest | ROC-AUC | 0.7022 | 1.0000 | +0.2978 |
| Classification | (a) RFM | Random Forest | PR-AUC | 0.1703 | 1.0000 | +0.8297 |
| Classification | (a) RFM | Random Forest | F1 @0.5 | 0.1817 | 1.0000 | +0.8183 |
| Regression | (a) RFM | Linear Regression | R² | -0.1416 | -0.0368 | +0.1048 |
| Regression | (a) RFM | Linear Regression | RMSE | 196.533 | 187.295 | -9.239 |
| Regression | (a) RFM | Linear Regression | MAE | 23.562 | 19.706 | -3.857 |
| Regression | (a) RFM | Decision Tree | R² | 0.1111 | 0.4421 | +0.3310 |
| Regression | (a) RFM | Decision Tree | RMSE | 173.424 | 137.390 | -36.033 |
| Regression | (a) RFM | Decision Tree | MAE | 21.824 | 6.678 | -15.146 |
| Regression | (a) RFM | Random Forest | R² | 0.1643 | 0.5387 | +0.3744 |
| Regression | (a) RFM | Random Forest | RMSE | 168.153 | 124.935 | -43.217 |
| Regression | (a) RFM | Random Forest | MAE | 21.501 | 6.222 | -15.279 |
| Classification | (b) RFM + behavioral | Logistic Regression | ROC-AUC | 0.7783 | 1.0000 | +0.2217 |
| Classification | (b) RFM + behavioral | Logistic Regression | PR-AUC | 0.1688 | 0.9999 | +0.8311 |
| Classification | (b) RFM + behavioral | Logistic Regression | F1 @0.5 | 0.1538 | 0.9088 | +0.7551 |
| Classification | (b) RFM + behavioral | Decision Tree | ROC-AUC | 0.7766 | 1.0000 | +0.2234 |
| Classification | (b) RFM + behavioral | Decision Tree | PR-AUC | 0.1822 | 1.0000 | +0.8178 |
| Classification | (b) RFM + behavioral | Decision Tree | F1 @0.5 | 0.1235 | 1.0000 | +0.8765 |
| Classification | (b) RFM + behavioral | Random Forest | ROC-AUC | 0.7973 | 1.0000 | +0.2027 |
| Classification | (b) RFM + behavioral | Random Forest | PR-AUC | 0.2026 | 1.0000 | +0.7974 |
| Classification | (b) RFM + behavioral | Random Forest | F1 @0.5 | 0.1492 | 1.0000 | +0.8508 |
| Regression | (b) RFM + behavioral | Linear Regression | R² | -0.1428 | -0.0424 | +0.1004 |
| Regression | (b) RFM + behavioral | Linear Regression | RMSE | 196.634 | 187.795 | -8.839 |
| Regression | (b) RFM + behavioral | Linear Regression | MAE | 23.622 | 21.072 | -2.550 |
| Regression | (b) RFM + behavioral | Decision Tree | R² | 0.0960 | 0.4369 | +0.3409 |
| Regression | (b) RFM + behavioral | Decision Tree | RMSE | 174.883 | 138.022 | -36.861 |
| Regression | (b) RFM + behavioral | Decision Tree | MAE | 21.764 | 6.723 | -15.041 |
| Regression | (b) RFM + behavioral | Random Forest | R² | 0.1542 | 0.5368 | +0.3825 |
| Regression | (b) RFM + behavioral | Random Forest | RMSE | 169.160 | 125.192 | -43.968 |
| Regression | (b) RFM + behavioral | Random Forest | MAE | 21.407 | 6.279 | -15.128 |
| Classification | (c) RFM + behavioral + segment | Logistic Regression | ROC-AUC | 0.7819 | 1.0000 | +0.2181 |
| Classification | (c) RFM + behavioral + segment | Logistic Regression | PR-AUC | 0.1803 | 0.9999 | +0.8196 |
| Classification | (c) RFM + behavioral + segment | Logistic Regression | F1 @0.5 | 0.1234 | 0.9092 | +0.7858 |
| Classification | (c) RFM + behavioral + segment | Decision Tree | ROC-AUC | 0.7743 | 1.0000 | +0.2257 |
| Classification | (c) RFM + behavioral + segment | Decision Tree | PR-AUC | 0.1830 | 1.0000 | +0.8170 |
| Classification | (c) RFM + behavioral + segment | Decision Tree | F1 @0.5 | 0.1360 | 1.0000 | +0.8640 |
| Classification | (c) RFM + behavioral + segment | Random Forest | ROC-AUC | 0.7955 | 1.0000 | +0.2045 |
| Classification | (c) RFM + behavioral + segment | Random Forest | PR-AUC | 0.2029 | 1.0000 | +0.7971 |
| Classification | (c) RFM + behavioral + segment | Random Forest | F1 @0.5 | 0.1443 | 1.0000 | +0.8557 |
| Regression | (c) RFM + behavioral + segment | Linear Regression | R² | -0.1427 | -0.0413 | +0.1014 |
| Regression | (c) RFM + behavioral + segment | Linear Regression | RMSE | 196.624 | 187.700 | -8.924 |
| Regression | (c) RFM + behavioral + segment | Linear Regression | MAE | 23.635 | 21.187 | -2.447 |
| Regression | (c) RFM + behavioral + segment | Decision Tree | R² | 0.0961 | 0.4369 | +0.3409 |
| Regression | (c) RFM + behavioral + segment | Decision Tree | RMSE | 174.880 | 138.022 | -36.858 |
| Regression | (c) RFM + behavioral + segment | Decision Tree | MAE | 21.742 | 6.723 | -15.019 |
| Regression | (c) RFM + behavioral + segment | Random Forest | R² | 0.1541 | 0.5368 | +0.3826 |
| Regression | (c) RFM + behavioral + segment | Random Forest | RMSE | 169.169 | 125.191 | -43.978 |
| Regression | (c) RFM + behavioral + segment | Random Forest | MAE | 21.406 | 6.279 | -15.127 |

- LEAKY setup is deliberately flawed and used only for comparison: features are rebuilt from Oct 1-31, which overlaps the Oct 25-31 target window. Same users, targets, split, models and K-means K.
- For RMSE and MAE a negative inflation means the leaky model looks better.
