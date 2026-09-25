# T2b. Classification: paired differences between feature sets (test set)

| Metric | Model | Comparison | Mean difference | Share of resamples improved | 95% CI |
|---|---|---|---|---|---|
| ROC-AUC | Logistic Regression | (b) - (a) | +0.0749 | 1.0000 | [+0.0641, +0.0856] |
| ROC-AUC | Logistic Regression | (c) - (b) | +0.0036 | 0.9720 | [-0.0001, +0.0074] |
| ROC-AUC | Logistic Regression | (c) - (a) | +0.0785 | 1.0000 | [+0.0680, +0.0890] |
| ROC-AUC | Decision Tree | (b) - (a) | +0.0827 | 1.0000 | [+0.0687, +0.0967] |
| ROC-AUC | Decision Tree | (c) - (b) | -0.0024 | 0.2260 | [-0.0087, +0.0035] |
| ROC-AUC | Decision Tree | (c) - (a) | +0.0803 | 1.0000 | [+0.0655, +0.0942] |
| ROC-AUC | Random Forest | (b) - (a) | +0.0949 | 1.0000 | [+0.0832, +0.1059] |
| ROC-AUC | Random Forest | (c) - (b) | -0.0018 | 0.0100 | [-0.0035, -0.0002] |
| ROC-AUC | Random Forest | (c) - (a) | +0.0931 | 1.0000 | [+0.0814, +0.1040] |
| F1 @tuned | Logistic Regression | (b) - (a) | -0.0208 | 0.0060 | [-0.0367, -0.0046] |
| F1 @tuned | Logistic Regression | (c) - (b) | +0.0119 | 0.9980 | [+0.0033, +0.0208] |
| F1 @tuned | Logistic Regression | (c) - (a) | -0.0089 | 0.0900 | [-0.0231, +0.0046] |
| F1 @tuned | Decision Tree | (b) - (a) | +0.0049 | 0.7560 | [-0.0102, +0.0182] |
| F1 @tuned | Decision Tree | (c) - (b) | +0.0000 | 0.0000 | [+0.0000, +0.0000] |
| F1 @tuned | Decision Tree | (c) - (a) | +0.0049 | 0.7560 | [-0.0102, +0.0182] |
| F1 @tuned | Random Forest | (b) - (a) | +0.0096 | 0.9000 | [-0.0064, +0.0244] |
| F1 @tuned | Random Forest | (c) - (b) | +0.0139 | 0.9920 | [+0.0024, +0.0257] |
| F1 @tuned | Random Forest | (c) - (a) | +0.0235 | 1.0000 | [+0.0098, +0.0366] |

- Paired bootstrap, 500 resamples of test users; positive = later feature set better.
