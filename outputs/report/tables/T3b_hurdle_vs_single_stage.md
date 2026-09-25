# T3b. Hurdle minus single-stage, all test users (negative = hurdle better)

| Feature set | Comparison | Metric | Mean difference | Share of resamples hurdle better | 95% CI |
|---|---|---|---|---|---|
| (a) RFM | Hurdle (prior-corrected) minus Random Forest (single-stage) | RMSE | +7.368 | 0.0180 | [+0.519, +14.880] |
| (a) RFM | Hurdle (prior-corrected) minus Random Forest (single-stage) | MAE | +0.513 | 0.0020 | [+0.192, +0.872] |
| (a) RFM | Hurdle (prior-corrected) minus Linear Regression (single-stage) | RMSE | -17.451 | 0.5980 | [-106.120, +43.423] |
| (a) RFM | Hurdle (prior-corrected) minus Linear Regression (single-stage) | MAE | -1.533 | 1.0000 | [-2.706, -0.779] |
| (b) RFM + behavioral | Hurdle (prior-corrected) minus Random Forest (single-stage) | RMSE | -5.089 | 0.9780 | [-10.882, -0.076] |
| (b) RFM + behavioral | Hurdle (prior-corrected) minus Random Forest (single-stage) | MAE | -2.141 | 1.0000 | [-2.440, -1.843] |
| (b) RFM + behavioral | Hurdle (prior-corrected) minus Linear Regression (single-stage) | RMSE | -28.950 | 0.7700 | [-125.045, +41.187] |
| (b) RFM + behavioral | Hurdle (prior-corrected) minus Linear Regression (single-stage) | MAE | -4.341 | 1.0000 | [-5.710, -3.344] |
| (c) RFM + behavioral + segment | Hurdle (prior-corrected) minus Random Forest (single-stage) | RMSE | -5.671 | 0.9980 | [-11.176, -0.854] |
| (c) RFM + behavioral + segment | Hurdle (prior-corrected) minus Random Forest (single-stage) | MAE | -2.115 | 1.0000 | [-2.408, -1.824] |
| (c) RFM + behavioral + segment | Hurdle (prior-corrected) minus Linear Regression (single-stage) | RMSE | -29.514 | 0.7720 | [-125.329, +39.987] |
| (c) RFM + behavioral + segment | Hurdle (prior-corrected) minus Linear Regression (single-stage) | MAE | -4.330 | 1.0000 | [-5.701, -3.317] |

- Paired bootstrap, 500 resamples of test users.
