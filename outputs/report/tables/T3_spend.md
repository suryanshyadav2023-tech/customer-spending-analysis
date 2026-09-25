# T3. Spend prediction: single-stage vs hurdle (future_spend, Oct 25-31, test set)

| Feature set | Evaluated on | n | Model | MAE [95% CI] | RMSE [95% CI] | R² [95% CI] |
|---|---|---|---|---|---|---|
| (a) RFM | All test users | 51,116 | Linear Regression (single-stage) | 23.562 [22.154, 25.353] | 196.533 [136.076, 280.519] | -0.1416 [-2.2154, 0.5280] |
| (a) RFM | All test users | 51,116 | Decision Tree (single-stage) | 21.824 [20.503, 23.402] | 173.424 [133.639, 225.163] | 0.1111 [-0.1272, 0.2167] |
| (a) RFM | All test users | 51,116 | Random Forest (single-stage) | 21.501 [20.240, 23.050] | 168.153 [130.582, 216.933] | 0.1643 [-0.0956, 0.2879] |
| (a) RFM | All test users | 51,116 | Hurdle: P(buy) x E[spend \| buy] | 22.017 [20.652, 23.684] | 175.351 [139.166, 220.363] | 0.0912 [-0.3036, 0.2811] |
| (a) RFM | Test buyers only | 1,276 | Linear Regression (single-stage) | 429.291 [394.039, 472.789] | 859.092 [718.420, 1,018.028] | 0.3361 [-0.1315, 0.6143] |
| (a) RFM | Test buyers only | 1,276 | Decision Tree (single-stage) | 455.883 [413.435, 510.967] | 1,024.249 [758.115, 1,370.480] | 0.0563 [-0.2508, 0.1781] |
| (a) RFM | Test buyers only | 1,276 | Random Forest (single-stage) | 440.615 [401.101, 492.743] | 987.847 [734.772, 1,308.328] | 0.1222 [-0.1934, 0.2630] |
| (a) RFM | Test buyers only | 1,276 | Hurdle: P(buy) x E[spend \| buy] | 451.075 [410.211, 505.456] | 1,009.734 [769.185, 1,304.358] | 0.0829 [-0.3409, 0.2751] |
| (b) RFM + behavioral | All test users | 51,116 | Linear Regression (single-stage) | 23.622 [22.202, 25.416] | 196.634 [135.924, 280.896] | -0.1428 [-2.2203, 0.5273] |
| (b) RFM + behavioral | All test users | 51,116 | Decision Tree (single-stage) | 21.764 [20.463, 23.340] | 174.883 [134.581, 226.225] | 0.0960 [-0.1498, 0.2070] |
| (b) RFM + behavioral | All test users | 51,116 | Random Forest (single-stage) | 21.407 [20.140, 22.969] | 169.160 [131.891, 217.242] | 0.1542 [-0.1156, 0.2793] |
| (b) RFM + behavioral | All test users | 51,116 | Hurdle: P(buy) x E[spend \| buy] | 19.271 [18.044, 20.894] | 164.302 [123.945, 213.152] | 0.2021 [0.0284, 0.2973] |
| (b) RFM + behavioral | Test buyers only | 1,276 | Linear Regression (single-stage) | 427.176 [392.086, 470.913] | 859.131 [717.551, 1,019.007] | 0.3361 [-0.1314, 0.6141] |
| (b) RFM + behavioral | Test buyers only | 1,276 | Decision Tree (single-stage) | 457.318 [415.292, 515.181] | 1,032.870 [761.504, 1,374.134] | 0.0404 [-0.2803, 0.1770] |
| (b) RFM + behavioral | Test buyers only | 1,276 | Random Forest (single-stage) | 442.007 [401.885, 494.213] | 992.576 [740.237, 1,308.607] | 0.1138 [-0.2114, 0.2556] |
| (b) RFM + behavioral | Test buyers only | 1,276 | Hurdle: P(buy) x E[spend \| buy] | 441.947 [401.024, 494.216] | 1,001.196 [735.570, 1,324.430] | 0.0983 [-0.1882, 0.2396] |
| (c) RFM + behavioral + segment | All test users | 51,116 | Linear Regression (single-stage) | 23.635 [22.217, 25.430] | 196.624 [135.919, 280.880] | -0.1427 [-2.2199, 0.5273] |
| (c) RFM + behavioral + segment | All test users | 51,116 | Decision Tree (single-stage) | 21.742 [20.444, 23.327] | 174.880 [134.582, 226.228] | 0.0961 [-0.1497, 0.2071] |
| (c) RFM + behavioral + segment | All test users | 51,116 | Random Forest (single-stage) | 21.406 [20.137, 22.962] | 169.169 [131.899, 217.246] | 0.1541 [-0.1158, 0.2793] |
| (c) RFM + behavioral + segment | All test users | 51,116 | Hurdle: P(buy) x E[spend \| buy] | 19.296 [18.068, 20.912] | 163.701 [123.839, 212.133] | 0.2079 [0.0339, 0.3036] |
| (c) RFM + behavioral + segment | Test buyers only | 1,276 | Linear Regression (single-stage) | 427.175 [392.055, 470.934] | 859.108 [717.543, 1,018.977] | 0.3361 [-0.1315, 0.6141] |
| (c) RFM + behavioral + segment | Test buyers only | 1,276 | Decision Tree (single-stage) | 457.622 [415.631, 515.492] | 1,032.892 [761.592, 1,374.148] | 0.0404 [-0.2799, 0.1770] |
| (c) RFM + behavioral + segment | Test buyers only | 1,276 | Random Forest (single-stage) | 441.926 [401.733, 494.133] | 992.626 [740.314, 1,308.646] | 0.1137 [-0.2116, 0.2555] |
| (c) RFM + behavioral + segment | Test buyers only | 1,276 | Hurdle: P(buy) x E[spend \| buy] | 441.845 [400.818, 493.732] | 997.511 [734.765, 1,316.492] | 0.1050 [-0.1861, 0.2474] |

- Hurdle stage 1 = classifier with the highest validation ROC-AUC per feature set ((a) RFM: Logistic Regression, (b) RFM + behavioral: Random Forest, (c) RFM + behavioral + segment: Random Forest), probabilities prior-corrected for class_weight='balanced'.
- Hurdle stage 2 = Random Forest regressor trained on 5,103 train users with future_spend > 0.
- 95% CI = percentile interval over 500 bootstrap resamples of test users; the buyers-only intervals use the buyers inside each resample.
