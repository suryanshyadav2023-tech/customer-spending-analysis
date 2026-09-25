# Worked example: hurdle prediction for three real test users

Model: hurdle on feature set (c) RFM + behavioral + segment. Stage 1 = Random Forest classifier (highest validation ROC-AUC), stage 2 = Random Forest regressor trained on train buyers only.
Selection rule (deterministic): within each group, the test user whose hurdle prediction is closest to the group's median hurdle prediction (ties: lowest user_id). Users are identified only as User A/B/C.
Values are computed at full precision and shown rounded.

## Feature values (Oct 1-24)

| Feature | User A | User B | User C |
|---|---|---|---|
| Recency (days since last purchase) | 3.814 | 60.000 | 60.000 |
| Has purchased (0/1) | 1 | 0 | 0 |
| Frequency (purchases) | 1 | 0 | 0 |
| Monetary (total purchase value) | 293.180 | 0.000 | 0.000 |
| Sessions | 1 | 3 | 1 |
| Product views | 1 | 30 | 3 |
| Add-to-cart events | 1 | 0 | 0 |
| Avg session duration (min) | 1.083 | 8.794 | 0.867 |
| Cart-to-view ratio | 1.000 | 0.000 | 0.000 |
| Purchase-to-cart ratio | 1.000 | 0.000 | 0.000 |
| Days active | 1 | 2 | 1 |
| Distinct categories viewed | 1 | 6 | 1 |
| Days since last event | 3.814 | 2.295 | 13.796 |
| Segment (K-means) | Recent buyers | Engaged browsers | Casual visitors |

## User A: segment 'Recent buyers' and bought in Oct 25-31

- Group size: 621 test users; group median hurdle prediction 44.1462.
- Stage 1 raw probability (Random Forest, balanced class weights): p_raw = 0.788520
- Raw odds: p_raw / (1 - p_raw) = 0.788520 / 0.211480 = 3.728587
- Prior correction: odds = 3.728587 x n1/n0 = 3.728587 x 5,103/199,358 = 0.095441
- Corrected probability: p = odds / (1 + odds) = 0.095441 / 1.095441 = 0.087126
- Stage 2 estimate: E[spend | buy] = 506.6944
- Hurdle prediction: p x E = 0.087126 x 506.6944 = 44.1462
- Actual spend Oct 25-31: 306.05
- Error (prediction - actual): -261.9038
- For comparison, single-stage Random Forest prediction: 76.0802

## User B: segment 'Engaged browsers' and did not buy in Oct 25-31

- Group size: 12,186 test users; group median hurdle prediction 7.2659.
- Stage 1 raw probability (Random Forest, balanced class weights): p_raw = 0.467366
- Raw odds: p_raw / (1 - p_raw) = 0.467366 / 0.532634 = 0.877463
- Prior correction: odds = 0.877463 x n1/n0 = 0.877463 x 5,103/199,358 = 0.022461
- Corrected probability: p = odds / (1 + odds) = 0.022461 / 1.022461 = 0.021967
- Stage 2 estimate: E[spend | buy] = 330.8223
- Hurdle prediction: p x E = 0.021967 x 330.8223 = 7.2672
- Actual spend Oct 25-31: 0.00
- Error (prediction - actual): +7.2672
- For comparison, single-stage Random Forest prediction: 10.2129

## User C: segment 'Casual visitors'

- Group size: 32,925 test users; group median hurdle prediction 2.9439.
- Stage 1 raw probability (Random Forest, balanced class weights): p_raw = 0.250889
- Raw odds: p_raw / (1 - p_raw) = 0.250889 / 0.749111 = 0.334916
- Prior correction: odds = 0.334916 x n1/n0 = 0.334916 x 5,103/199,358 = 0.008573
- Corrected probability: p = odds / (1 + odds) = 0.008573 / 1.008573 = 0.008500
- Stage 2 estimate: E[spend | buy] = 346.3427
- Hurdle prediction: p x E = 0.008500 x 346.3427 = 2.9439
- Actual spend Oct 25-31: 0.00
- Error (prediction - actual): +2.9439
- For comparison, single-stage Random Forest prediction: 3.2296

