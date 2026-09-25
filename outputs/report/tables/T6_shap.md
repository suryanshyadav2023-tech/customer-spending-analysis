# T6. Top SHAP features, overall (top 10) and per segment (top 5)

| Scope | Ranking | n users | Rank | Feature | Mean \|SHAP\| |
|---|---|---|---|---|---|
| Overall | all features | 2,000 | 1 | Days since last event | 0.0528 |
| Overall | all features | 2,000 | 2 | Recency (days since last purchase) | 0.0428 |
| Overall | all features | 2,000 | 3 | Segment: Casual visitors | 0.0395 |
| Overall | all features | 2,000 | 4 | Monetary (total purchase value) | 0.0311 |
| Overall | all features | 2,000 | 5 | Avg session duration (min) | 0.0305 |
| Overall | all features | 2,000 | 6 | Frequency (purchases) | 0.0226 |
| Overall | all features | 2,000 | 7 | Has purchased (0/1) | 0.0214 |
| Overall | all features | 2,000 | 8 | Product views | 0.0194 |
| Overall | all features | 2,000 | 9 | Days active | 0.0174 |
| Overall | all features | 2,000 | 10 | Segment: Engaged browsers | 0.0149 |
| Segment: Casual visitors | all features | 1,000 | 1 | Days since last event | 0.0483 |
| Segment: Casual visitors | all features | 1,000 | 2 | Recency (days since last purchase) | 0.0430 |
| Segment: Casual visitors | all features | 1,000 | 3 | Segment: Casual visitors | 0.0387 |
| Segment: Casual visitors | all features | 1,000 | 4 | Avg session duration (min) | 0.0345 |
| Segment: Casual visitors | all features | 1,000 | 5 | Monetary (total purchase value) | 0.0306 |
| Segment: Casual visitors | excluding seg dummies | 1,000 | 1 | Days since last event | 0.0483 |
| Segment: Casual visitors | excluding seg dummies | 1,000 | 2 | Recency (days since last purchase) | 0.0430 |
| Segment: Casual visitors | excluding seg dummies | 1,000 | 3 | Avg session duration (min) | 0.0345 |
| Segment: Casual visitors | excluding seg dummies | 1,000 | 4 | Monetary (total purchase value) | 0.0306 |
| Segment: Casual visitors | excluding seg dummies | 1,000 | 5 | Frequency (purchases) | 0.0220 |
| Segment: Engaged browsers | all features | 1,000 | 1 | Days since last event | 0.0618 |
| Segment: Engaged browsers | all features | 1,000 | 2 | Segment: Casual visitors | 0.0433 |
| Segment: Engaged browsers | all features | 1,000 | 3 | Recency (days since last purchase) | 0.0395 |
| Segment: Engaged browsers | all features | 1,000 | 4 | Monetary (total purchase value) | 0.0252 |
| Segment: Engaged browsers | all features | 1,000 | 5 | Segment: Engaged browsers | 0.0229 |
| Segment: Engaged browsers | excluding seg dummies | 1,000 | 1 | Days since last event | 0.0618 |
| Segment: Engaged browsers | excluding seg dummies | 1,000 | 2 | Recency (days since last purchase) | 0.0395 |
| Segment: Engaged browsers | excluding seg dummies | 1,000 | 3 | Monetary (total purchase value) | 0.0252 |
| Segment: Engaged browsers | excluding seg dummies | 1,000 | 4 | Frequency (purchases) | 0.0200 |
| Segment: Engaged browsers | excluding seg dummies | 1,000 | 5 | Days active | 0.0185 |
| Segment: Recent buyers | all features | 1,000 | 1 | Days since last event | 0.0605 |
| Segment: Recent buyers | all features | 1,000 | 2 | Recency (days since last purchase) | 0.0517 |
| Segment: Recent buyers | all features | 1,000 | 3 | Monetary (total purchase value) | 0.0469 |
| Segment: Recent buyers | all features | 1,000 | 4 | Has purchased (0/1) | 0.0427 |
| Segment: Recent buyers | all features | 1,000 | 5 | Segment: Casual visitors | 0.0337 |
| Segment: Recent buyers | excluding seg dummies | 1,000 | 1 | Days since last event | 0.0605 |
| Segment: Recent buyers | excluding seg dummies | 1,000 | 2 | Recency (days since last purchase) | 0.0517 |
| Segment: Recent buyers | excluding seg dummies | 1,000 | 3 | Monetary (total purchase value) | 0.0469 |
| Segment: Recent buyers | excluding seg dummies | 1,000 | 4 | Has purchased (0/1) | 0.0427 |
| Segment: Recent buyers | excluding seg dummies | 1,000 | 5 | Frequency (purchases) | 0.0328 |

- Model: Random Forest classifier for will_purchase on (c) RFM + behavioral + segment (highest validation ROC-AUC among Random Forest classifiers). TreeExplainer, positive class.
- Segment indicators are constant within a segment, so their per-segment SHAP reflects the gap to the overall baseline; the 'excluding seg dummies' ranking leaves them out.
