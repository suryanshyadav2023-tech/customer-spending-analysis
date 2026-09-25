# Cluster naming

K = 3. Names are assigned by code (`common.name_segments`) from TRAIN-user means of Oct 1-24 features.

## Rule

1. Buyer cluster: share of users with a purchase in Oct 1-24 >= 0.5. One buyer cluster -> "Recent buyers"; several -> "Recent buyers, low/mid/high spend" by mean monetary.
2. Non-buyer cluster: "Engaged browsers" if mean product views AND mean sessions are both >= the train-population means (views 12.49, sessions 2.82); otherwise "Casual visitors".
3. The Oct 25-31 purchase rate is NOT used by the rule; it is listed only as a descriptive check.
4. Previous names (by mean monetary only) are listed for traceability.

## Profile numbers behind each name (train users)

| Name | Previous name | Train users | Purchased Oct 1-24 (share) | Mean monetary | Mean frequency | Mean recency (days) | Mean product views | Mean sessions | Mean days active | Oct 25-31 purchase rate (not used) | Why |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Casual visitors | Medium | 130,792 | 0.001 | 0.047 | 0.001 | 59.95 | 3.29 | 1.34 | 1.16 | 0.0086 | purchase share 0.001 < 0.5 -> non-buyer; views 3.29 < 12.49 and sessions 1.34 < 2.82 |
| Engaged browsers | Low | 50,967 | 0.000 | 0.001 | 0.000 | 60.00 | 26.76 | 4.98 | 3.35 | 0.0282 | purchase share 0.000 < 0.5 -> non-buyer; views 26.76 >= 12.49 and sessions 4.98 >= 2.82 |
| Recent buyers | High | 22,702 | 0.999 | 646.451 | 2.050 | 10.56 | 33.40 | 6.48 | 3.58 | 0.1122 | purchase share 0.999 >= 0.5 -> buyer cluster; mean recency 10.56 days (all purchases fall in Oct 1-24) |
