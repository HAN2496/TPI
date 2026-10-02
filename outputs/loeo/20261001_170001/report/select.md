# Feature selection (projection path + held-out MLPD, one-SE rule)
folds: 5; scoring budgets: [0, 5]; full size: 29; selected size: 2
Nogueira stability of the selected sets across folds: 0.038

| candidate | n_feat | MLPD | SE | Δ vs full | SE(Δ) | folds |
|---|---|---|---|---|---|---|
| k=2 | 2 | -0.5917 | 0.0559 | -0.0286 | 0.0335 | 5 |
| k=5 | 5 | -0.5488 | 0.0613 | +0.0143 | 0.0221 | 5 |
| k=10 | 10 | -0.5545 | 0.0503 | +0.0086 | 0.0110 | 5 |
| full | 27 | -0.5631 | 0.0477 | +0.0000 | 0.0000 | 5 |
| pip | 18 | -0.5498 | 0.0619 | +0.0133 | 0.0176 | 5 |

## Features selected at k=2 (fold frequency)

- IMU_LongAccelVal__p2p: 2/5
- IMU_LatAccelVal__band_low: 2/5
- Bounce_rate_6D__band_low: 1/5
- Bounce_rate_6D__p2p: 1/5
- IMU_LongAccelVal__p95_abs: 1/5
- IMU_LongAccelVal__crest: 1/5
- Bounce_rate_6D__p95_abs: 1/5
- Pitch_rate_6D__crest: 1/5

synthetic truth: common=['Pitch_rate_6D__band_mid', 'IMU_LongAccelVal__p95_abs', 'Bounce_rate_6D__band_low', 'IMU_LatAccelVal__band_low', 'IMU_VerAccelVal__p95_abs', 'Bounce_rate_6D__p2p'], individual=['IMU_LongAccelVal__p2p', 'IMU_LongAccelVal__crest', 'Pitch_rate_6D__crest']