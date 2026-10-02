# Sensor subsets (exhaustive, refit + held-out MLPD)
folds: 5; scoring budgets: [0, 5]

| subset | channels | MLPD | SE | Δ vs all | SE(Δ) |
|---|---|---|---|---|---|
| Pitch_rate_6D+Bounce_rate_6D | 2 | -0.5201 | 0.0491 | +0.0430 | 0.0083 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 4 | -0.5244 | 0.0520 | +0.0387 | 0.0078 |
| Bounce_rate_6D | 1 | -0.5354 | 0.0514 | +0.0277 | 0.0052 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LongAccelVal | 3 | -0.5359 | 0.0465 | +0.0272 | 0.0130 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 4 | -0.5406 | 0.0488 | +0.0225 | 0.0133 |
| Bounce_rate_6D+IMU_LatAccelVal | 2 | -0.5410 | 0.0598 | +0.0221 | 0.0139 |
| Bounce_rate_6D+IMU_VerAccelVal | 2 | -0.5432 | 0.0571 | +0.0199 | 0.0129 |
| Pitch_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.5486 | 0.0590 | +0.0145 | 0.0163 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LatAccelVal | 3 | -0.5498 | 0.0518 | +0.0133 | 0.0138 |
| Bounce_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.5505 | 0.0630 | +0.0127 | 0.0189 |
| Bounce_rate_6D+IMU_LongAccelVal | 2 | -0.5514 | 0.0500 | +0.0117 | 0.0114 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.5519 | 0.0440 | +0.0112 | 0.0150 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal | 3 | -0.5539 | 0.0476 | +0.0092 | 0.0118 |
| IMU_LongAccelVal | 1 | -0.5552 | 0.0535 | +0.0079 | 0.0156 |
| Pitch_rate_6D+IMU_LatAccelVal | 2 | -0.5606 | 0.0523 | +0.0025 | 0.0048 |
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.5622 | 0.0603 | +0.0009 | 0.0206 |
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 3 | -0.5627 | 0.0543 | +0.0004 | 0.0143 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 5 | -0.5631 | 0.0477 | +0.0000 | 0.0000 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 3 | -0.5633 | 0.0481 | -0.0002 | 0.0080 |
| Pitch_rate_6D | 1 | -0.5637 | 0.0435 | -0.0006 | 0.0102 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 3 | -0.5651 | 0.0566 | -0.0019 | 0.0262 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.5684 | 0.0609 | -0.0052 | 0.0188 |
| Pitch_rate_6D+IMU_VerAccelVal | 2 | -0.5718 | 0.0513 | -0.0087 | 0.0049 |
| IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.5734 | 0.0545 | -0.0103 | 0.0175 |
| Pitch_rate_6D+IMU_LongAccelVal | 2 | -0.5734 | 0.0460 | -0.0103 | 0.0222 |
| IMU_LongAccelVal+IMU_LatAccelVal | 2 | -0.5742 | 0.0432 | -0.0111 | 0.0150 |
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 3 | -0.5743 | 0.0452 | -0.0112 | 0.0217 |
| IMU_VerAccelVal | 1 | -0.5776 | 0.0456 | -0.0145 | 0.0077 |
| IMU_VerAccelVal+IMU_LongAccelVal | 2 | -0.5776 | 0.0435 | -0.0145 | 0.0255 |
| IMU_VerAccelVal+IMU_LatAccelVal | 2 | -0.5783 | 0.0345 | -0.0152 | 0.0184 |
| IMU_LatAccelVal | 1 | -0.5804 | 0.0492 | -0.0173 | 0.0079 |

## Leave-one-sensor-out (Δ MLPD vs all channels)

- without Pitch_rate_6D: -0.0052 ± 0.0188
- without Bounce_rate_6D: +0.0009 ± 0.0206
- without IMU_VerAccelVal: +0.0112 ± 0.0150
- without IMU_LongAccelVal: +0.0225 ± 0.0133
- without IMU_LatAccelVal: +0.0387 ± 0.0078