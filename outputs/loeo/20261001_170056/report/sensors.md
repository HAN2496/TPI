# Sensor subsets (exhaustive, refit + held-out MLPD)
folds: 10; scoring budgets: [0, 10]

| subset | channels | MLPD | SE | Δ vs all | SE(Δ) |
|---|---|---|---|---|---|
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 3 | -0.6621 | 0.0360 | +0.0146 | 0.0119 |
| IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.6674 | 0.0339 | +0.0094 | 0.0119 |
| Pitch_rate_6D+IMU_VerAccelVal | 2 | -0.6688 | 0.0354 | +0.0079 | 0.0187 |
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.6701 | 0.0403 | +0.0066 | 0.0065 |
| IMU_VerAccelVal | 1 | -0.6702 | 0.0332 | +0.0065 | 0.0217 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 4 | -0.6710 | 0.0400 | +0.0057 | 0.0072 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 3 | -0.6711 | 0.0427 | +0.0056 | 0.0052 |
| IMU_VerAccelVal+IMU_LatAccelVal | 2 | -0.6723 | 0.0345 | +0.0045 | 0.0176 |
| Bounce_rate_6D+IMU_VerAccelVal | 2 | -0.6732 | 0.0371 | +0.0035 | 0.0164 |
| IMU_LongAccelVal | 1 | -0.6758 | 0.0398 | +0.0010 | 0.0169 |
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 3 | -0.6766 | 0.0415 | +0.0001 | 0.0128 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 5 | -0.6767 | 0.0449 | +0.0000 | 0.0000 |
| IMU_VerAccelVal+IMU_LongAccelVal | 2 | -0.6769 | 0.0397 | -0.0002 | 0.0079 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.6782 | 0.0497 | -0.0015 | 0.0060 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal | 3 | -0.6786 | 0.0391 | -0.0019 | 0.0137 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 4 | -0.6798 | 0.0375 | -0.0031 | 0.0137 |
| IMU_LongAccelVal+IMU_LatAccelVal | 2 | -0.6807 | 0.0397 | -0.0039 | 0.0132 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 3 | -0.6812 | 0.0440 | -0.0044 | 0.0122 |
| Bounce_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.6889 | 0.0589 | -0.0122 | 0.0163 |
| Bounce_rate_6D+IMU_LongAccelVal | 2 | -0.6904 | 0.0589 | -0.0137 | 0.0161 |
| IMU_LatAccelVal | 1 | -0.6905 | 0.0081 | -0.0138 | 0.0427 |
| Bounce_rate_6D+IMU_LatAccelVal | 2 | -0.6923 | 0.0533 | -0.0156 | 0.0139 |
| Bounce_rate_6D | 1 | -0.6945 | 0.0542 | -0.0178 | 0.0142 |
| Pitch_rate_6D+IMU_LongAccelVal | 2 | -0.6967 | 0.0424 | -0.0200 | 0.0082 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LongAccelVal | 3 | -0.6990 | 0.0550 | -0.0222 | 0.0132 |
| Pitch_rate_6D+Bounce_rate_6D | 2 | -0.7014 | 0.0496 | -0.0247 | 0.0120 |
| Pitch_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.7025 | 0.0456 | -0.0258 | 0.0084 |
| Pitch_rate_6D+IMU_LatAccelVal | 2 | -0.7041 | 0.0352 | -0.0274 | 0.0183 |
| Pitch_rate_6D | 1 | -0.7047 | 0.0356 | -0.0280 | 0.0165 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LatAccelVal | 3 | -0.7057 | 0.0514 | -0.0290 | 0.0161 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.7099 | 0.0653 | -0.0331 | 0.0226 |

## Leave-one-sensor-out (Δ MLPD vs all channels)

- without Pitch_rate_6D: -0.0015 ± 0.0060
- without Bounce_rate_6D: +0.0066 ± 0.0065
- without IMU_VerAccelVal: -0.0331 ± 0.0226
- without IMU_LongAccelVal: -0.0031 ± 0.0137
- without IMU_LatAccelVal: +0.0057 ± 0.0072