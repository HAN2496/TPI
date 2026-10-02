# Sensor subsets (exhaustive, refit + held-out MLPD)
folds: 10; scoring budgets: [0, 10, 20]

| subset | channels | MLPD | SE | Δ vs all | SE(Δ) |
|---|---|---|---|---|---|
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 3 | -0.6567 | 0.0367 | +0.0146 | 0.0115 |
| Pitch_rate_6D+IMU_VerAccelVal | 2 | -0.6639 | 0.0360 | +0.0075 | 0.0189 |
| IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.6642 | 0.0348 | +0.0072 | 0.0122 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 3 | -0.6642 | 0.0437 | +0.0072 | 0.0048 |
| IMU_VerAccelVal | 1 | -0.6651 | 0.0343 | +0.0063 | 0.0217 |
| Bounce_rate_6D+IMU_VerAccelVal | 2 | -0.6655 | 0.0384 | +0.0059 | 0.0166 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal | 4 | -0.6660 | 0.0406 | +0.0054 | 0.0065 |
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.6671 | 0.0407 | +0.0043 | 0.0062 |
| IMU_VerAccelVal+IMU_LatAccelVal | 2 | -0.6686 | 0.0354 | +0.0028 | 0.0177 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 5 | -0.6714 | 0.0457 | +0.0000 | 0.0000 |
| IMU_LongAccelVal | 1 | -0.6715 | 0.0407 | -0.0001 | 0.0179 |
| IMU_VerAccelVal+IMU_LongAccelVal | 2 | -0.6717 | 0.0408 | -0.0003 | 0.0080 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal | 3 | -0.6718 | 0.0402 | -0.0004 | 0.0140 |
| Pitch_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 3 | -0.6731 | 0.0420 | -0.0017 | 0.0131 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 4 | -0.6748 | 0.0383 | -0.0034 | 0.0138 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.6752 | 0.0502 | -0.0038 | 0.0053 |
| Bounce_rate_6D+IMU_VerAccelVal+IMU_LatAccelVal | 3 | -0.6760 | 0.0449 | -0.0046 | 0.0124 |
| IMU_LongAccelVal+IMU_LatAccelVal | 2 | -0.6772 | 0.0406 | -0.0058 | 0.0148 |
| Bounce_rate_6D+IMU_LongAccelVal | 2 | -0.6851 | 0.0597 | -0.0137 | 0.0163 |
| Bounce_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.6852 | 0.0594 | -0.0138 | 0.0164 |
| Bounce_rate_6D+IMU_LatAccelVal | 2 | -0.6867 | 0.0541 | -0.0153 | 0.0141 |
| Bounce_rate_6D | 1 | -0.6879 | 0.0553 | -0.0165 | 0.0146 |
| IMU_LatAccelVal | 1 | -0.6890 | 0.0111 | -0.0177 | 0.0434 |
| Pitch_rate_6D+IMU_LongAccelVal | 2 | -0.6913 | 0.0433 | -0.0199 | 0.0094 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LongAccelVal | 3 | -0.6924 | 0.0558 | -0.0210 | 0.0136 |
| Pitch_rate_6D+Bounce_rate_6D | 2 | -0.6949 | 0.0505 | -0.0235 | 0.0128 |
| Pitch_rate_6D | 1 | -0.6980 | 0.0366 | -0.0266 | 0.0168 |
| Pitch_rate_6D+IMU_LatAccelVal | 2 | -0.6997 | 0.0359 | -0.0283 | 0.0188 |
| Pitch_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 3 | -0.6997 | 0.0460 | -0.0283 | 0.0101 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LatAccelVal | 3 | -0.6999 | 0.0522 | -0.0285 | 0.0167 |
| Pitch_rate_6D+Bounce_rate_6D+IMU_LongAccelVal+IMU_LatAccelVal | 4 | -0.7050 | 0.0659 | -0.0336 | 0.0227 |

## Leave-one-sensor-out (Δ MLPD vs all channels)

- without Pitch_rate_6D: -0.0038 ± 0.0053
- without Bounce_rate_6D: +0.0043 ± 0.0062
- without IMU_VerAccelVal: -0.0336 ± 0.0227
- without IMU_LongAccelVal: -0.0034 ± 0.0138
- without IMU_LatAccelVal: +0.0054 ± 0.0065