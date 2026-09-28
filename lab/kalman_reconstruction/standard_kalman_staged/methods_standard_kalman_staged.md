# 표준 Kalman filter에서 출발하는 단계적 복원 실험

`lab/kalman_reconstruction/standard_kalman_staged/` — 2026-09-08

목표는 이전과 같다: IMU(수직·종가속도, 요·롤 각속도), 휠속 4채널, 모터 토크만으로 `Pitch_rate_6D`와 `Bounce_rate_6D`를 복원한다. 다른 점은 **절차**다. 기존 lab은 물리 파라미터·잡음 공분산·관측 이득을 모두 라벨에 대한 NRMSE로 Powell fit했고, 그 결과 파라미터가 탐색 경계에 붙고 거울상 해가 생겨 "KF 구조를 가진 지도학습 필터"가 되었다 (`../methods.md` §5.6, §8). 여기서는 교과서 절차를 그대로 따른다.

1. 동역학 행렬은 물리·제원·데이터의 직접 측정(모달 식별)으로 **고정**한다.
2. 잡음 공분산 (Q, R)만 데이터에서 추정하되, 라벨 없이 **innovation 최대우도**로 한다 (Särkkä & Svensson 2023 §16; Shumway & Stoffer 1982).
3. 라벨(`*_6D`)은 **검증에만** 쓴다. 라벨을 fit에 쓴 결과는 비교용으로만 남기고 ⚠ 표시한다.
4. 필터 일관성(NIS ≈ 1, innovation 백색성)을 라벨 성적보다 먼저 본다 (Bar-Shalom, Li & Kirubarajan 2001 §5.4).
5. 한 단계에 한 요소만 추가한다. 성능이 안 나와도 다음 단계로 넘어가고, 방법을 바꾸지 않는다.

표준에서 벗어난 조치는 본문에 ⚠ 로 표시하고 §8에 모아 "무엇을, 왜, 표준은 무엇, 근거"를 적었다.

---

## 1. 실험 진행 방식

```text
Stage 0 ─ 데이터 사실 확정 (KF 없음)
   │  시간 정렬 상수, 6D 처리 체인, 기하(부호·레버암), 모달 파라미터
   ▼
Stage 1 ─ 교과서 운동학 KF (차량 파라미터 불필요)
   │  Bounce: 관성항법 수직 채널          Pitch: 중력누설 관측기
   ▼
Stage 2 ─ 휠속 채널 추가 (Toyota Liu et al. 2015 의 4항 분해)
   │  Δv_w = ℓ q + κ a_b + s_f T_f + s_r T_r
   ▼
Stage 3 ─ pitch-plane half-car (heave + pitch) 로 두 target 결합
   │  노면: 앞바퀴 latent + wheelbase 지연 재생 (Louam et al. 1988)
   ▼
Stage 4 ─ 식별 검증: 합성 복원, 기하 프로파일 우도
```

| 단계 | 답하려는 질문 | 새로 들어가는 것 | 산출물 |
|---|---|---|---|
| 0 | 센서·라벨의 시간축, 부호, 기하, 차량 모드는 무엇인가 | 교차상관, 전달함수, 회귀, free decay | `stage0_*.csv/png/json` |
| 1 | 차량 모델 없이 교과서 필터만으로 어디까지 가나 | 3상태 수직채널, 5상태 중력누설 KF | `stage1_metrics.csv`, `stage1_*_waveforms.png` |
| 2 | 휠속 앞뒤 차이가 라벨 없이 pitch 부호·스케일을 고정하나 | 관측 행 1개 | `stage2_*.csv/png` |
| 3 | 물리 half-car + wheelbase 지연이 두 target을 한 필터로 내나 | 2-DOF sprung + 노면 latent 2 + 종방향 3 | `stage3_*.csv/png` |
| 4 | 추정한 파라미터는 식별 가능한가, 기하 가정에 얼마나 민감한가 | 시뮬레이션 재적합, 격자 프로파일 | `stage4_*.csv/png` |

### 1.1 공통 프로토콜

| 항목 | 설정 |
|---|---|
| 표본 | 100 Hz, 에피소드 앞 10 s (1000 샘플), speed bump 통과가 t ≈ 4–6 s |
| Train | 2,429 에피소드 (test 5명·validation 조현석 제외) |
| Dev-test | test 5명 중 김재호·김진명·김태근 128 에피소드 |
| Sealed | 신민철·이강근 34 에피소드 — 이 문서의 어떤 수치에도 쓰지 않음 |
| Fit 에피소드 | train에서 시간순 균등 120개 |
| Optimizer | Powell, maxiter 25, best-visited 재시작 2회 (`../pitch_staged_reconstruction.fit`) |
| 목적함수 ML | innovation 에너지 $-\log p(y_{1:T}\mid p)$ 의 샘플 평균 (Särkkä & Svensson Thm 16.9), 처음 100 샘플 제외 |
| 목적함수 SUP ⚠ | 라벨 NRMSE (기존 lab 방식). 비표준, 비교용 |
| 출력 보정 | Pitch: 이득 $= s_p\,180/\pi$ 고정(부호 $s_p$는 Stage 0), 오프셋만 train 평균. Bounce ⚠: 이득·오프셋 train 최소제곱 (라벨 단위 미확인) |

지표 (dev-test, 에피소드별 → 분위수):

| 지표 | 뜻 |
|---|---|
| corr | zero-lag 상관의 median [p10, p90] |
| aligned | 라벨을 Stage 0에서 측정한 6D–IMU 지연만큼 옮긴 뒤의 상관 median |
| RMSE | 보정 후 RMSE median (Pitch deg/s, Bounce 라벨 단위) |
| lag | 부호 있는 지연 median [ms], 양수 = 추정이 라벨보다 늦음 |
| free gain | 이득을 자유로 회귀했을 때 값. Pitch가 물리 스케일이면 $s_p \cdot 57.3$ |
| NIS | 정규화 innovation 제곱 평균 / 관측 차원, 일관 필터면 ≈ 1 |
| ACF(1) | 백색화 innovation 의 1-샘플 자기상관, 백색이면 ≈ 0 |

실행:

```powershell
python -m lab.kalman_reconstruction.standard_kalman_staged.run_standard_kalman_staged all --fit-episodes 120 --maxiter 25
python -m lab.kalman_reconstruction.standard_kalman_staged.run_standard_kalman_staged 3     # 개별 단계 (앞 단계 산출물 필요)
```

### 1.2 기호와 규약

좌표는 차체 고정, x 전방, z 상방, pitch 각 (θ)은 **nose-up 양수**, pitch 각속도 $q = \dot\theta$. 이것이 물리 부호이고, 라벨 `Pitch_rate_6D`와의 관계 $q_{\text{phys}} = s_p\, q_{\text{label}}$ 의 부호 $s_p$는 Stage 0 §3.3에서 데이터로 정한다.

| 기호 | 뜻 | 출처 |
|---|---|---|
| $a_z$ | IMU 수직가속도 [m/s²], $(\texttt{IMU\_VerAccelVal}-1)\cdot 9.81$ | 센서 |
| $a_x$ | IMU 종가속도 [m/s²], $\texttt{IMU\_LongAccelVal}\cdot 9.81$ | 센서 |
| $v_w$, $\Delta v_w$ | 휠속 4채널 평균 [m/s], 앞 평균 − 뒤 평균 | 센서 |
| $T_f, T_r$ | 앞/뒤 모터 토크 [Nm] (`MCU_Mg2EstTqVal` → 앞, `MCU_Mg1EstTqVal` → 뒤, `../methods.md` §5.8 매핑) | 센서 |
| $z, \dot z$ | 차체 CG 상하 변위·속도 | 상태 |
| $\theta, q$ | pitch 각·각속도 (nose-up +) | 상태 |
| $a_b$ | 차체 종가속도 (잠재, OU) | 상태 |
| $b_x$ | $a_x$ bias + $g\cdot$구배 (random walk) | 상태 |
| $v_x$ | 종속도 | 상태 |
| $r_f, r_r$ | 앞/뒤 바퀴 아래 노면 높이 | 상태 |
| $x_I, h_I$ | IMU 의 CG 대비 전방 거리·높이 [m] | 기하 |
| $x_6$ | 6D 센서의 CG 대비 후방 거리 [m] | 기하 |
| $L$ | 축거 [m] | 기하 |
| $\omega_p, \zeta_p$ / $\omega_h, \zeta_h$ | pitch / heave 고유각진동수·감쇠비 | 모달 |
| $k_w, k_T, k_6$ | 휠속·토크·라벨을 IMU 시간축에 맞추는 지연 [샘플] | Stage 0 |

강체 운동학 (Groves 2013 §2; Titterton & Weston 2004): CG에서 $x$ 앞, $h$ 위에 있는 점의 가속도는

$$a_{z,P} = \ddot z + x\,\dot q, \qquad a_{x,P} = a_b - h\,\dot q + g\,\theta$$

(회전 원심항 $\omega\times(\omega\times r)$는 $|q| < 0.6$ rad/s 라 무시). 6D 위치의 수직속도는 $\dot z - x_6\, q$.

---

## 2. 센서 체인 다이어그램

```text
                 진행 방향 →
   6D 센서 (후방 <1.5 m)            CG            IMU (전방 <0.5 m, 높이 h_I)
   ┌─────────┐                      ●             ┌─────────┐
   │ gyro q  │ ── Pitch_rate_6D     │             │ a_z, a_x│ ── CAN (지연 k_w 대비)
   │ ∫a_z,HP │ ── Bounce_rate_6D    │             │ yaw, roll│
   └─────────┘                      │             └─────────┘
        x_6                         │                  x_I
   ◄────────────────────────────────┼─────────────────►
                                    │
   휠속 FL FR RL RR ── 빠른 채널 (지연 기준점)          모터 토크 T_f, T_r
```

Stage 0 이 정하는 것: 각 채널의 상대 지연, 6D bounce 가 $a_z$ 의 어떤 함수인지, $s_p$, $h_I$, $x_I + x_6$, 그리고 free decay 로 $\omega_p, \zeta_p, \omega_h, \zeta_h$.

---

## 3. Stage 0 — 데이터 사실 확정

KF 를 돌리기 전에 모델에 상수로 들어갈 것들을 데이터에서 직접 잰다. 모두 train 에피소드만 쓴다.

### 3.1 시간 정렬

같은 물리량을 두 센서가 재는 쌍의 교차상관 정점으로 상대 지연을 잰다 (Knapp & Carter 1976). 0.3–8 Hz zero-phase 대역통과 후 계산, 양수 = 뒤 채널이 늦음.

| 쌍 | 뜻 |
|---|---|
| `IMU_RollRtVal` → `Roll_rate_6D` | CAN IMU 와 6D 칩의 시간축 차이 (같은 자이로 물리량) → $k_6$ |
| 휠속 좌우차 → `IMU_YawRtVal` | 휠속 CAN 과 IMU CAN 의 지연 (순수 기구학 관계) → $k_w$ |
| 휠속 미분 → $a_x$ | 위와 같은 지연 + 타이어·구동계 동역학 (참고용) |
| 토크 → 휠속 미분 | 구동계 응답 (참고, ⚠ 정렬 상수 $k_T = k_w +$ 이 값으로 사용) |
| $a_z$ → d(`Bounce_rate_6D`)/dt | 칩 처리 체인의 위상 (참고) |

이후 모든 모델에서 휠속·토크는 $k_w, k_T$ 만큼 **늦춰** IMU 시간축에 맞춘다 (인과적). 라벨은 평가 때 $k_6$ 만큼 옮긴 "aligned corr" 을 함께 보고한다. 기존 lab 은 이 지연을 1차 지연 상태 $\tau_I$ 로 두고 라벨로 fit 했는데, 측정 가능한 상수를 fit 하는 것은 표준이 아니다.

### 3.2 6D bounce 처리 체인

기억할 사실: `Bounce_rate_6D` 는 6D 칩이 자기 $a_z$ 를 HP + 적분한 파생 신호다. 그러면 $a_z \to$ bounce 의 전달함수는

$$H(f) \approx s_b\, K \cdot \frac{1}{j2\pi f}\cdot \frac{jf}{f_c + jf}\cdot e^{-j2\pi f\tau}$$

이어야 한다 (부호 $s_b$, 이득 $K$, 1차 HP 코너 $f_c$, 지연 $\tau$). Welch 교차스펙트럼을 에피소드 평균해 $H = S_{ab}/S_{aa}$ 와 coherence 를 구하고, $|H|\cdot 2\pi f$ 에 HP 크기를, 위상에 지연을 최소제곱 fit 한다. Stage 3 의 bounce 출력에는 이 $f_c$ 의 인과 1차 HP 를 걸어 라벨과 같은 처리 체인으로 비교한다 ⚠(§8-7).

### 3.3 기하와 부호 — 라벨을 이용한 보정 ⚠

물리 상수 $g = 9.81$ 은 **고정**하고, 센서 모델 $a_x = a_b + g\theta - h_I\dot q$ 의 부호 1비트와 $h_I$ 를 데이터에서 잰다. $a_b$ 는 휠속 미분 $\dot v_w$ 로 근사한다.

**Pitch 부호 $s_p$ (저주파).** 레버암 항은 $h_I\omega^2\theta$ 라 0.5 Hz 아래에서는 $g\theta$ 의 5 % 미만이다. 0.05–0.5 Hz 대역에서

$$a_x - \dot v_w \;\sim\; c\,\theta_{\text{label}}$$

을 회귀하면 $c \approx s_p\, g$ 이어야 하므로 $s_p = \mathrm{sign}(c)$. 교차 확인: 가속 시 차는 squat(nose-up) 하므로 물리 $\theta$ 는 토크와 양의 상관이어야 한다 → $\mathrm{corr}(\theta_{\text{label}}, T)$ 의 부호가 $s_p$ 와 같아야 한다.

**IMU 높이 $h_I$ (주파수별).** 정현파에서 $\dot q = -\omega^2\theta$ 이므로 $\theta_{\text{label}} \to (a_x - \dot v_w)$ 전달함수의 실수부는

$$\mathrm{Re}\,H(f) = s_p\,(g + h_I\,\omega^2), \qquad \omega = 2\pi f$$

Welch 교차스펙트럼(nperseg 512)으로 $H$ 를 구해 0.3–3 Hz · coherence > 0.5 인 점에서 $a + b\,\omega^2$ 를 최소제곱 fit 하면 $g_{\text{est}} = a\,s_p$ (검산: 9.81 근처여야 함), $h_I = b\,s_p$. $a_b$ 제거 방식($\dot v_w$ 전부 / 0.6배 / 없음)을 바꿔 민감도를 본다 — $\dot v_w$ 에는 Liu 의 휠속 pitch 항이 섞여 있어 $h_I$ 가 완전히 깨끗하게 나오지 않는다.

**첫 실행(run A)의 실패.** 처음에는 시간영역 회귀 $a_x - g\,s\,\theta_{\text{label}} \sim c_1\dot v_w + c_2\dot q_{\text{label}}$ 을 두 부호 가설로 돌려 R² 가 큰 쪽을 택했다. 그런데 대역통과한 $\theta$ 와 $\dot q$ 는 피치 대역에서 $\dot q \approx -\omega^2\theta$ 로 공선이라, 어느 부호를 넣어도 $\dot q$ 항이 $g\theta$ 를 흡수해 R² 비교가 무의미했고 $h_I = -0.149$ m (CG 아래) 가 나왔다. 이 값으로 돌린 Stage 1 의 `grav_hi_ml` 은 부호가 뒤집혔다(corr −0.53). run A 산출물은 `outputs/run_a_regression_h_imu/` 에 그대로 두었고, 위 주파수별 방법으로 바꾼 것이 run B(본문 수치)다. 이것은 성능을 위한 방법 변경이 아니라 보정 통계량의 추정법 오류 수정이며, ⚠ §8-12 에 기록했다.

**6D 레버암.** $\frac{d}{dt}\text{Bounce}_{6D} \approx \beta\,(a_z - (x_I + x_6)\,\dot q)$ 이므로

$$\dot b_{6D} \sim \beta_1 a_z + \beta_2 \dot q_{\text{label}}, \qquad x_I + x_6 = -\beta_2/(\beta_1 s_p)$$

이득 $\beta$ 를 몰라도 비율로 레버암 합이 나온다.

이 세 상수(부호 1비트, $h_I$, $x_I + x_6$)는 라벨을 써서 정했다. 배포 관점에서는 장착 도면·제원으로 대신할 수 있는 값이고, 회귀는 그것을 확인하는 보정 절차다. 라벨로 **필터 파라미터**를 fit 한 것이 아니라는 점이 기존 lab 과의 차이다.

### 3.4 Speed bump 특성과 모달 식별

```text
 az, q                 앞바퀴 통과      뒷바퀴 통과 (L/v ≈ 0.3 s 뒤)
   ▲                      │                 │
   │            ~~~~~~~~~~┤ 충격 ┤~~~~~~~~~~┤ 충격 ┤  ┆←── free decay 2 s ──→┆
   │                      │                 │       ┆  A e^{-ζωt} cos(ω_d t+φ)
   └──────────────────────┴─────────────────┴───────┴────────────────────────► t
                          t_f = 앞바퀴 휠속 스파이크        t_f + 0.6 s
```

- **앞바퀴 통과 시각 $t_f$**: 1–10 Hz 휠속 변동의 최대점 (bump 를 밟는 순간 하중·반경 변화로 휠속이 튄다, Liu et al. 2015 §2.3).
- **축거 $L$**: 앞→뒤 휠속 변동의 교차상관 지연 × 속도로 재려 했으나 실패했다 (§3.5). $L = 2.95$ m 가정 유지 ⚠(§8-4).
- **모달 파라미터**: $t_f + 0.6$ s 부터 2 s 구간을 free decay 로 보고 $A e^{-\zeta\omega t}\cos(\omega\sqrt{1-\zeta^2}\,t + \phi) + c$ 를 fit 한다. $(f, \zeta)$ 격자마다 선형 최소제곱(cos, sin, 상수)을 풀어 잔차 최소 격자점을 택한다 — 로그 감쇠법의 격자판이다. pitch 는 `Pitch_rate_6D`, heave 는 $a_z$ 에 적용해 $(\omega_p, \zeta_p), (\omega_h, \zeta_h)$ 의 median 을 취한다. 이 값들은 Stage 1 shaping filter 와 Stage 3 half-car 의 **시작값/고정값**이 된다. Welch PSD 정점으로 교차 확인한다.

### 3.5 Stage 0 결과

**시간 정렬** (`stage0_alignment_lags.csv`, train 2,429 에피소드, 양수 = 뒤 채널이 늦음):

| 쌍 | p10 | median | p90 | 채택 |
|---|---:|---:|---:|---|
| `IMU_RollRtVal` → `Roll_rate_6D` | 0 | +10 ms | +10 ms | $k_6 = -1$ 샘플 (6D 가 10 ms 늦음 → 평가 시 라벨을 10 ms 앞당김) |
| 휠속 좌우차 → `IMU_YawRtVal` | +20 | **+40 ms** | +60 | $k_w = 4$ 샘플 |
| 휠속 미분 → $a_x$ | +40 | +50 ms | +60 | 참고 (40 ms 지연 + ~10 ms 동역학) |
| 토크 → 휠속 미분 | −100 | +20 ms | +100 | $k_T = k_w + 2 = 6$ 샘플 (분산이 커 신뢰도 낮음) |
| $a_z$ → d(`Bounce_rate_6D`)/dt | −30 | −10 ms | 0 | 참고 (칩 HP 의 위상 앞섬) |

**6D 처리 체인** (`stage0_transfer_function.png`): $a_z \to$ bounce 는 부호 +, 이득 $K = 10.3$, 1차 HP 코너 $f_c = 0.77$ Hz, 지연 −12 ms(bounce 가 앞섬)로 fit 된다. 그러나 coherence 가 0.8 을 넘는 대역은 0.4–1.5 Hz 뿐이고, $|H|\cdot 2\pi f$ 는 2–2.5 Hz 에서 0.65 로 꺼진 뒤 4 Hz 에서 1.15 로 솟는다. 즉 "HP + 적분기" 는 1.5 Hz 아래에서만 맞고, 그 위는 IMU 위치와 6D 위치의 응답 차이거나 칩 내부의 추가 필터다. 이득 10.3 은 라벨 단위가 m/s 가 아니라는 뜻이다 (⚠ §8-2).

**기하와 부호** (`stage0_geometry_regression.csv`):

| 양 | 값 | 방법 | 비고 |
|---|---:|---|---|
| $s_p$ | **−1** | 0.05–0.5 Hz: $a_x - \dot v_w \sim c\,\theta_{\text{label}}$, $c = -8.98$ | $|c| \approx g$ 로 검산 통과. $\mathrm{corr}(\theta_{\text{label}}, T) = -0.15 < 0$ 도 일치. **`Pitch_rate_6D` 양수 = nose-down** |
| $h_I$ | **+0.149 m** (CG 위) | $\mathrm{Re}\,H = a + b\omega^2$, $a_b = \dot v_w$ 제거 | $g_{\text{est}} = 11.3$ |
| $h_I$ 민감도 | −0.29 / +0.02 m | $a_b$ 제거 없음 / 0.6 $\dot v_w$ | 제거 없음은 $g_{\text{est}} = 18.9$ 로 검산 실패(a_b–θ 상관 혼입). 0.6배는 $g_{\text{est}} = 8.45$ |
| $x_I + x_6$ | **+0.066 m** | $\dot b_{6D} \sim 7.10\,a_z + 0.467\,\dot q_{\text{label}}$, R² 0.58 | 회사 정보(최대 2 m)와 달리 **레버항이 없다**. 두 센서가 같은 수직운동을 본다 |
| 첫 시도 (run A) | −0.149 m | 시간영역 제약 회귀 | 공선성 오류, 폐기 |

기존 lab 문서는 라벨을 nose-up 양수로 적었는데(`docs/claude_explanation/pitch_rate_exploration.md`), 중력누설 부호와 squat 상관은 반대를 가리킨다. $a_x$ 는 토크와 상관 +0.96 이라 전방 양수가 확실하므로, 6D 칩의 pitch 축 규약이 nose-down 양수(ISO 8855 의 y 좌측 축 기준 우손 회전)일 가능성이 크다. 회사에 확인할 항목이다.

**Bump 특성과 모달** (`stage0_bump_modal.csv`, `stage0_bump_and_modes.png`):

| 양 | p10 | median | p90 | 비고 |
|---|---:|---:|---:|---|
| 통과 속도 [m/s] | 5.8 | 7.9 | 10.5 | |
| 축거 추정 시도 [m] | 0.35 | 0.50 | 3.94 | **실패**: 휠속·$a_z$·pitch 자기상관 모두 0.5–0.7 m(피치 반주기 × 속도)에 몰림. $L = 2.95$ 가정 유지 |
| pitch free decay $f$ [Hz] | 1.20 | **1.80** | 2.05 | 289 에피소드 |
| pitch free decay $\zeta$ | 0.02 | **0.22** | 0.34 | |
| heave($a_z$) free decay $f$ [Hz] | 1.30 | **1.65** | 2.26 | |
| heave($a_z$) free decay $\zeta$ | 0.04 | **0.24** | 0.42 | |
| PSD 정점 pitch / $a_z$ [Hz] | | 1.56 / 1.37 | | 정상 여기 기준. free decay 가 0.2–0.3 Hz 높은 것은 감쇠·비선형(bump stop) 영향 |

두 모드가 1.4–1.8 Hz 로 가깝고 감쇠비 0.2 대인 것은 승용차 ride 의 교과서 범위(Gillespie ch. 5: 1–1.5 Hz, $\zeta$ 0.2–0.4)와 부합한다.

---

## 4. Stage 1 — 교과서 운동학 KF

차량 파라미터 없이, 센서 모델과 운동학만으로 만드는 표준 필터 두 개다.

### 4.1 Bounce: 관성항법 수직 채널 (Groves 2013 §5.3, Gelb 1974 §4)

```text
   a_z (IMU) ──► ∫ ──► v ──► ∫ ──► z
        │   bias b (RW) 를 빼고 적분
        └── 의사관측 z = 0, v = 0 (큰 R) 가 저주파 drift 를 잡음 = HP
```

상태 $x = [z, v, b]^{\mathsf T}$, 입력 $u = a_z$:

$$\dot z = v, \qquad \dot v = u - b + w_v, \qquad \dot b = w_b$$

$$A_c = \begin{bmatrix}0&1&0\\0&0&-1\\0&0&0\end{bmatrix}, \quad B_c = \begin{bmatrix}0\\1\\0\end{bmatrix}, \quad Q_c = \mathrm{diag}(0, q_v, q_b)$$

관측은 실제 센서가 아니라 **의사관측** $y = [0, 0]^{\mathsf T} = [z, v]^{\mathsf T} + e$, $R = \mathrm{diag}(r_z, r_v)$. 이것이 INS 수직 채널의 표준 감쇠 방법이고(기압계 aiding 을 0 으로 대체), 정상상태에서는 $q/r$ 비가 코너를 정하는 2차 HP 적분기와 같다. 변형 `kin_v` 는 $z$ 없이 $[v, b]$ 만 두고 $v = 0$ 만 의사관측한다 (1차 HP).

기존 lab 의 1-DOF latent-force 모델 4종(RW/OU/Matérn)이 모두 0.91–0.92 에 몰린 이유가 "외란이 $a_z$ 를 통째로 흡수해 인과 BP 적분기가 된 것"이었다면, 이 3상태 필터가 같은 값을 내야 한다. 그러면 그 계열은 은퇴한다.

파라미터 $(q_v, q_b, r_z, r_v)$ 는 의사관측 우도로 ML ⚠(§8-10) 와 SUP 두 가지. 출력 $\hat v$ → 이득·오프셋 보정 → Bounce. `oracle_q` 행은 진단용으로 $\hat v - (x_I + x_6)\,q_{\text{label}}$ 을 평가해 6D 위치 보정이 얼마나 도움이 되는지(라벨 $q$ 를 썼으므로 배포 불가) 본다.

### 4.2 Pitch: 중력누설 관측기 (Tseng, Xu & Hrovat 2007; Ryu & Gerdes 2004)

```text
                        θ (nose-up +)
   a_x,IMU = a_b + g·θ + b_x − h_I·q̇      ← 중력이 x축으로 새는 양 g·θ 가 정보
   v_w     = v_x                          ← 휠속이 a_b 의 적분을 고정
             │
             └─ a_b: OU 잠재 종가속       θ,q: 2차 shaping filter (ω_p, ζ_p)
```

상태 $x = [\theta, q, b_x, a_b, v_x]^{\mathsf T}$:

$$\dot\theta = q, \qquad \dot q = -\omega_p^2\theta - 2\zeta_p\omega_p q + w_p$$

$$\dot b_x = w_b, \qquad \dot a_b = -\lambda_a a_b + w_a, \qquad \dot v_x = a_b$$

$$A_c = \begin{bmatrix}0&1&0&0&0\\-\omega_p^2&-2\zeta_p\omega_p&0&0&0\\0&0&0&0&0\\0&0&0&-\lambda_a&0\\0&0&0&1&0\end{bmatrix}, \quad Q_c = \mathrm{diag}(0, q_p, q_x, 2\sigma_a^2\lambda_a, 0)$$

관측 $y = [a_x, v_w]^{\mathsf T}$:

$$a_x = a_b + g\theta + b_x - h_I\dot q + e_x = (g + h_I\omega_p^2)\,\theta + 2h_I\zeta_p\omega_p\, q + b_x + a_b + e_x$$

$$v_w = v_x + e_w$$

$$C = \begin{bmatrix} g + h_I\omega_p^2 & 2h_I\zeta_p\omega_p & 1 & 1 & 0\\ 0&0&0&0&1\end{bmatrix}$$

- pitch 를 백색잡음 구동 2차 진동자로 두는 것은 Singer(1970) 계열의 shaping filter 이고, $(\omega_p, \zeta_p)$ 는 Stage 0 free decay 값으로 고정한다.
- 구배 $\gamma_g$ 를 별도 상태로 두지 않는다. $g\gamma_g$ 와 $b_x$ 는 같은 관측 행에 같은 RW 로 들어가 구분 불가(비관측)이므로 하나로 합친다 (`methods_gpt.md` 모델 3 이 둘을 다 둔 것은 비식별).
- 휠속은 미분 $\dot v_w$ 가 아니라 속도 $v_w$ 자체를 관측한다. 미분은 100 Hz 양자화 잡음과 구동 스파이크가 커서 (§3.5 회귀에서 $a_x \sim 0.2$–$0.6\,\dot v_w$) 적분형 관측이 조건이 좋다. 이것이 `methods_gpt.md` 모델 1 의 형태다.
- $h_I\dot q$ 항의 process noise $w_p$ 가 관측에 새는 상관잡음은 무시한다 ⚠(작음).
- 관측성: $\theta$ 는 진동, $b_x$ 는 RW 라 동역학으로 구분된다. 상수 $\theta$(정적 트림)는 $b_x$ 와 구분 불가하지만 rate 에는 무관.

변형: `grav_h0` ($h_I = 0$), `grav_hi` ($h_I$ = Stage 0 값 고정), `grav_hfree` ($h_I$ 를 ML 로 추정 — 라벨 없이 레버암이 식별되는지 시험).

### 4.3 Stage 1 결과

Dev-test 128 에피소드 (`stage1_metrics.csv`, `stage1_*_waveforms.png`). corr 은 median [p10, p90].

**Bounce** (출력 이득·오프셋 train 보정 ⚠):

| 모델 | 목적 | corr | aligned | RMSE | lag | free gain | NIS | ACF(1) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `kin_v` [v, b] | ML ⚠ | 0.301 [0.22, 0.39] | 0.264 | 0.638 | +40 | −68.7 | 0.38 | 0.83 |
| `kin_v` [v, b] | SUP ⚠ | **0.938** [0.90, 0.97] | 0.933 | 0.238 | 0 | +8.9 | 0.00 | 1.00 |
| `kin_zv` [z, v, b] | ML ⚠ | 0.301 | 0.264 | 0.638 | +40 | −68.7 | 0.19 | 0.83 |
| `kin_zv` [z, v, b] | SUP ⚠ | 0.937 [0.90, 0.96] | 0.930 | 0.240 | +10 | +8.3 | 0.00 | 1.00 |
| `kin_zv` + oracle $q$ 보정 | SUP | 0.940 | 0.932 | 0.232 | +10 | +8.5 | | |
| 참고: 기존 1-DOF RW/OU/Matérn/QC2 | SUP | 0.912–0.920 | | 0.259–0.276 | | | | |
| 참고: 인과 BP(0.3–8 Hz) 적분, KF 없음 | — | 0.879 | | | +40 | | | |

- **교과서 2상태 필터가 기존 1-DOF 계열(0.912–0.920)을 넘는다** (0.938). RW/OU/Matérn 외란 변형 비교는 필요 없었다는 뜻이다. 기존 결과들이 몰려 있던 이유(외란이 $a_z$ 를 흡수해 BP 적분기가 됨)와 정합한다.
- 의사관측 ML 은 예상대로 퇴화했다: $r_v \to$ 하한($4.5\times10^{-5}$), $q_b \to 4.0$. "관측 0" 을 완벽히 믿어 $\hat v \approx 0$ 이 되고 bias 가 신호를 먹는다. ⚠ §8-10: 표준은 $r_v$ 를 예상 속도 분산으로 **설정**하는 것이며, 여기서는 SUP 한 개 노브 fit 이 그 역할을 했다.
- SUP 의 $r_v = 3.5$–7.4 [(m/s)²] 는 $q_v = 3\times10^{-4}$ 와 함께 HP 코너를 정한다. NIS 0 · ACF(1) 1.0 은 의사관측이 실제 잡음이 아니라는 뜻이라 일관성 지표로는 무의미하다.
- Oracle $q$ 보정은 +0.003 뿐: Stage 0 의 $x_I + x_6 \approx 0$ 과 일치한다. **6D 위치 보정으로 얻을 것이 없다.**

**Pitch** (이득 $-180/\pi$ 고정, 오프셋 train; free gain 이 −57.3 이면 물리 스케일):

| 모델 | $h_I$ | 목적 | corr | aligned | RMSE [deg/s] | lag | free gain | NIS | ACF(1) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `grav_h0` | 0 | **ML** | 0.754 [0.58, 0.86] | 0.774 | 6.34 | −20 | −15.9 | 0.99 | 0.51 |
| `grav_h0` | 0 | SUP ⚠ | 0.522 [0.23, 0.78] | 0.521 | 3.51 | +10 | −32.2 | 0.04 | 0.59 |
| `grav_hi` | +0.149 | **ML** | 0.706 [0.51, 0.85] | 0.710 | **2.79** | +10 | **−49.0** | 0.99 | 0.51 |
| `grav_hi` | +0.149 | SUP ⚠ | 0.815 [0.64, 0.91] | 0.798 | 2.47 | +20 | −56.2 | 6.10 | 0.55 |
| `grav_hfree` | ML → 0.54 | ML | 0.695 [0.48, 0.83] | 0.680 | 3.39 | +30 | −134 | 0.99 | 0.51 |
| `grav_hfree` | SUP → 0.166 | SUP ⚠ | 0.819 [0.65, 0.91] | 0.801 | 2.47 | +10 | −58.8 | 6.05 | 0.58 |
| 참고: 기존 `a_naive` (같은 센서, 지연·레버암 없음) | | SUP | 0.462 | | 3.71 | | 28 | | |
| 참고: 기존 `a_full` (+지연 상태 +레버암, 라벨 fit) | | SUP | 0.836 | | 2.26 | | 53 | | |

읽는 법:

- **라벨 없는 ML 이 부호를 맞추고(모든 ML 행 free gain 음수 = $s_p$ 방향) corr 0.70–0.75 를 낸다.** 기존 lab 의 라벨-프리 ML 은 0.06–0.18 (`../methods.md` §5.8) 이었다. 차이는 모델이 아니라 Stage 0 의 상수(부호, 정렬, $h_I$)를 fit 하지 않고 넣은 것이다.
- **스케일은 $h_I$ 가 정한다.** $h_I = 0$ 이면 $a_x$ 의 피치 대역 진폭을 $g\theta$ 하나로 설명해야 해서 $\hat\theta$ 가 3.6배 커진다(free gain −15.9, RMSE 6.3). $h_I = +0.149$ 를 넣으면 관측 계수가 $g + h_I\omega_p^2 = 9.81 + 19.0 = 28.8$ 로 커져 free gain −49.0, 물리 스케일의 86 % 에 온다. Stage 0 의 $h_I$ 가 옳은 방향이라는 라벨 쪽 증거다. SUP 로 $h_I$ 를 풀면 +0.166 m 로 수렴해 이를 재확인한다.
- ML 로 $h_I$ 를 풀면 0.54 m 로 달아난다($q_p \to 0.01$). $a_x$ 의 피치 대역 분산을 큰 레버암과 작은 $\theta$ 로 설명하는 쪽이 우도상 유리하기 때문이다 — $a_x$ 한 채널로는 $(h_I, \sigma_\theta)$ 곱만 식별된다. 레버암은 제원으로 고정해야 한다.
- 모든 ML 해가 $r_x \to$ 하한($4.5\times10^{-5}$) 이다. 우도는 $a_x$ 를 완벽히 믿고 모델 오차를 process noise 로 돌린다. NIS ≈ 1 은 자기 공분산에 맞춘 결과이고 ACF(1) ≈ 0.5 는 innovation 이 백색이 아님(모델 불일치)을 말한다.
- SUP 는 $h_I = 0$ 에서 ML 보다 나쁘다(0.52). 라벨 NRMSE 는 진폭 3.6배를 억누르려 $q_a \to 18.8$ 로 $a_b$ 를 키워 피치 정보를 지운다 — 목적함수가 다르면 "좋은 필터" 도 다르다는 예.
- lag 는 −20 ~ +30 ms 로 정렬 후 기존(10–60 ms)보다 작다.

---

## 5. Stage 2 — 휠속 앞뒤 차이 채널

### 5.1 근거: Liu, Hozumi, Morita & Higuchi (Toyota, 2015)

휠속 센서는 회전부(허브)와 고정부(너클)로 나뉘어 있어, 차체·서스펜션 운동이 상대 회전으로 읽힌다. 논문은 휠속 변동을 네 항의 합으로 유도하고 실차로 확인했다:

$$\omega_{\text{meas}} = \omega_{\text{body}} + \omega_{\text{sus}} + \omega_{\text{tire}} + \omega_{\text{torq}}$$

$$r\,\omega_{\text{body}} = (r + h_s)\,q, \qquad r\,\omega_{\text{sus}} = \tan\theta_s\;\dot z_s, \qquad r\,\omega_{\text{tire}} = \frac{V\,\eta(F_z)}{2k_t r}\,\delta F_z$$

($r$ 타이어 유효반경, $h_s$ 센서 높이, $\theta_s$ 측면도 스윙암 각 — 앞 스트럿은 작고 뒤 트레일링암은 크다, $\eta$ 하중-반경 계수, $k_t$ 타이어 강성, $\delta F_z$ 하중 변동). 그 위에 4-DOF pitch-plane half-car 와 KF 로 pitch 속도를 추정해 자이로와 비교했다. 즉 기존 lab 이 "근거 없음"으로 적었던 휠속 관측식은 사실 문헌이 있다. 다만 우리는 $h_s, \theta_s, \eta, k_t$ 를 모르므로 앞뒤 차이로 뭉친 형태를 쓴다:

$$\Delta v_w = v_{w,f} - v_{w,r} = \ell\, q + \kappa\, a_b + s_f T_f + s_r T_r + e_d$$

- $\ell$ [m]: 앞뒤 (스윙암·하중) 항의 차이가 pitch 에 비례하는 유효 레버. 앞뒤가 반대 부호로 스트로크하므로 $q$ 항만 살아남는다.
- $\kappa$ [s]: 하중이동 + 구동 슬립이 $a_b$ 에 비례하는 항.
- $s_f, s_r$: 토크 feedthrough (슬립).

### 5.2 회귀로 계수 고정, KF 관측 행 추가

$(\ell, \kappa, s_f, s_r)$ 은 train 라벨 회귀로 정한다 ⚠(§8-5). 속도 3분위별로 다시 회귀해 하중항이 $V$ 에 비례하는지(Liu 식 (19)) 본다. 그 뒤 Stage 1 의 `grav_hi` 모델에 관측 행 하나를 더한다:

$$y = [a_x,\ v_w,\ \Delta v_w]^{\mathsf T}, \qquad C_3 = [\,0,\ \ell,\ 0,\ \kappa,\ 0\,], \qquad D_3 = [s_f,\ s_r]$$

변형 `wheel_free` 는 $\ell, \kappa$ 를 ML 로 추정한다. 라벨 없는 우도가 회귀값과 같은 부호·크기를 찾으면 "휠속 채널이 pitch 의 부호·스케일을 센서만으로 고정한다"는 기존 관찰(`../methods.md` §5.8: ML corr 0.17 → 0.55)이 표준 절차에서 재현되는 것이다.

### 5.3 Stage 2 결과

**회귀** (`stage2_wheel_speed_regression.csv/png`, train, 0.3–5 Hz zero-phase, $q$ 는 물리 부호):

| 부분집합 | n | R² | R² ($q$ 만) | $\ell$ [m] | $\kappa$ [s] | $s_f$ | $s_r$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전체 | 2,429 | 0.47 | 0.14 | **0.175** | −0.0097 | −7e-5 | −3.6e-4 |
| 속도 3.0–7.5 m/s | 802 | 0.53 | 0.02 | 0.109 | −0.0077 | 3e-4 | −3e-4 |
| 속도 7.5–9.0 m/s | 801 | 0.53 | 0.14 | 0.148 | −0.0065 | −1e-4 | −3e-4 |
| 속도 9.0–16.9 m/s | 826 | 0.56 | 0.30 | **0.275** | −0.0072 | −2e-4 | −5e-4 |

- $\ell$ 은 양수(물리 부호 nose-up 기준), 기존 문서의 −0.2~−0.3 m 은 라벨 부호 기준이라 같은 값이다.
- **$\ell$ 이 속도에 거의 비례한다** (0.11 → 0.28 m). Liu 식 (19) 의 하중항 $V\eta\,\delta F_z/(2k_t r)$ 이 지배적이라는 뜻이고, 고정 $\ell$ 은 타협이다. 속도 스케줄 $\ell(V) = \ell_0 + \ell_1 V$ 는 시변 $C$ 를 가진 표준 선형 KF 로 처리할 수 있다 (§9 제안).
- $q$ 만으로는 R² 0.14 지만 나머지 항($\dot v_w$, $V a_z$, $V\dot q$, 토크)을 넣어도 0.47 — $\Delta v_w$ 의 절반은 이 선형식 밖(좌우 비대칭, 슬립, 양자화)에 있다. 토크 feedthrough 는 무시할 수준.

**KF** (`stage2_metrics.csv`, `stage2_pitch_waveforms.png`; $h_I = 0.149$, $\ell, \kappa, s$ 고정):

| 모델 | 목적 | corr | aligned | RMSE | lag | free gain | NIS | ACF(1) | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `pitch_wheel` | ML | 0.646 [0.34, 0.76] | 0.617 | 3.21 | +40 | −40.9 | 1.00 | 0.62 | Stage 1 `grav_hi` ML 0.706 보다 **나빠짐** |
| `pitch_wheel` | SUP ⚠ | 0.844 [0.63, 0.91] | 0.819 | 2.40 | +20 | −58.5 | 6.55 | 0.65 | Stage 1 SUP 0.815 → +0.03 |
| `pitch_wheel_free` ($\ell,\kappa$ ML) | ML | 0.518 [0.18, 0.72] | 0.487 | 3.37 | +40 | −35.3 | 0.91 | 0.57 | $\ell \to 1.0$ (상한) |
| `pitch_wheel_free` ($\ell,\kappa$ SUP) | SUP ⚠ | **0.884** [0.74, 0.93] | 0.863 | 2.21 | +10 | −68.9 | 4.26 | 0.64 | $\ell = 0.547$, $\kappa = -0.071$ |
| 참고: 기존 `c_wheel` / `d_torque` | SUP | 0.887 / 0.936 | | 2.01 / 1.48 | | 54 / 57 | | | 지연·레버암·토크 모두 라벨 fit |
| 참고: 기존 `c_wheel` | ML | 0.545 | | | | 11 | | | |

- **SUP 에서는 휠속 채널이 단계적으로 도움이 된다** (0.815 → 0.844 → 0.884). 기존 `c_wheel` 0.887 과 같은 수준이며, 우리 쪽은 지연·레버암을 fit 하지 않았다.
- **ML 에서는 반대로 나빠진다** (0.706 → 0.646). 우도는 $\Delta v_w$ 채널을 설명하려 $q_x$ 를 0.006 → 0.28 로 키우고($b_x$ 가 $a_x$ 의 저주파를 흡수) $\theta$ 의 정보를 줄인다. $\ell$ 을 풀면 상한 1.0 으로 달아난다. 즉 **라벨 없는 우도는 이 채널의 계수를 식별하지 못한다.** 기존 문서의 "ML 0.17 → 0.55 점프" 는 깨진 기준선 대비였고, 정렬·상수를 제대로 넣은 기준선(0.706) 대비로는 이득이 없다.
- 회귀 $\ell$ (0.175) 과 SUP 자유 $\ell$ (0.547) 이 3배 다르다. 회귀는 $\Delta v_w$ 의 분산을 설명하는 계수, SUP 는 라벨을 맞추는 계수라 다를 수 있고, 속도 의존성(0.11–0.28)이 그 사이에 있다.

---

## 6. Stage 3 — pitch-plane half-car

### 6.1 모델

```text
(옆에서 본 그림)                                      진행 방향 →
              z ↑   θ (nose-up +)
       ┌──────────────────────────────┐
       │  sprung mass  m_s, I_y       │  ← a_b (종가속, OU)  → 하중이동 모멘트 g_u·a_b
       └──┬────────────────────────┬──┘
      k_r │c_r                 k_f │c_f
    ~~~~~~┴~~~~ r_r          ~~~~~~┴~~~~ r_f     노면: r_f latent (OU 속도), r_r(t) = r̂_f(t − L/v)
       ├── l_r ──┤├──── l_f ────┤
                 6D (x_6 뒤)      IMU (x_I 앞, h_I 높이)
관측: a_z = z̈ + x_I q̇,  a_x = a_b + gθ + b_x − h_I q̇,  v_w = v_x,  Δv_w = ℓq + κa_b + s·T
출력: q → Pitch,   ż − x_6 q → 칩 HP → Bounce_6D
```

Gillespie(1992) 5장의 pitch-bounce 2-DOF ride 모델이다. unsprung 질량은 두지 않는다 — wheel-hop(10–15 Hz)은 두 target 의 대역(< 3 Hz) 밖이고, $a_z$ 하나로 $k_t/k_s$ 가 식별되지 않았다는 것이 기존 lab 의 결론이다 (`../methods.md` §4). 질량 정규화 계수:

$$a_f = \tfrac{\omega_h^2}{2}(1+\varepsilon),\quad a_r = \tfrac{\omega_h^2}{2}(1-\varepsilon),\quad b_f = \zeta_h\omega_h(1+\varepsilon),\quad b_r = \zeta_h\omega_h(1-\varepsilon),\quad j_r = j\, l_f l_r = I_y/m_s$$

상태 $x = [z, \dot z, \theta, q, r_f, \dot r_f, r_r, \dot r_r, a_b, b_x, v_x]^{\mathsf T}$ (11), 입력 $u = [T_f, T_r]^{\mathsf T}$. 앞뒤 서스펜션 힘 (질량 정규화, 위 방향 +):

$$F_f = -a_f(z + l_f\theta - r_f) - b_f(\dot z + l_f q - \dot r_f), \qquad F_r = -a_r(z - l_r\theta - r_r) - b_r(\dot z - l_r q - \dot r_r)$$

$$\ddot z = F_f + F_r + w_z, \qquad \dot q = \frac{l_f F_f - l_r F_r}{j_r} + g_u\, a_b + w_q$$

$$\dot r_f = \dot r_f,\quad \ddot r_f = -\lambda_r \dot r_f + w_r, \qquad \dot a_b = -\lambda_a a_b + w_a,\quad \dot b_x = w_b,\quad \dot v_x = a_b$$

관측 $y = [a_z, a_x, v_w, \Delta v_w]^{\mathsf T}$:

$$a_z = \ddot z + x_I\dot q + e_z, \qquad a_x = a_b + g\theta + b_x - h_I\dot q + e_x, \qquad v_w = v_x + e_w, \qquad \Delta v_w = \ell q + \kappa a_b + s_f T_f + s_r T_r + e_d$$

($\ddot z, \dot q$ 는 위 동역학 식으로 치환해 $C$ 를 상태의 선형식으로 만든다.)

파라미터 취급:

| 기호 | 취급 | 출처 |
|---|---|---|
| $l_f = l_r = L/2$, $L = 2.95$ | 고정 ⚠ | 가정 (§8-4) |
| $x_I, x_6, h_I$ | 고정 | Stage 0 회귀 |
| $\ell, \kappa, s_f, s_r$ | 고정 | Stage 2 회귀 |
| $\lambda_a, \sigma_a^2, q_x, r_x, r_w, r_d$ | 고정 | Stage 2 `pitch_wheel_ml` 의 ML 값 |
| $\omega_h, \zeta_h$ | ML (시작값 = Stage 0 heave free decay) | |
| $\varepsilon, j, g_u$ | ML (범위: $\varepsilon\in[-.8,.8]$, $j\in[.3,3]$, $g_u\in[-5,5]$) | 승용차 물리 범위 |
| $\lambda_r, q_r, q_{\text{body}}, r_z$ | ML | |

### 6.2 노면 입력: 지연 재생 vs 독립

Speed bump 는 정상 랜덤 노면(ISO 8608)이 아니라 결정적 펄스다. 뒷바퀴는 앞바퀴가 본 프로파일을 $L/v$ 뒤에 밟는다 (Louam, Wilson & Sharp 1988; Gillespie 의 wheelbase filtering). 두 처리를 비교한다.

- `halfcar_delay`: $r_f, \dot r_f$ 만 latent 로 추정하고, $r_r, \dot r_r$ 상태는 process noise 0 · 초기 분산 $10^{-8}$ 로 두고 매 스텝 예측치를 $\hat r_f(t - L/v)$ 의 필터 사후평균으로 덮어쓴다 (휠속 누적 거리로 인덱싱). 재생값의 불확실성은 공분산에 반영하지 않는다 ⚠(§8-6).
- `halfcar_indep`: $r_r$ 도 독립 OU 로 추정한다 (기존 `pitch_road` 와 같은 구조).

### 6.3 출력

Pitch 는 $q$. Bounce 는 6D 위치의 수직속도 $\dot z - x_6 q$ 에 Stage 0 에서 식별한 코너 $f_c$ 의 인과 1차 HP 를 걸어 라벨 처리 체인과 맞춘 뒤 이득·오프셋 보정한다. 한 필터가 두 target 을 낸다는 것이 이 단계의 요점이고, SUP 목적함수는 pitch 라벨만 쓰므로 bounce 는 SUP 에서도 라벨을 보지 않은 출력이다.

### 6.4 Stage 3 결과

`stage3_metrics.csv`, `stage3_{pitch,bounce}_waveforms.png`, `stage3_halfcar_states_median.png`. 잡음·휠속 계수는 Stage 2 `pitch_wheel_ml` 값 고정, $x_I = 0.016$, $x_6 = 0.049$ (Stage 0), $h_I = 0.149$.

| 모델 | 목적 | Pitch corr | aligned | RMSE | free gain | Bounce corr | Bounce RMSE | NIS | ACF(1) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `halfcar_delay` | ML | 0.429 [0.12, 0.58] | 0.378 | 9.08 | −7.0 | 0.194 | 0.655 | 1.03 | 0.55 |
| `halfcar_delay` | SUP ⚠ | 0.651 [0.43, 0.84] | 0.654 | 3.14 | −45.9 | **0.826** | 0.413 | 3.22 | 0.72 |
| `halfcar_indep` | ML | 0.441 [0.15, 0.59] | 0.375 | 7.07 | −9.1 | 0.352 | 0.626 | 1.05 | 0.54 |
| `halfcar_indep` | SUP ⚠ | 0.757 [0.53, 0.87] | 0.737 | 2.83 | −52.1 | 0.627 | 0.532 | 1.31 | 0.59 |
| 참고: Stage 2 `pitch_wheel` | ML / SUP | 0.646 / 0.844 | | | | — | | | |
| 참고: Stage 1 `kin_v` | SUP | — | | | | 0.938 | 0.238 | | |
| 참고: 기존 `pitch_delay` (4-DOF, 라벨 fit) | SUP | 0.882 | | 2.23 | | — | | | |

식별된 플랜트 파라미터 (ML / SUP):

| | $f_h$ [Hz] | $\zeta_h$ | $\varepsilon$ | $j$ | $g_u$ | $\lambda_r$ | $q_r$ | $q_{\text{body}}$ | $r_z$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `delay` ML | **0.60↓** | 0.15 | **0.80↑** | **3.0↑** | 0.66 | 17.6 | 0.17 | 0.32 | ↓ |
| `delay` SUP | 1.23 | 0.84 | 0.34 | 1.00 | 1.07 | 41.7 | 5e-4 | ~0 | 0.002 |
| `indep` ML | 0.94 | 0.52 | 0.06 | 1.95 | 0.82 | 0.57 | 0.04 | 0.15 | ↓ |
| `indep` SUP | 2.18 | 0.05↓ | −0.49 | 2.05 | 1.19 | 50↑ | 0.004 | 0.008 | 0.03 |

(↓↑ = 탐색 경계. Stage 0 free decay 시작값 $f_h = 1.65$, $\zeta_h = 0.24$.)

- **물리 half-car 를 얹자 pitch 가 나빠졌다.** ML 0.706(Stage 1) → 0.646(Stage 2) → 0.43(Stage 3); SUP 0.884 → 0.65–0.76. 기존 4-DOF `pitch_delay` 0.882 도 못 미친다.
- ML 해는 경계에 붙었다: heave 0.6 Hz(하한), $\varepsilon = 0.8$(앞 강성 9배), $j = 3$(상한). free decay 의 1.65 Hz 를 버리고 저주파 플랜트로 간 것은 우도가 $a_z$ 의 one-step 예측에 지배되기 때문이다 — $a_z$ 는 분산이 크고 $r_z$ 가 하한이라 우도 대부분이 이 채널에서 나오고, 피치는 부수적이다. **Stage 4 합성 복원(§7.3)은 이 파라미터가 데이터에서 식별 가능함을 보이므로, 문제는 optimizer 가 아니라 모델 불일치다.**
- SUP `delay` 는 $f_h = 1.23$, $j = 1.0$, $\varepsilon = 0.34$, $g_u = 1.07$ 로 승용차 범위에 들어왔지만 $\zeta_h = 0.84$ 가 비물리적으로 크다(과감쇠로 플랜트를 "끄고" 노면 상태가 대신함).
- **Bounce 는 half-car 로 0.83(SUP delay) 이 최고**이고 Stage 1 의 3상태 필터 0.938 에 못 미친다. $x_I + x_6 \approx 0$ 이라 두 센서가 같은 수직운동을 보므로, "pitch 로 6D 위치를 보정한다"는 half-car 의 존재 이유가 이 데이터에는 없다.
- `delay` vs `indep`: ML 은 비슷(0.43/0.44), SUP 는 pitch 에서 indep 가(0.76 vs 0.65), bounce 에서 delay 가(0.83 vs 0.63) 낫다. 기존 문서의 "지연 연결이 이긴다" 는 이 축소 모델에서는 재현되지 않는다.
- 상태 그림(median 에피소드): ML 추정 pitch 는 라벨보다 진폭 2배·고주파 잡음이 크고, bounce 는 라벨의 1/5 진폭이다. 노면 $\hat r_f$ 는 −0.5 m 까지 흘러내리는데 이는 $r_f$ 가 관측 불가한 상수·기울기 방향(blocking zero $s = 0$)을 가지기 때문이고 rate 출력에는 무해하다.

---

## 7. Stage 4 — 식별 검증

### 7.1 합성 복원

`halfcar_delay` 의 ML 파라미터를 참값으로 놓고 실제 토크·속도(재생 거리)를 입력으로 시뮬레이션해 $y$ 를 만든 뒤, 기본 시작점에서 ML 로 재적합한다. 참값이 돌아오면 ML 목적함수가 그 파라미터를 식별할 수 있다는 뜻이고, 안 돌아오면 데이터가 아니라 모델·목적함수의 비식별이다 (Ljung 1999 §13; `../methods.md` §3 의 bound sensitivity 와 같은 취지).

### 7.2 기하 프로파일

$x_I + x_6 \in \{0.5, 1, 1.5, 2\}$ m × $h_I \in \{0, 0.2, 0.4, 0.6\}$ m 격자에서 잡음 두 개($q_{\text{body}}, r_z$)만 재적합하고 dev innovation 에너지·pitch corr·bounce corr 을 본다. 라벨 없는 우도가 회사 제원 범위 안에서 최소를 갖는지, 아니면 평평한지(비식별)를 보는 프로파일 우도(Raue et al. 2009)의 거친 판이다.

### 7.3 Stage 4 결과

**합성 복원** (`stage4_synthetic_recovery.csv`; 참값 = `halfcar_delay_ml`, 120 에피소드 시뮬레이션, 기본 시작점에서 ML 재적합):

| 파라미터 | 참값 | 시작값 | 복원값 |
|---|---:|---:|---:|
| $f_h$ [Hz] | 0.600 | 1.65 | 0.615 |
| $\zeta_h$ | 0.148 | 0.24 | 0.190 |
| $\varepsilon$ | 0.800 | 0 | 0.800 |
| $j$ | 3.00 | 1.0 | 3.00 |
| $g_u$ | 0.664 | 0 | 0.663 |
| $\lambda_r$ | 17.6 | 2.0 | 19.2 |
| $q_r$ | 0.168 | 0.018 | 0.081 |
| $q_{\text{body}}$ | 0.322 | 0.0025 | 0.334 |
| $r_z$ | 4.5e-5 | 0.1 | 4.5e-5 |

에너지: 참값 −4.4488, 복원 −4.4481. 시뮬레이션 참 $q$ 와 복원 필터 $\hat q$ 의 corr 0.982. **ML 목적함수와 Powell 은 이 모델의 파라미터를 합성 데이터에서 복원한다** ($q_r$ 만 절반). 따라서 §6.4 의 경계 해는 optimizer 실패가 아니라, 실제 데이터가 이 2-DOF half-car + OU 노면으로 설명되지 않는다는 뜻이다.

**기하 프로파일** (`stage4_geometry_profile.csv/png`; `halfcar_delay_ml` 에서 $q_{\text{body}}, r_z$ 만 재적합):

| | dev innovation 에너지 (낮을수록 좋음) | pitch corr | bounce corr |
|---|---|---|---|
| $h_I$ 방향 | −0.2: −4.30, 0: −4.36, **+0.2: −4.39**, +0.4: −4.36~−4.39 | 0.41–0.43 (평평) | |
| $x_I + x_6$ 방향 | 0 → 1.5 m 에서 변화 ≤ 0.01 (평평) | 0.43 → 0.41 | **0.19 → 0.04** (0 이 최선) |

- 라벨 없는 우도가 $h_I \approx +0.2$ m 를 선호한다 — Stage 0 의 +0.149 와 같은 방향. $h_I < 0$ 은 뚜렷히 나쁘다.
- $x_I + x_6$ 은 우도에 평평하다(비식별). 대신 bounce corr 이 0 에서 최대라 Stage 0 의 "레버항 없음" 과 일치한다.
- pitch corr 은 기하에 거의 무감(0.41–0.43) — Stage 3 의 pitch 부진은 기하 때문이 아니다.

---

## 8. 표준에서 벗어난 조치 ⚠ 종합

| # | 조치 | 왜 | 표준은 | 근거 |
|---|---|---|---|---|
| 1 | SUP 목적함수(라벨 NRMSE)로 잡음·구조 파라미터 fit 한 행을 함께 보고 | 기존 lab 과 비교, ML 이 라벨과 어긋날 때 원인 진단 | 잡음은 ML/EM 또는 ALS, 라벨은 검증 | Shumway & Stoffer 1982; Odelson, Rajamani & Rawlings 2006 |
| 2 | Bounce 출력 이득·오프셋을 train 라벨로 최소제곱 | 라벨 단위 미확인 (Stage 0 이득 $K \approx 10$ 이 SI 가 아님) | 출력 단위 고정 | 칩 스펙 확인 필요 |
| 3 | $s_p$, $h_I$, $x_I + x_6$ 를 라벨 회귀로 결정 | 장착 도면 없음 | 제원·도면, 또는 라벨 없는 정적 캘리브레이션 | Groves §4 (lever arm calibration) |
| 4 | 축거 $L = 2.95$ 고정 | 휠속·$a_z$·pitch 자기상관으로 추정 시도 → 0.5–0.7 m(피치 반주기×속도)에 몰려 실패 | 제원 | 차량 제원 요청 |
| 5 | 휠속 계수 $\ell, \kappa, s_f, s_r$ 를 라벨 회귀로 고정 | $h_s, \theta_s, \eta, k_t$ 미상 | Liu 식 (28)–(29) 의 기하 계수 | Liu et al. 2015 |
| 6 | 뒷바퀴 노면을 앞바퀴 추정치의 지연 재생으로, 공분산 미반영 | 가변 지연을 상태로 두면 차원 폭발 | 지연 상태 증강(Padé) 또는 preview 관측기 | Louam et al. 1988; Kwon, Kang & Yi 2020 |
| 7 | Bounce 출력에 칩 HP($f_c$) 적용 | 라벨이 HP+적분 파생 신호 | 물리 속도를 그대로 비교 | Stage 0 전달함수 |
| 8 | 토크 정렬 $k_T$ 를 토크→휠가속 지연으로 | 토크 CAN 지연을 따로 잴 쌍이 없음 | 타임스탬프 | 정렬 오차 ≤ 1 샘플 수준 |
| 9 | 구배 상태 제거, $b_x$ 에 흡수 | 같은 관측 행의 RW 두 개는 비관측 | 구배 추정에는 GPS/속도 융합 필요 | Sahlholm & Johansson 2010 |
| 10 | 의사관측(0)의 innovation 우도로 Stage 1 bounce 잡음 fit | 실제 관측이 $a_z$ 입력 하나뿐 | $r_v$ 를 예상 속도 분산으로 설정 | Groves §5.3 |
| 11 | Powell best-visited 재시작 (미분 없는 최적화) | 기존 코드 재사용 | 로그 파라미터화 + BFGS(수치/해석 gradient) 또는 EM | Särkkä & Svensson §16.3–16.4; Shumway & Stoffer 1982 |
| 12 | Stage 0 $h_I$ 추정법을 시간영역 제약 회귀(run A) → 주파수별 전달함수 기울기(run B)로 교체 | $\theta$·$\dot q$ 공선성으로 run A 값(−0.149 m)이 부호까지 틀림; Stage 1 에서 부호 반전으로 드러남 | 장착 도면; 또는 정적 캘리브레이션 | §3.3. run A 산출물 보존 |
| 13 | 구배·bias 통합, 휠속을 미분 대신 속도로 관측 | 비관측 상태 제거, 미분 잡음 회피 | 표준 (관측 가능한 최소 상태, 센서가 재는 양을 관측) | Groves §14; `methods_gpt.md` 모델 1 |

---

## 9. 결론과 다음 제안

### 9.1 단계별 결론

| 단계 | 질문 | 답 |
|---|---|---|
| 0 | 센서·라벨의 사실 | 휠속 CAN 이 IMU 보다 40 ms 빠름, 6D 는 IMU 보다 10 ms 늦음. `Pitch_rate_6D` 양수 = **nose-down**. IMU 는 CG 위 ≈ 0.15 m. **6D bounce 에 pitch 레버항 없음**($x_I + x_6 \approx 0.07$ m). 칩 bounce = 10.3 × HP(0.77 Hz) ∫$a_z$, 1.5 Hz 위는 다른 필터. pitch 1.8 Hz / heave 1.65 Hz, $\zeta \approx 0.22$. 축거는 식별 실패 |
| 1 | 차량 모델 없는 교과서 필터 | Bounce: 3상태 INS 수직 채널 **0.938** > 기존 1-DOF 4종(0.912–0.920). Pitch: 중력누설 KF, 라벨 없는 ML 로 부호 정확·corr 0.71–0.75, $h_I$ 고정 시 스케일 86 %. 라벨 fit 은 0.82 |
| 2 | 휠속 채널이 라벨 없이 부호·스케일을 고정하나 | **아니오.** SUP 는 0.815 → 0.884 로 올라가지만 ML 은 0.706 → 0.646 으로 내려가고 $\ell$ 은 경계로 달아남. 회귀는 Liu 의 속도 비례 하중항 구조를 확인($\ell$: 0.11 → 0.28 m) |
| 3 | 물리 half-car 로 두 target 을 한 필터로 | **실패.** ML 은 경계 해(heave 0.6 Hz, $j$ 3), pitch 0.43. SUP 도 pitch 0.65–0.76, bounce 0.83 으로 Stage 1–2 의 단순 필터보다 나쁨 |
| 4 | 식별 가능성 | 합성 데이터에서는 ML 이 파라미터를 복원(corr 0.98) → Stage 3 의 실패는 **모델 불일치**. 우도는 $h_I \approx +0.2$ 를 선호하고 $x_I + x_6$ 에는 평평 |

전체 그림: **표준 절차(모델 고정 + 라벨 없는 ML) 로 얻은 최선은 pitch 0.71–0.75(Stage 1), bounce 는 SUP 한 노브로 0.938(Stage 1) 이다.** 기존 lab 의 라벨 fit 최고(pitch 0.936, `d_torque`)와 비교하면 pitch 에서 0.2 의 격차가 있고, 그 격차의 대부분은 "라벨로 필터를 조정했는가" 에서 온다. 물리 모델을 복잡하게 할수록(Stage 3) 라벨 없는 우도는 오히려 나빠졌다.

### 9.2 왜 라벨 없는 우도가 pitch 에 약한가

1. **우도는 관측의 one-step 예측 오차를 본다.** $a_x$ 분산의 대부분은 $a_b$(구동)와 잡음이고 pitch 는 작은 몫이다. $r_x, r_z$ 가 하한으로 가는 것은 우도가 센서를 완벽히 믿고 모델 오차를 process noise 로 흡수한다는 뜻이며, 그 결과 innovation 이 유색(ACF(1) 0.5–0.6)이다. 잘 맞춘 우도 ≠ 좋은 pitch 추정.
2. **$a_x$ 한 채널로는 $(h_I, \sigma_\theta)$ 의 곱만 식별된다.** 레버암을 제원으로 못 박아야 스케일이 나온다 (Stage 1 `hi` vs `h0` vs `hfree`).
3. **휠속 채널의 계수 $\ell$ 은 속도 의존**이라 고정 $\ell$ 은 모델 오차가 되고, ML 은 그 오차를 $\ell$ 을 키우거나 $b_x$ 를 키워 흡수한다.

### 9.3 다음 제안 (표준 안에서, 우선순위 순)

방법을 바꾸는 항목이므로 실행 전 선택을 부탁한다.

1. **측정잡음 $R$ 을 우도 대신 데이터의 정적 구간에서 정한다.** bump 앞 3 s(t < 3 s) 의 $a_x, a_z, v_w, \Delta v_w$ 분산(또는 Allan variance)으로 $R$ 을 고정하고 ML 은 $Q$ 만 추정. 표준 실무(Bar-Shalom §5; Groves §4)이고 $r \to 0$ 퇴화를 막는다. 비용 낮음.
2. **속도 스케줄 휠속 관측** $\ell(V) = \ell_0 + \ell_1 V$: 시변 $C_k$ 의 선형 KF (표준). Stage 2 회귀에서 $\ell_0 \approx 0$, $\ell_1 \approx 0.03$ s.
3. **Bounce 의 $r_v$ 를 라벨 없이 설정**: $a_z$ 스펙트럼을 $1/(2\pi f)^2$ 로 적분한 예상 속도 분산(0.3–10 Hz)을 $r_v$ 로 (Groves §5.3). SUP 0.938 과 비교하면 "한 노브" 의 라벨 의존을 없앨 수 있다.
4. **Optimizer 교차 확인**: 로그 파라미터화 + L-BFGS(수치 gradient) 로 Stage 1–2 ML 을 재현. Powell 국소해인지 확인 (비용 낮음).
5. **회사 확인 요청**: 6D 칩의 pitch 축 규약(nose-down 양수?), bounce 처리(HP 코너·단위·이득 10.3), IMU/6D 장착 위치(높이 포함), 축거, 앞뒤 하중 배분. 이 다섯 값이 있으면 §8-2,3,4 의 ⚠ 가 사라진다.
6. **Half-car 재시도 조건**: 5번 제원이 오거나, 최소한 $l_f \ne l_r$·$f_h, \zeta_h$ 를 free decay 로 고정하고 ML 은 노면·잡음만 추정하는 형태. 그 전에는 Stage 1–2 의 5상태 필터가 실용 기준선이다.
7. **라벨을 쓸 거라면 표준 형태로**: 라벨 NRMSE 로 필터 전체를 fit 하는 대신, ML 로 얻은 필터 위에 출력 이득·오프셋(또는 저차 FIR) 만 라벨로 보정하는 두 단계 — Stage 1 `grav_hi` ML(free gain −49) 에 이득만 보정하면 스케일이 맞는다.

### 9.4 기록: 이 실험에서 바꾸지 않은 것

성능이 낮아도 다음은 건드리지 않았다: shaping filter 의 $(f_p, \zeta_p)$ 고정값, 지연 상수, $g = 9.81$, 출력 이득 $\pm 180/\pi$, 단계 순서, 목적함수 정의(ML/SUP), 탐색 경계. 바꾼 것은 §8-12,13 두 가지이며 성능이 아니라 오류 수정·관측성 이유다.

---

## 10. 참고문헌

- Bar-Shalom, Y., Li, X. R., Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley. §5.4 filter consistency (NIS/NEES).
- Gelb, A. (ed.) (1974). *Applied Optimal Estimation*. MIT Press. ch. 4 shaping filters, complementary filter.
- Gillespie, T. D. (1992). *Fundamentals of Vehicle Dynamics*. SAE. ch. 5 ride (pitch-bounce, wheelbase filtering).
- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed. Artech. §2 lever arm, §5.3 vertical channel.
- Knapp, C. H., Carter, G. C. (1976). The generalized correlation method for estimation of time delay. *IEEE TASSP* 24(4).
- Kwon, B., Kang, D., Yi, K. (2020). Wheelbase preview control of an active suspension with a disturbance-decoupled observer. *Proc. IMechE Part D*.
- Liu, Y., Hozumi, J., Morita, M., Higuchi, A. (2015). Body attitude state estimation using wheel rolling speed variations. *Int. J. Automotive Engineering* 6, 135–142. https://www.jstage.jst.go.jp/article/jsaeijae/6/4/6_20154070/_pdf
- Ljung, L. (1999). *System Identification: Theory for the User*, 2nd ed. Prentice Hall. ch. 13 identifiability.
- Louam, N., Wilson, D. A., Sharp, R. S. (1988). Optimal control of a vehicle suspension incorporating the time delay between front and rear wheel inputs. *Vehicle System Dynamics* 17(6), 317–336.
- Odelson, B. J., Rajamani, M. R., Rawlings, J. B. (2006). A new autocovariance least-squares method for estimating noise covariances. *Automatica* 42(2).
- Rajamani, R. (2012). *Vehicle Dynamics and Control*, 2nd ed. Springer. ch. 11–12 suspensions.
- Raue, A. et al. (2009). Structural and practical identifiability analysis by exploiting the profile likelihood. *Bioinformatics* 25(15).
- Ryu, J., Gerdes, J. C. (2004). Integrating inertial sensors with GPS for vehicle dynamics control. *ASME J. Dyn. Sys. Meas. Control* 126(2).
- Sahlholm, P., Johansson, K. H. (2010). Road grade estimation for look-ahead vehicle control using multiple measurement runs. *Control Engineering Practice* 18(11).
- Särkkä, S., Svensson, L. (2023). *Bayesian Filtering and Smoothing*, 2nd ed. Cambridge. ch. 16 parameter estimation (`../bfs_book_2023_online.pdf`).
- Shumway, R. H., Stoffer, D. S. (1982). An approach to time series smoothing and forecasting using the EM algorithm. *J. Time Series Analysis* 3(4).
- Singer, R. A. (1970). Estimating optimal tracking filter performance for manned maneuvering targets. *IEEE TAES* 6(4).
- Titterton, D., Weston, J. (2004). *Strapdown Inertial Navigation Technology*, 2nd ed. IET.
- Tseng, H. E., Xu, L., Hrovat, D. (2007). Estimation of land vehicle roll and pitch angles. *Vehicle System Dynamics* 45(5), 433–443. https://www.tandfonline.com/doi/abs/10.1080/00423110601169713
