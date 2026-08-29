# Kalman reconstruction methods

## 1. 실험 설정

`Bounce_rate_6D` 없이 허용된 IMU와 wheel speed로 Bounce 및 추가 vehicle state를 복원한다.

- Train: 2,729 episode
- Driver-held-out test: 162 episode
- Sampling rate: 100 Hz
- Classical parameter fitting: train에서 균등하게 고른 최대 300 episode
- Affine calibration: train 전체
- Hybrid early stopping: train driver 중 `조현석` 300 episode

Vertical acceleration과 Bounce 출력은 다음과 같다.

$$a_z = (IMU\_VerAccelVal - 1) 9.81$$

$$Bounce_{pred} = gain v_s + offset$$

`gain`과 `offset`은 train에서만 구한다. 따라서 KF state의 절대 물리 단위를 검증한 실험은 아니다.

## 2. 공통 Kalman filter

모든 모델은 하나의 `StateSpace.filter()`를 사용한다.

$$x_{k|k-1} = A x_{k-1|k-1}$$

$$P_{k|k-1} = A P_{k-1|k-1} A^T + Q$$

$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$

$$x_{k|k} = x_{k|k-1} + K_k (y_k - H x_{k|k-1})$$

Covariance는 Joseph form으로 갱신한다. 연속시간 model은 matrix exponential과 Van Loan 방법으로 이산화한다.

## 3. 1-DOF latent-force models

공통 vehicle model은 다음과 같다.

$$x_v = [z, v]^T$$

$$\dot{z} = v$$

$$\dot{v} = -\omega^2 z - 2 \zeta \omega v + d$$

$$a_z = -\omega^2 z - 2 \zeta \omega v + d + e$$

Branlard et al.은 기계 시스템의 미지 외력을 state에 추가한 acceleration observation model을 제시한다.

- [Branlard et al., 2020, Wind Energy Science](https://wes.copernicus.org/articles/5/1155/2020/)

### 3.1 Random walk

$$x = [z, v, d]^T$$

$$d_dot = w_d$$

Nayek et al.의 Property 1과 Eq. 57은 기존 random-walk input model을 GPLFM의 특수한 경우로 설명한다.

- [Nayek et al., 2019, MSSP](https://doi.org/10.1016/j.ymssp.2019.03.048)

가장 작은 baseline이지만 acceleration-only에서는 일정한 displacement와 이를 상쇄하는 외란을 구분하지 못한다. 저주파 drift가 발생할 수 있다.

- [Naets et al., 2015, MSSP](https://www.sciencedirect.com/science/article/pii/S0888327014002180)

### 3.2 OU

$$x = [z, v, d]^T$$

$$\dot{d} = -\lambda d + w_d$$

Exponential covariance를 1차 Markov state로 표현한 모델이다. Random walk보다 stationary하지만 하나의 correlation time만 표현한다.

- [Nayek et al., Appendix A, Eq. 71](https://arxiv.org/pdf/1904.00093#page=34)

### 3.3 Matern 3/2와 5/2

Matern 3/2는 `x = [z, v, d, d_dot]^T`, Matern 5/2는 `x = [z, v, d, d_dot, d_ddot]^T`를 사용한다. 외란의 smoothness를 단계적으로 높인다.

- [Nayek et al., Appendix A, Eq. 72-73](https://arxiv.org/pdf/1904.00093#page=34)

현재 Matern의 `lambda`가 fitting 상한에 도달하므로 물리적인 road correlation parameter로 해석할 수 없다.

## 4. 2-DOF quarter-car

State와 road input은 다음과 같다.

$$x = [z_s, v_s, z_u, v_u]^T$$

$$u = r$$

$$m_s z_{s,ddot} = -k_s(z_s-z_u)-c_s(v_s-v_u)$$

$$m_u z_{u,ddot} = k_s(z_s-z_u)+c_s(v_s-v_u)-k_t(z_u-r)$$

Observation은 sprung vertical acceleration 하나이다.

$$a_z = [-k_s/m_s, -c_s/m_s, k_s/m_s, c_s/m_s] x + e$$

Road는 state로 직접 추가하지 않고 zero-mean stochastic input으로 처리한다.

$$x_{k+1} = A x_k + G r_k$$

$$Q_x = G q_r G^T$$

이 구조와 posterior road 복원은 Agebjar et al.을 따른다.

- [Agebjar et al., 2025, IEEE FUSION](https://doi.org/10.23919/FUSION65864.2025.11123970)
- [Doumiati et al., 2011, ACC](https://doi.org/10.1109/ACC.2011.5991595)

현재 normalized parameter는 `m_s/m_u = 20/3`으로 고정한다. `k_t/k_s`가 fitting 상한 20에 도달하므로 wheel-hop parameter를 물리값으로 단정할 수 없다.

## 5. Half-car, RTS와 IRI

HC8 state는 다음과 같다.

$$x = [z_s, v_s, theta, theta_dot, z_{u,l}, v_{u,l}, z_{u,r}, v_{u,r}]^T$$

Vertical acceleration과 0.5 Hz causal high-pass를 적용한 lateral acceleration을 관측한다. 좌우가 완전히 대칭인 현재 model에서는 average road가 heave를, differential road가 roll을 독립적으로 구동한다. 따라서 lateral signal을 추가해도 Bounce가 QC2보다 개선되지 않는 것이 구조적으로 정상이다.

Roll 복원은 실패했다 (held-out roll corr median 0.215, `kf_metrics.csv`). 현재 lateral 관측식은 `H[1] = f[3]`, 즉 $a_{lat} = \ddot\phi + e$로 roll 각가속도(rad/s²)를 단위 이득으로 m/s²에 대응시킨다. 레버암 $h$, 중력누설 $g\phi$, 그리고 실제로 $a_{lat}$을 지배하는 코너링 항 $v\,r$(yaw rate)이 모두 빠져 있어 차원과 내용 모두 맞지 않는다. 최소한 $a_{lat} \approx h\ddot\phi + g\phi + v r$ 형태로 바꾸기 전까지 HC8의 roll 출력은 사용하지 않는다.

- [Agebjar et al., 2025, full paper](https://arxiv.org/pdf/2507.12317)

RTS는 filtered state에 backward recursion을 적용하는 offline smoother이다.

$$J_k = P_{k|k} A^T P_{k+1|k}^{-1}$$

$$x_{k|T} = x_{k|k} + J_k (x_{k+1|T} - A x_{k|k})$$

- [Rauch, Tung, and Striebel, 1965](https://doi.org/10.2514/3.3166)
- [Xue et al., 2020, MSSP](https://doi.org/10.1016/j.ymssp.2020.106722)

RTS는 held-out Bounce에서 필터보다 크게 나쁘다. 그러나 그 원인은 model mismatch가 아니라 **target의 위상**이다. 파라미터를 다시 fit하지 않고 필터와 같은 파라미터로 smoother만 돌려도 무너지고, KF와 무관한 zero-phase 적분도 똑같이 어긋난다 (부호 있는 lag: 양수 = 추정이 target보다 늦음).

| 추정 | corr (0-lag) | 부호 있는 lag | lag 정렬 후 corr |
|---|---:|---:|---:|
| QC2 필터 (필터 파라미터) | 0.917 | +10 ms | 0.926 |
| QC2 RTS (같은 필터 파라미터) | 0.692 | +60 ms | 0.807 |
| QC2 RTS (RTS 재적합 파라미터, 표의 값) | 0.758 | +60 ms | — |
| Matern 3/2 필터 / RTS (같은 파라미터) | 0.918 / 0.743 | +20 / +50 ms | 0.928 / 0.835 |
| $a_z$의 인과 2차 BP(0.3–8 Hz) 적분, KF 없음 | 0.879 | +40 ms | 0.932 |
| $a_z$의 zero-phase BP 적분 | 0.735 | +60 ms | 0.825 |

또한 $d(\text{Bounce}_{6D})/dt$가 CAN IMU $a_z$보다 20 ms(p10–p90: 10–50 ms) **선행**한다. 즉 `Bounce_rate_6D`는 물리 속도가 아니라 6D 칩의 인과 HP+적분 처리와 CAN 지연이 얹힌, 자기 위상을 가진 신호이고, 인과 필터의 위상이 우연히 그것과 맞는다. RTS 재적합이 $f = 1.06$ Hz, $\zeta = 0.05$(하한)로 간 것은 optimizer가 위상을 억지로 맞춘 결과다. 같은 이유로 학습 모델은 target 위상을 배울 수 있어 Bi-LSTM(offline)이 이기고, 물리 zero-phase인 RTS는 진다. 자세한 정리는 §10.

Road posterior는 wheel-speed 누적 거리로 0.1 m spatial grid에 보간한다. Golden Car를 80 km/h로 통과시켜 40 m rolling IRI를 계산한다. 실제 road 또는 reference IRI가 없으므로 이는 estimate-only 출력이다.

## 5.5 Pitch-plane half-car (`pitch_hc`)

Target이 `Pitch_rate_6D`인 유일한 모델이다. State, 입력, 관측은 다음과 같다.

$$x = [z_s, v_s, \theta, \dot\theta, z_{u,f}, v_{u,f}, z_{u,r}, v_{u,r}, v]^T, \qquad u = a_x^{IMU}, \qquad y = [v_{w,f}, v_{w,r}, a_z]^T$$

동역학은 앞뒤 축을 가진 4-DOF pitch-plane half-car에 종속도 $v$($\dot v = u$)를 붙인 것이다. 절대 질량·강성 대신 식별 가능한 조합으로 정규화한다: heave $\omega_z, \zeta$, 앞뒤 비대칭 $\epsilon$ ($k_f/m_s = \omega_z^2(1+\epsilon)/2$), pitch 관성비 $j = I_y/(m_s l_f l_r)$, tire 비 $\gamma = k_t/k_s$, 하중이동 이득 $g_u$ ($\ddot\theta \mathrel{+}= g_u u$). 기하 $l_f = 1.45$, $l_r = 1.50$ m와 질량비 $\rho = m_s/m_u = 20$은 고정한다.

- [Rajamani, Vehicle Dynamics and Control, ch. 12](https://doi.org/10.1007/978-1-4614-1433-9)

관측이 이 모델의 새 부분이다. 휠속(축 평균, m/s)을 차체 운동의 관측으로 사용한다.

$$v_{w,f} = v + \beta\,(v_s + l_f\dot\theta - v_{u,f}) - \lambda_f\dot\theta, \qquad v_{w,r} = v + \beta\,(v_s - l_r\dot\theta - v_{u,r}) - \lambda_r\dot\theta$$

$\beta$는 서스펜션 스트로크 속도가 휠 중심 전후 운동으로 새는 기구학 계수, $\lambda$는 pitch 회전 레버다. 이 데이터에서 $v_{w,f}-v_{w,r}$와 $\dot\theta$의 회귀 기울기 −0.2~−0.3 m가 실측된다 (`docs/claude_explanation/pitch_rate_exploration.md`). 세 번째 관측은 $a_z = \ddot z_s$. Road는 QC2/HC8과 동일하게 앞뒤 독립 zero-mean 입력으로 두며, 앞바퀴 노면을 wheelbase 지연으로 뒷바퀴에 재생하는 커플링은 없다.

외란 변형은 1-DOF latent-force 가족(§3)과 같은 블록을 $\ddot\theta$ 행에 증강한다 (`models.py`의 `disturbance()` + `augment()` 공용).

$$\ddot\theta \mathrel{+}= d, \qquad \text{ou: } \dot d = -\lambda_d d + w, \qquad \text{osc2: } \ddot d + 2\zeta_d\omega_d\dot d + \omega_d^2 d = w, \qquad \mathbb{E}[d^2] = \sigma_d^2$$

- [Nayek et al., 2019, MSSP](https://doi.org/10.1016/j.ymssp.2019.03.048)

osc2는 $\zeta_d = 1$이면 Matérn 3/2, $\zeta_d < 1$이면 quasi-periodic latent force다. 관측성은 cascade PBH로 확인된다: 외란 극이 plant의 $d \to y$ transmission zero와 겹치지 않으면 관측 가능한데, 이 plant의 구조적 공통 zero는 $s = 0$ 하나뿐이므로(상수 모멘트는 정적 트림만 바꿔 속도·가속도 센서에 안 보임) $\omega_d > 0$ 또는 $\lambda_d > 0$이면 성립한다. 같은 이유로 random-walk 모멘트는 불가하다.

노면 변형은 두 가지다. `pitch_road*`는 노면 수직 변위를 명시적 상태로 둔다: $\dot r_f = w_r$, $\dot r_r = w_r$ (random-walk 변위, ISO 8608의 $f^{-2}$ 변위 스펙트럼에 대응), tire force $F_{t,i} = k_{t,i}(z_{u,i} - r_i)$. `pitch_delay*`는 $r_f$ 하나만 상태로 두고 뒷바퀴 노면을 wheelbase 지연 재생으로 결정한다: $r_r(t) = \hat r_f(t - L/v)$, 휠속 누적 거리 기반 인덱스로 필터 안에서 기지 입력처럼 재생(공분산 무시 근사). 상수 노면 방향은 $s=0$ blocking zero 때문에 비관측이므로 $\hat r$은 절대 높이가 아닌 상대 프로파일(estimate-only)이다. 독립 RW에서는 앞뒤 반대부호 상수(정적 pitch 트림과 등가) 방향도 비관측이라 $\hat r_f, \hat r_r$이 서로 반대로 drift할 수 있는데(state plot에서 관찰됨), 지연 연결은 이 자유도를 구조적으로 제거한다.

식별은 train 300 episode, calibrated NRMSE, Powell maxiter 300이며 두 가지 보강을 둔다: (i) scipy의 bounded Powell이 방문한 최적점보다 나쁜 점을 반환하는 비단조 결함이 있어 objective가 최적 방문점을 직접 추적해 반환하고(비유한 cost는 1e6 벌점), (ii) 외란 변형(_ou/_osc)은 cold start와 함께 parent 모델 fit + $\sigma_d^2 \to 0$ 퇴화점 warm start를 돌려 낮은 loss를 채택한다 — warm의 시작 loss가 parent와 같으므로 변형이 parent보다 나빠질 수 없다. Held-out 결과:

| Method | Corr median [p10, p90] | RMSE median | Median abs lag |
|---|---:|---:|---:|
| pitch_hc | 0.718 [0.43, 0.82] | 2.94 | 30 ms |
| pitch_hc_ou | 0.793 [0.59, 0.87] | 2.51 | 20 ms |
| pitch_hc_osc | 0.879 [0.67, 0.94] | 2.06 | 10 ms |
| pitch_road | 0.723 [0.45, 0.83] | 2.92 | 30 ms |
| pitch_road_osc | **0.905** [0.73, 0.95] | 1.82 | 10 ms |
| pitch_delay | 0.722 [0.50, 0.84] | 2.89 | 30 ms |
| pitch_delay_osc | 0.807 [0.62, 0.89] | 2.50 | 20 ms |

관찰: (1) osc2 외란이 지배적 요소다 — 어느 노면 처리 위에서든 parent 대비 +0.10~0.18을 더하며, $\omega_d \approx 9$–11 rad/s (1.4–1.7 Hz), $\zeta_d$ 소로 식별되어 **1–2 Hz 노면 pitch 여기 대역의 quasi-periodic 모멘트**를 흡수한다. (2) 노면 상태 자체(백색 가속 vs RW 변위 vs 지연 연결)는 단독으로는 0.72 수준에서 갈리지 않는다. (3) 물리적으로 올바른 지연 연결(delay_osc 0.807)이 유연한 독립 노면(road_osc 0.905)보다 오히려 낮다 — 좌우 평균 휠속, 앞뒤 트랙 차이, 재생 근사 등 지연 제약이 데이터와 완전히 맞지 않는 것으로 보이며, 독립 노면 + osc2의 여분 자유도가 이를 흡수한다.

### 2단계 확장 (`pitch2`: 토크 입력, $a_x$ 관측, 유색 노면)

`pitch_road_osc`를 anchor로 세 단계를 적층했다. 각 단계는 parent fit + 퇴화점 warm start로 식별하므로 parent보다 나빠질 수 없다.

**pitch_tq** — 모터 토크(MCU est)를 입력에 추가: anti-squat/anti-lift 직접 모멘트 $\ddot\theta \mathrel{+}= g_f\tau_f + g_r\tau_r$ (Gillespie ch. 7), 구동 슬립 feedthrough $v_{w,i} \mathrel{+}= s_i\tau_i$. 식별: $g_f \approx 0$, $g_r = -1.8\times10^{-3}$ (뒤축 지배), $s \approx 7\times10^{-4}$ m/s/Nm.

**pitch_ax** — $a_x$를 입력에서 관측으로 전환: 잠재 종가속 $a_b$ (OU, 토크 구동 $\dot a_b = -\lambda_a(a_b - g_v(\tau_f + \tau_r))$), 구배 $\gamma$와 bias $b_x$ (RW), $a_x^{IMU} = a_b + g(\theta + \gamma) + b_x$. 중력누설이 $\theta$ 저주파의 오염원에서 정보원으로 바뀐다는 것이 설계 의도다 (road-grade 추정 문헌의 표준 구조: Lingman & Schmidtbauer 2002 VSD; Sebsadji et al. 2008 ACC). 식별 $\tau_a = 1/\lambda_a \approx 49$ ms — 별도 신호 분석에서 측정한 휠→차체 지연 60–80 ms와 정합. 다만 §5.6에서 보듯 적합된 $\hat\theta$가 거울상·축소 스케일이라 $g\theta$ 항은 실제 fit에서 설계대로 작동하지 않으며, $r_{a_x} = e^{-8}$(하한)과 torque-driven $a_b$가 $a_x$를 대신 설명한다. 개선 자체는 유효하나 메커니즘 해석은 보류한다.

**pitch_axou** — 노면 RW를 OU로 유색화: $\lambda_{road}$가 탐색 하한(0.01 s⁻¹, $\tau$ 100 s ≫ 에피소드)으로 수렴해 사실상 RW로 되돌아감 — 데이터가 ISO 8608형 $f^{-2}$ 스펙트럼을 지지.

| Method | Corr | Corr 0.3–1 Hz | Corr 1–3 Hz | Amp ratio | RMSE |
|---|---:|---:|---:|---:|---:|
| pitch_road_osc (anchor) | 0.905 | 0.866 | 0.952 | 0.82 | 1.82 |
| pitch_tq | 0.912 | 0.880 | 0.959 | 0.84 | 1.70 |
| pitch_ax | **0.927** | **0.913** | 0.964 | 0.87 | **1.58** |
| pitch_axou | 0.926 | 0.912 | 0.964 | 0.86 | 1.60 |

$a_x$ 관측 전환이 겨냥대로 저주파 대역(0.866 → 0.913)을 가장 크게 개선했다. 최종 `pitch_ax` 기준 비인과 선형 상한(FIR corr 0.96) 대비 격차 0.033. 전 모델에서 error–amplitude correlation이 0.5 이상으로 오차가 큰 진폭 구간에 집중되어 있어(`pitch2_metric_grid.png`), 비선형 감쇠(EKF 또는 스트로크 부호 스케줄드 KF)가 다음 후보다.

**탐색 경계에 도달한 파라미터** (`pitch_metrics.csv`, `pitch2_metrics.csv`; 문서에 $\lambda_{road}$만 적혀 있었으나 실제로는 다음 전부):

| Model | 상한 | 하한 |
|---|---|---|
| pitch_hc | $\gamma = 20$ | — |
| pitch_hc_ou | — | $\zeta = 0.05$, $j = 0.3$, $\gamma = 3$ |
| pitch_road | — | $\gamma = 3$ |
| pitch_road_osc | $f = 4.0$ Hz, $j = 3$, $\gamma = 20$ | — |
| pitch_delay / pitch_delay_osc | $\gamma = 20$ (osc) | $\zeta = 0.05$, $j = 0.3$ |
| pitch_tq | $f = 4.0$, $j = 3$, $\gamma = 20$ | $q_{body} = e^{-16}$ |
| pitch_ax / pitch_axou / pitch_eps | $f = 4.0$, $j = 3$, $\gamma = 20$, $g_v = 0.05$ | $q_{long} = e^{-12}$, $q_{grade} = q_{bias} = e^{-16}$, $r_{a_x} = e^{-8}$; axou $\lambda_{road} = 0.01$; eps $\varepsilon_c = -0.9$ |

$q_{grade} = q_{bias} \to 0$은 구배·bias random walk가 상수로 퇴화했다는 뜻이고, $f = 4$ Hz heave와 $j = 3$은 승용차 물리값(1–1.5 Hz, $j \approx 0.8$–1.0)이 아니다. 이 값들은 §5.6의 평평한 손실 계곡이 경계에서 끝난 자리이지 식별 결과가 아니다.

## 5.6 Pitch 파라미터의 비식별성

**손실의 불변성.** 목적함수는 affine calibration을 포함한다.

$$J(p) = \min_{g,c}\ \frac{\big\|\, g\,\hat{\dot\theta}(p) + c - \dot\theta^{6D} \big\|_2}{\sigma_{6D}}$$

따라서 어떤 $p'$이 $\hat{\dot\theta}(p') = s\,\hat{\dot\theta}(p)$ ($s \ne 0$, 음수 포함)를 만들면 $g' = g/s$로 $J(p') = J(p)$다. $y = abx$에서 $ab$만 식별되는 상황과 같고, 출력 파형과 corr 비교에는 무해하다. 문제는 이 모델이 $\theta \to s\theta$를 거의 자유롭게 실현할 수 있다는 것이다. $\theta$가 데이터와 연결되는 경로를 보면:

| 경로 | $\theta$ 스케일·부호 고정 | 이유 |
|---|:---:|---|
| 휠속 $v_{w,f} = v + \beta(\cdot) + (\beta l_f - \lambda_f)\dot\theta$, $v_{w,r} = \cdots + (-\beta l_r - \lambda_r)\dot\theta$ | ✗ | $\beta, \lambda_f, \lambda_r$ 자유, 부호 무제한 → $\lambda_f' = \beta l_f - (\beta l_f - \lambda_f)/s$ 로 어떤 $s$든 흡수 |
| 피치 관성 $\ddot\theta = \dfrac{-l_f F_f + l_r F_r}{j\, l_f l_r}$ | ✗ | $j' \approx j/s$ 가 응답 스케일을 조정, 고유진동수 변화는 $\omega, \varepsilon$ 이 보상 |
| heave–pitch 커플링 $\ddot z_s \ni -\dfrac{\omega^2}{2}\big[\varepsilon(l_f + l_r) + (l_f - l_r)\big]\theta = -\dfrac{\omega^2}{2}(2.95\,\varepsilon - 0.05)\,\theta$ | 거의 ✗ | $\varepsilon \in [-0.9, 0.9]$ 로 크기·부호 모두 조절. $s = -1$ 은 $\varepsilon' = -\varepsilon + 0.034$ |
| $g_u$, $g_f$, $g_r$ (하중이동·토크 모멘트) | ✗ | 부호 자유 |
| 잠재 외란 $d$ (osc2, $\sigma_d^2$ 자유) | ✗ | 남는 $\ddot\theta$ 전부 흡수 |
| $a_x = a_b + g\,\theta + g\,\gamma + b_x$ | 원리상 ○, 실제 ✗ | 유일한 고정 부호·고정 계수 항. 그러나 $a_b$(OU, 분산 자유)가 같은 채널을 설명하고 $r_{a_x} \to 0$ 이라 $\theta$ 를 못 박지 못함 |

결과적으로 손실 지형에는 $(j, \varepsilon, \omega, \beta, \lambda_f, \lambda_r, \sigma_d^2, g_u)$ 를 따라 이어지는 **근사적으로 평평한 계곡**과 $s = -1$ 에 대응하는 **거울상 해**가 있다. 계곡이 정확한 대칭이 아니라 고정 기하 $l_f, l_r$ 때문에 살짝 기울어져 있어서, optimizer는 미세한 이득을 좇아 경계까지 굴러간다(위 표). 거울상은 이산 대칭이라 계곡과 달리 시작한 쪽에서 넘어가지 못한다.

**데이터의 증거.**

- Calibration gain: `pitch_road_osc` $-639$, `pitch_ax` $-456$, `pitch_eps` $-376$. target이 deg/s이면 단위 변환은 $+180/\pi = +57.3$ 이어야 하므로 $\hat{\dot\theta}$ 는 **부호가 뒤집히고 7–11배 작다** ($\hat{\dot\theta} \approx -0.09\,\dot\theta$).
- 휠속 레버 front − rear $= \beta(l_f + l_r) - \lambda_f + \lambda_r$: `pitch_road_osc` $+5.09$ m, `pitch_ax` $+3.81$ m. §5.5 서두에서 근거로 인용한 실측 회귀 기울기는 $-0.2$~$-0.3$ m. 부호가 반대이고 20배 크다. $-0.25 / 5.09 \approx -0.05$ 로 gain의 크기와 정합한다.
- 거울상에 앉은 원인: `model_spec` 시작점($\beta = 0.13$, $\lambda_f = \lambda_r = 0.55$)의 레버가 $+0.38$ m 로 이미 실측과 반대 부호다.
- 외란이 플랜트를 대체: `pitch_road_osc` 의 플랜트 피치 고유진동수는 $\omega_\theta^2 \approx (a_f l_f^2 + a_r l_r^2)/(j l_f l_r) \approx 216$ → 2.3 Hz (unsprung 고정 근사)인데 실측 pitch PSD 피크는 ≈ 1 Hz 이고, $d$ 는 $\omega_d = 9.16$ rad/s $= 1.46$ Hz, $\zeta_d = 0.23$ 으로 식별됐다. 차체 피치 모드를 half-car가 아니라 **잠재 외란이 모델링**하고 플랜트는 통과 경로가 된 것이다. osc2 의 $+0.10$~$0.18$ 개선은 "노면 pitch 여기 대역의 quasi-periodic 모멘트"보다 이 해석이 더 그럴듯하다.

**무엇이 유효하고 무엇이 아닌가.** 출력 모양만 보는 변형 간 corr 비교(road vs delay vs osc, tq/ax 적층)는 비식별성과 무관하게 성립한다. 성립하지 않는 것은 (i) $j, \varepsilon, \gamma, \beta, \lambda, g_u, g_f, g_r$ 의 물리 해석, (ii) 고정 부호 물리 항($g\theta$, wheelbase 지연, $l_f/l_r$ 기하)이 물리대로 작동했다는 서사다. $\tau_a$ 처럼 시간상수(스케일 무관)만 상대적으로 안전하다.

**식별성 회복** (비용 낮은 순). 대수적으로 하나의 파라미터로 뭉치는 것이 아니라 자유도 쪽을 외부 정보로 못 박는 것이다.

1. 휠속 레버를 실측으로 고정 ($\beta = 0$, $\lambda_f - \lambda_r = -0.25$ m 수준). 관측식이 $\dot\theta$ 의 스케일·부호를 핀하므로 gain 이 $+57$ 근처로 오는지가 즉시 검증.
2. Pitch 만 calibration 을 빼고 $g = 57.3$ 고정 fit. $a_x$ 의 $g\theta$ 항이 비로소 제약으로 작동.
3. $j, \varepsilon$ 을 차량 제원으로 고정 ($j \approx 0.8$–1.0, $\varepsilon \approx \pm 0.1$).
4. 합성 검증: 알려진 파라미터로 시뮬레이션 → 재fit → 복원되는 파라미터 확인 (§3 Matern 의 bound sensitivity 와 같은 취지).
5. 1–3 후에도 osc2 가 필요하면 그때 "노면 여기 모멘트" 해석이 정당화된다.
6. 토크 채널 매핑 (`estimate_pitch`: Mg2 → 앞, Mg1 → 뒤) 은 근거 문서가 없다. "$g_r$ 뒤축 지배" 는 이 가정에 걸려 있으므로 확인 전까지 보류.

## 6. 실제 데이터 기반 1-DOF KF + LSTM

Hybrid baseline은 train objective가 1-DOF 후보 중 가장 낮은 Matern 3/2이다. Test 성능을 보고 baseline을 선택하지 않았다.

$$x_k = [z_k, v_k, d_k, d_{dot,k}]^T$$

KF의 one-step prediction과 innovation은 다음과 같다.

$$x_{k|k-1} = A x_{k-1|k-1}$$

$$innovation_k = a_{z,k} - H x_{k|k-1}$$

Causal LSTM 입력은 현재와 과거의 다음 다섯 값이다.

$$feature_k = [z_k, v_k, d_k, d_{dot,k}, innovation_k]$$

KF의 calibrated Bounce와 학습 target은 다음과 같다.

$$Bounce_{KF,k} = gain v_k + offset$$

$$residual_k = Bounce_k - Bounce_{KF,k}$$

2-layer causal LSTM은 `residual_k`만 출력한다.

$$Bounce_{hybrid,k} = Bounce_{KF,k} + LSTM(feature_{0:k})$$

이 구조에는 합성 data가 없다. LSTM 입력도 모두 `IMU_VerAccelVal`에서 online KF로 계산하며 wheel speed나 다른 sensor를 추가하지 않았다. Bi-directional layer, future window, centered convolution도 없으므로 sample 기준 online causal method이다.

Data 분리는 다음과 같다.

- KF parameter fitting, affine calibration, LSTM fitting: 실제 train 2,429 episode
- Early stopping: 실제 validation driver `조현석` 300 episode
- 최종 평가: 기존 driver-held-out test 162 episode

Validation driver는 non-test driver 중 전체 train의 10%에 가장 가까운 episode 수를 가진 driver로 결정한다. KF parameter도 validation을 제외하고 다시 fitting하므로 validation Bounce label이 KF feature 생성에 사용되지 않는다.

이 방법은 full state를 LSTM으로 복원하는 기존 2-DOF LSTM-KF가 아니다. KF가 physics feature를 만들고 LSTM이 최종 Bounce residual을 보정하는 현재 데이터용 hybrid이다. 따라서 LSTM 개선이 `z`, `d`, `d_dot`의 정확도 개선을 뜻하지는 않는다.

## 7. 실제 데이터 기반 model-free models

Model-free 실험은 KF state, vehicle parameter, road model을 전혀 사용하지 않는다. 각 episode의 허용 센서 13개를 직접 입력하고 `Bounce_rate_6D` 한 개를 sequence-to-sequence regression으로 복원한다. 입력과 target normalization 통계는 train 2,429 episode에서만 계산하며, validation과 test에는 train 통계를 그대로 적용한다.

### 7.1 Online

- LSTM: 2-layer unidirectional LSTM, hidden size 64
- GRU: 2-layer unidirectional GRU, hidden size 64
- Causal Transformer: projection size 64, 4 heads, 3 encoder layers, learned position embedding, upper-triangular attention mask

세 모델의 시점 `k` 출력은 입력 `0:k`만 사용한다. 구현 test에서는 시점 `k+1` 이후 입력을 무작위로 바꾼 전후의 `0:k` 출력을 비교하며, 세 모델 모두 최대 절대 오차 0을 기록했다.

### 7.2 Offline

- Bi-LSTM: 2-layer bidirectional LSTM, hidden size 64
- 1-D U-Net: 2-level encoder-decoder, skip connection, symmetric convolution
- Full-attention Transformer: causal mask 없이 episode 전체에 attention

Offline model은 현재 시점의 앞뒤 문맥을 모두 사용할 수 있다. 따라서 online model과 수치 비교는 가능하지만 동일한 배포 조건의 비교는 아니다.

### 7.3 학습과 평가

- Loss: normalized `Bounce_rate_6D`의 sample-wise MSE
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001
- Scheduler: validation MSE 기준 ReduceLROnPlateau
- 최대 30 epoch, validation이 8 epoch 연속 개선되지 않으면 종료
- Seed: 42
- Train: 2,429 episode
- Validation: `조현석` 300 episode
- Driver-held-out test: 162 episode

Checkpoint 선택과 normalization에는 test label을 사용하지 않는다. 현재 결과는 한 seed와 한 split의 구현 검증 결과이며 모델별 parameter 수를 동일하게 맞춘 capacity-controlled 비교는 아니다.

## 8. Held-out 결과

| Method | Mode | Corr median | RMSE median | Median abs lag |
|---|---|---:|---:|---:|
| RW | Online | 0.9123 | 0.2763 | 10 ms |
| OU | Online | 0.9141 | 0.2680 | 20 ms |
| Matern 3/2 | Online | 0.9185 | 0.2587 | 10 ms |
| Matern 5/2 | Online | 0.9198 | 0.2604 | 10 ms |
| QC2 | Online | 0.9166 | 0.2614 | 10 ms |
| HC8 | Online | 0.9166 | 0.2614 | 10 ms |
| QC2 + RTS | Offline | 0.7581 | 0.4173 | 60 ms |
| HC8 + RTS | Offline | 0.7581 | 0.4173 | 60 ms |
| Matern 3/2, hybrid split | Online | 0.9179 | 0.2591 | 10 ms |
| Matern 3/2 + causal LSTM | Online | 0.9843 | 0.1178 | 0 ms |
| Kinematic KF ($a_z$ BP 0.2–25 Hz, 상수가속 모델) | Online | 0.8535 | 0.3454 | 40 ms |
| LSTM, $a_z$ 단독 1채널 | Online | 0.9757 | 0.1398 | 0 ms |
| Kinematic KF + LSTM 속도 융합 | Online | 0.9748 | 0.1451 | 10 ms |
| LSTM, model-free | Online | 0.9908 | 0.0900 | 0 ms |
| GRU, model-free | Online | 0.9918 | 0.0841 | 0 ms |
| Causal Transformer, model-free | Online | 0.9863 | 0.1081 | 0 ms |
| Bi-LSTM, model-free | Offline | 0.9927 | 0.0790 | 0 ms |
| 1-D U-Net, model-free | Offline | 0.9765 | 0.1399 | 0 ms |
| Full-attention Transformer, model-free | Offline | 0.9819 | 0.1204 | 0 ms |

같은 KF parameter와 calibration을 사용하는 직접 비교에서 causal LSTM residual correction이 correlation과 RMSE를 모두 크게 개선했다. 다만 한 seed와 한 driver-held-out split의 결과이므로 여러 seed와 split에서 재현되기 전에는 일반화된 성능 향상으로 단정하지 않는다.

같은 $a_z$ 한 채널을 KF 없이 LSTM에 직접 넣은 ablation (`kinematic_lstm_kf_metrics.csv`, `imu_lstm`)은 0.9757 로, KF state를 feature로 준 hybrid 0.9843 보다 낮다. 즉 hybrid의 이득 중 일부는 KF feature 자체에서 온다. 반대로 LSTM 속도를 kinematic KF에 관측으로 융합한 `lstm_kf` 는 process variance 가 $6.6 \times 10^6$ 으로 퇴화하여 (KF가 LSTM 출력을 그대로 통과) $a_z$ 단독 LSTM 보다 낮다. 이 융합은 현재 구성에서 의미가 없다. 또한 `lstm_kf` 의 velocity variance 와 process variance 는 validation label 로 맞춘다.

RW·OU·Matern·QC2 가 0.912–0.920 에 몰리는 이유: 1-DOF 전부에서 $\sigma_d^2 = e^{5}$(상한), $q_v = e^{-12}$(하한) 이라 $d$ 가 $a_z$ 를 통째로 흡수하고, KF 는 사실상 인과 2차 bandpass 적분기다. 손으로 만든 인과 BP 적분(0.879, lag 정렬 후 0.932)과의 차이는 대부분 위상 튜닝이다. 0.005 수준의 외란 모델 간 차이는 한 split·한 seed 에서 의미가 없다. 표의 classical 행(rw, matern32 의 $f = 3.000$)은 $f \le 3$ Hz 경계 시절의 결과이며 현재 `model_spec` 은 $(0.5, 8)$ 이다.

## 9. 핵심 한계

- 모든 physical parameter가 Bounce label에 대한 supervised objective로 fitting되었다.
- Vehicle parameter와 road input을 단일 acceleration만으로 독립적으로 식별했다고 볼 수 없다.
- Road, IRI, `z_s`, `z_u`, `v_u`의 실제 ground truth가 없다.
- HC8은 좌우 대칭, centered IMU, linear suspension을 가정한다. Lateral 관측식이 $\ddot\phi$ 단위 이득이라 roll 은 실패했다 (§5).
- RTS 의 성능 저하는 recursion 오류도 model mismatch 도 아니라 target 의 위상 특성 때문이다 (§5, §10). Zero-phase 추정을 이 target 으로 평가하면 구조적으로 불리하다.
- `Bounce_rate_6D` 는 IMU 와 위상 정렬이 안 된 파생 신호다 ($d(\text{Bounce})/dt$ 가 IMU $a_z$ 보다 20 ms 선행). 처리 체인이 확인되기 전까지 "lag" 는 추정기 지연이 아니라 target 과의 상대 위상으로 읽어야 한다.
- LSTM은 Bounce residual을 직접 학습하므로 latent state가 더 정확해졌다는 의미가 아니다.
- Hybrid의 Matern parameter 중 일부가 탐색 경계에 도달하므로 physical parameter가 아니라 supervised filter로 해석해야 한다. 1-DOF 전부에서 $\sigma_d^2$ 상한, $q_v$ 하한.
- Pitch 모델은 $\theta$ 의 스케일·부호가 비식별이다 (§5.6). 변형 간 출력 비교만 유효하고 파라미터 해석은 무효. 대부분의 플랜트 파라미터가 경계에 있다.
- `fit_model` 의 best-visited 보정이 `hc_objective`, `rts_objective`, `hc_smooth_objective`, calibration ablation 에는 없다. 이 fit 들은 Powell 의 비단조 결함에 그대로 노출된다.
- 큰 개선 폭은 seed 반복, driver별 결과, model-free LSTM과의 ablation으로 다시 확인해야 한다.
- 하나의 driver-held-out split 결과이므로 split 안정성은 별도로 확인해야 한다.
- Model-free 결과는 한 seed이며 architecture별 parameter 수가 약 40,000개부터 166,000개까지 다르다.
- Offline model은 미래 sample을 사용하므로 실시간 성능으로 해석할 수 없다.
- 현재 model-free target은 Bounce 하나이며 latent vehicle state를 복원하지 않는다.

## 10. Lag 지표와 인과 추정기의 지연

### 10.1 지표 정의

`metrics()` 는 평균을 뺀 target $b$ 와 복원 $\hat b$ 의 cross-correlation 을 $|\tau| \le 0.5$ s 안에서 최대화하는 시차를 찾고 그 절댓값을 보고한다.

$$\tau^\ast = \arg\max_{|\tau| \le 50} \sum_t \big(b_t - \bar b\big)\big(\hat b_{t+\tau} - \bar{\hat b}\big), \qquad \text{lag} = |\tau^\ast| / f_s$$

Correlation 은 zero-lag 에서 계산되므로 파형이 완벽해도 시간이 어긋나면 떨어진다. lag 는 그 오차 중 타이밍 성분만 분리해 보려는 진단이며, cross-correlation 정점으로 시간차를 재는 것은 time-delay estimation 의 표준 방법이다 (Knapp & Carter 1976, GCC). 이 lab 에서는 **보고만 하고 어떤 후보정도 하지 않는다**. 온라인에서는 미래 샘플이 필요하므로 후보정이 불가능하기도 하다.

현재 구현의 한계:

- `abs()` 로 부호를 버린다. 필터는 지연만 가능하다는 전제였지만, RTS 처럼 "어느 쪽으로" 어긋났는지가 진단의 핵심일 때 정보가 사라진다. 부호 있는 lag 를 같이 보고해야 한다.
- 1-sample (10 ms) 양자화. median 이 0/10/60 ms 로만 나온다. 정점 주변 3점 포물선 보간으로 sub-sample 추정이 가능하다.
- 단일 스칼라. 필터 지연은 주파수마다 다른데 (HP 는 위상 앞섬, 적분은 $-90°$) cross-correlation 정점은 에너지가 큰 대역 (1–3 Hz) 의 위상차를 대표할 뿐이다. §5 의 "lag 정렬 후에도 RTS 가 0.81 에 머문다" 가 그 증거로, 한 값으로 못 맞추는 주파수 의존 위상차가 남는다. `waveform_metrics` 의 대역별 corr 을 같이 본다.

### 10.2 인과 필터에는 지연이 있다

KF 는 예측과 현재 측정을 합치지만 $\hat x_{k|k}$ 는 $y_{1:k}$ 만 쓴다. 위치를 직접 관측하는 정상상태 스칼라 KF 는

$$\hat x_k = \hat x_{k-1} + \alpha\,(y_k - \hat x_{k-1}) = (1-\alpha)\,\hat x_{k-1} + \alpha\, y_k, \qquad G(z) = \frac{\alpha}{1 - (1-\alpha) z^{-1}}$$

의 지수평활기이고 저주파 group delay 는 $(1-\alpha)/\alpha$ 샘플이다. $\alpha$ 는 정상상태 Riccati 해 $\alpha = P^-/(P^- + R)$ 로 $Q/R$ 이 정한다. $R \to 0$ 이면 $\alpha \to 1$, 지연 0 이지만 잡음이 그대로 통과하고, $R$ 이 크면 예측을 믿어 부드럽지만 늦다. **잡음 억제와 지연은 인과 필터에서 한 쌍의 트레이드오프**이고 KF 는 그 트레이드오프를 모델 기준으로 최적으로 정할 뿐 지연을 없애지 않는다. 정상상태 KF 는 Wiener 필터와 같고, 인과 LTI 필터는 group delay 를 가진다.

이 lab 에서는 측정이 가속도, 원하는 것이 속도이며 노면 외란 $d$ 가 랜덤 상태라 예측이 불가능하다. $d$ 가 바뀌면 innovation 이 몇 스텝 쌓여야 $\hat d, \hat v$ 가 따라가고, 그 과도응답이 lag 다. 지연을 줄이는 길은 (a) 측정을 더 믿기 (잡음↑), (b) 모델이 미래를 예측하게 하기 (기지 입력 $u$; wheelbase 지연 모델이 앞바퀴 노면으로 뒷바퀴를 예측하는 것이 이것), (c) 미래 샘플 쓰기 (RTS, Bi-LSTM), (d) target 의 위상 자체를 배우기 (LSTM 의 0 ms) 뿐이다.

### 10.3 이 lab 에서 측정되는 lag 의 의미

Fit 목적함수가 zero-lag NRMSE 이므로 optimizer 는 이미 $Q, R, \zeta, f$ 를 움직여 target 의 위상에 맞도록 필터를 골랐고, 남은 10–20 ms 는 그 트레이드오프의 잔여다. 그런데 target 은 물리 속도가 아니라 자기 위상을 가진 파생 신호라서

$$\tau_{\text{measured}} \;\approx\; \tau_{\text{KF}} \;+\; \tau_{\text{sensor}} \;-\; \tau_{\text{target}}$$

로 KF 고유 지연, CAN IMU 지연, target 처리 체인의 위상 (인과 HP 는 앞섬) 이 섞여 있다. RTS 는 $\tau_{\text{KF}}$ 를 없앴는데도 $\tau_{\text{target}}$ 이 남아 오히려 60 ms 어긋나 보이고, zero-phase BP 적분도 같은 60 ms 를 보인다 (§5 표). LSTM 의 0 ms 는 target 이 센서의 인과 함수이므로 인과 모델이 그 위상을 그대로 배울 수 있다는 뜻이지 지연이 없는 물리 추정이라는 뜻이 아니다. Target 의 처리 체인을 확인하기 전까지 online/offline 물리 추정기의 lag 비교는 추정 품질이 아니라 위상 정합을 재는 것으로 읽어야 한다.

## 11. 결과 파일

- `outputs/kf_metrics.csv`: classical KF held-out metric
- `outputs/kf_models.png`: waveform과 correlation 분포
- `outputs/kf_states_median.png`: 1-DOF와 QC2 state
- `outputs/kf_extensions_median.png`: HC8와 RTS state
- `outputs/qc2_road_posterior.npz`, `outputs/hc8_road_posterior.npz`: road posterior
- `outputs/kf_spatial_iri.npz`, `outputs/kf_iri_summary.csv`: spatial road와 IRI
- `outputs/pitch_metrics.csv`, `outputs/pitch_parameters.npz`: pitch_hc held-out metric과 parameter
- `outputs/pitch_models.png`, `outputs/pitch_states_median.png`: pitch waveform과 state
- `outputs/matern32_lstm_metrics.csv`: Matern 3/2와 hybrid held-out 비교
- `outputs/matern32_lstm_models.png`: worst, median, best waveform과 correlation 분포
- `outputs/matern32_lstm_median.png`: Bounce, residual correction, innovation
- `outputs/matern32_lstm_predictions.npz`: held-out baseline, hybrid, correction
- `outputs/matern32_lstm.pt`: network, normalization, Matern parameter와 validation 정보
- `outputs/model_free_metrics.csv`: 6개 model의 parameter 수, validation과 held-out metric
- `outputs/model_free_models.png`: online/offline median waveform과 correlation 분포
- `outputs/model_free_predictions.npz`: held-out prediction
- `outputs/model_free.pt`: 6개 network weight, normalization과 split 정보
- `outputs/pitch2_metrics.csv`, `outputs/pitch2_parameters.npz`, `outputs/pitch2_*.png`: pitch 2단계 (tq/ax/axou/eps)
- `outputs/oscillator_metrics.csv`: 2-state oscillator vs RW ablation
- `outputs/calibration_ablation_metrics.csv`: affine calibration 유무 ablation
- `outputs/matern32_bound_sensitivity.{csv,json}`: $f$, $\lambda$ 상한 확장 sweep
- `outputs/kinematic_lstm_kf_metrics.csv`: kinematic KF, $a_z$ 단독 LSTM, 융합
- `outputs/ou_lstm_*`, `outputs/rw_lstm_*`: OU/RW 기반 hybrid
