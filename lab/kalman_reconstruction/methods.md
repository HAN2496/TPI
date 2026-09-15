# Kalman reconstruction methods

## 1. 실험 설정

`Bounce_rate_6D` 없이 허용된 IMU와 wheel speed로 Bounce 및 추가 vehicle state를 복원한다.

- Train: 2,729 episode
- Driver-held-out test: 162 episode
- Sampling rate: 100 Hz
- Classical parameter fitting: train에서 균등하게 고른 최대 300 episode
- Affine calibration: train 전체 (Bounce 는 gain·offset 자유, Pitch 는 gain $= 180/\pi$ 고정·offset 만 추정)
- Hybrid early stopping: train driver 중 `조현석` 300 episode

세 target은 허용 센서에 직접 측정값이 있는지에 따라 문제의 성격이 다르다.

| Target | 직접 센서 (CAN IMU) | 문제의 성격 |
|---|---|---|
| `Roll_rate_6D` | 있음 (`IMU_RollRtVal`) | 이득·오프셋·위상 보정. 차량 모델 불필요 |
| `Bounce_rate_6D` | 없음 ($a_z$만) | 적분 + 필터. 1-DOF로 충분 |
| `Pitch_rate_6D` | **없음** (pitch 자이로 없음) | 휠속·$a_z$·$a_x$ 융합. 차량 모델이 실제로 필요한 유일한 target |

Vertical acceleration과 Bounce 출력은 다음과 같다.

$$a_z = (IMU\_VerAccelVal - 1) 9.81$$

$$\hat b = g\,\hat v_s + c$$

`gain`과 `offset`은 train에서만 구한다. 따라서 KF state의 절대 물리 단위를 검증한 실험은 아니다.

## 2. 공통 Kalman filter

모든 모델은 하나의 `StateSpace.filter()`를 사용한다.

$$x_{k|k-1} = A x_{k-1|k-1}$$

$$P_{k|k-1} = A P_{k-1|k-1} A^T + Q$$

$$K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}$$

$$x_{k|k} = x_{k|k-1} + K_k (y_k - H x_{k|k-1})$$

Covariance는 Joseph form으로 갱신한다. 연속시간 model은 matrix exponential과 Van Loan 방법으로 이산화한다.

## 3. 1-DOF latent-force models

```
      ┌──────┐  z (차체 상하 변위)
      │  m   │
      └──┬───┘
       k │ c        스프링·댐퍼가 "땅"에 직접 붙어 있음 (바퀴 없음)
   ▔▔▔▔▔▔▔▔▔▔▔▔
```

차체를 질량 하나로 보고, 노면·바퀴·모델 오차는 전부 외란 가속도 $d$ 하나로 뭉갠다. 관측은 $a_z$ 하나, 출력은 $v$ → Bounce. 변형(RW/OU/Matern)의 차이는 "$d$를 얼마나 매끄럽다고 가정하나"뿐이고, `oscillator`는 $d$ 없이 진동자만으로 $a_z$를 설명하는 ablation이다. 가장 단순하고 빠르지만, 파라미터가 경계로 가면(§8) 물리 해석은 없다.

**상태.** 차체 변위·속도에 외란 블록을 붙인다 (`onedof()` + `augment()`).

| 상태 | 뜻 | 단위 |
|---|---|---|
| $z$ | 차체 상하 변위 (평형 기준) | m |
| $v = \dot z$ | 차체 상하 속도 → Bounce | m/s |
| $d$ (+ $\dot d$, $\ddot d$) | 관측되지 않는 외란 가속도. 변형에 따라 0–3개 | m/s² |

**입력.** 없음. 노면·바퀴·모델 오차는 $d$가 대신한다.

**연속시간 동역학.** 벡터 표기 ($d$ 블록은 변형에 따라 0–3개, 입력 없음):

$$x = [z,\ v,\ d,\ \dot d,\ \ddot d]^{\mathsf T}, \qquad y = a_z$$

$$\dot z = v$$

$$\dot v = -\omega^2 z - 2\zeta\omega\, v + d + w_v$$

$$\omega = 2\pi f$$

$w_v$는 intensity $q_v$의 백색잡음이고, $z$ 식에는 수치 안정용 $10^{-10}$만 둔다. 외란 블록은 `disturbance()`가 만든다.

| 변형 | $d$ 상태 | 동역학 | 백색잡음 intensity | 정상 분산 |
|---|---|---|---|---|
| `oscillator` | 없음 | — | — | — |
| `rw` | $d$ | $\dot d = w$ | $\sigma_d^2$ | 발산 ($P_0 = 10$) |
| `ou` | $d$ | $\dot d = -\lambda d + w$ | $2\sigma_d^2\lambda$ | $\sigma_d^2$ |
| `matern32` | $d, \dot d$ | $\ddot d = -\lambda^2 d - 2\lambda\dot d + w$ | $4\sigma_d^2\lambda^3$ | $\sigma_d^2$ |
| `matern52` | $d, \dot d, \ddot d$ | $d^{(3)} = -\lambda^3 d - 3\lambda^2\dot d - 3\lambda\ddot d + w$ | $\tfrac{16}{3}\sigma_d^2\lambda^5$ | $\sigma_d^2$ |

**관측.** $a_z$ 하나. 가속도 식의 우변이 그대로 $H$ 행이다.

$$a_z = -\omega^2 z - 2\zeta\omega\, v + d + e, \qquad e \sim \mathcal N(0, r)$$

$$H = [-\omega^2,\ -2\zeta\omega,\ 1,\ 0, \ldots]$$

**이산화·초기 공분산.** Van Loan (§2), $\Delta t = 0.01$ s. $P_0 = \mathrm{diag}(1, 1)$ ($z, v$) ⊕ 외란 블록의 정상 공분산 (Lyapunov 해; RW는 10). $\hat x_0 = 0$.

**파라미터** (`model_spec`).

| 기호 | 뜻 | 코드 | 취급 | 범위 | 시작 |
|---|---|---|---|---|---|
| $f$ | 차체 고유진동수 [Hz] | `p[0]` | 추정 | 0.5–8 | 1.3 |
| $\zeta$ | 감쇠비 | `p[1]` | 추정 | 0.05–5 | 0.3 |
| $q_v$ | $v$ 식 process-noise intensity | `log_qv = p[2]` | 추정 | $e^{-12}$–$e^{3}$ | $e^{-5}$ |
| $\sigma_d^2$ | 외란 정상 분산 [(m/s²)²] | `log_qf = p[3]` | 추정 | $e^{-12}$–$e^{5}$ | 1 |
| $r$ | $a_z$ 측정잡음 분산 [(m/s²)²] | `log_r = p[4]` | 추정 | $e^{-12}$–$e^{4}$ | $e^{-3}$ |
| $\lambda$ | 외란 상관 속도 [1/s] (`ou`/`matern`) | `p[5]` | 추정 | 0.05–100 | 2 |

`oscillator`는 `p[3]`을 쓰지 않는다 (dead parameter, Powell이 그래도 탐색함).

**출력.** $\hat v$ → $g\,\hat v + c$ → Bounce ($g, c$는 train 최소제곱).

변형별 비고와 근거:

- 공통 구조(미지 외력을 state에 추가한 acceleration observation): [Branlard et al., 2020, Wind Energy Science](https://wes.copernicus.org/articles/5/1155/2020/)
- `rw`: Nayek et al.의 Property 1·Eq. 57이 random-walk input을 GPLFM의 특수형으로 설명. Acceleration-only에서는 일정한 displacement와 이를 상쇄하는 외란을 구분하지 못해 저주파 drift가 생길 수 있다 ([Naets et al., 2015, MSSP](https://www.sciencedirect.com/science/article/pii/S0888327014002180)).
- `ou`: exponential covariance의 1차 Markov 표현. RW보다 stationary하지만 상관 시간 하나만 표현 ([Nayek et al., Appendix A, Eq. 71](https://arxiv.org/pdf/1904.00093#page=34)).
- `matern32`/`matern52`: 외란 smoothness를 단계적으로 높임 ([Nayek et al., Appendix A, Eq. 72–73](https://arxiv.org/pdf/1904.00093#page=34)). $\lambda$가 상한에 도달하므로 물리적 road correlation parameter로 해석할 수 없다.
- [Nayek et al., 2019, MSSP](https://doi.org/10.1016/j.ymssp.2019.03.048)

## 4. 2-DOF quarter-car

```
      ┌──────┐  z_s   sprung mass (차체의 1/4)
      │  m_s │
      └──┬───┘
     k_s │ c_s       서스펜션
      ┌──┴───┐  z_u   unsprung mass (바퀴 + 허브)
      │  m_u │
      └──┬───┘
     k_t │           타이어 (스프링만)
   ~~~~~~┴~~~~~~  r   노면 높이
```

차의 한 모서리(바퀴 하나 + 그 위의 차체 1/4)만 떼어 낸 모델이라 quarter-car다. 1-DOF와 달리 바퀴가 별도 질량으로 들어와 wheel-hop 모드(보통 10–15 Hz)가 생기고, 노면 $r$이 명시적 입력이 되어 posterior로 추정할 수 있다(→ IRI). 절대 질량 대신 비율로 정규화한다: $a = \omega^2 = k_s/m_s$, $b = 2\zeta\omega = c_s/m_s$, $\rho = m_s/m_u$, $\gamma = k_t/k_s$. 관측은 여전히 $a_z$ 하나, 출력은 $v_s$ → Bounce. 1-DOF 대비 얻는 것은 노면 추정이고 대가는 $\gamma$ 하나인데, wheel-hop이 CG의 $a_z$에 거의 안 보여 $\gamma$는 식별되지 않는다.

**상태** (`quarter_car()`).

| 상태 | 뜻 |
|---|---|
| $z_s, v_s$ | sprung mass 변위·속도 → Bounce |
| $z_u, v_u$ | unsprung mass 변위·속도 (estimate-only) |

$$x = [z_s, v_s, z_u, v_u]^{\mathsf T}$$

**입력.** 노면 높이 $r$ — 상태가 아니라 분산 $q_r$의 zero-mean 백색 확률 입력.

**연속시간 동역학.** 벡터 표기:

$$x = [z_s,\ v_s,\ z_u,\ v_u]^{\mathsf T}, \qquad u = r, \qquad y = a_z$$

물리식:

$$m_s\ddot z_s = -k_s(z_s - z_u) - c_s(v_s - v_u)$$

$$m_u\ddot z_u = k_s(z_s - z_u) + c_s(v_s - v_u) - k_t(z_u - r)$$

정규화식:

$$\dot z_s = v_s$$

$$\dot v_s = -a\,(z_s - z_u) - b\,(v_s - v_u)$$

$$\dot z_u = v_u$$

$$\dot v_u = \rho a\,(z_s - z_u) + \rho b\,(v_s - v_u) - \rho\gamma a\,(z_u - r)$$

$a = k_s/m_s = \omega^2$, $b = c_s/m_s = 2\zeta\omega$, $\rho = m_s/m_u$, $\gamma = k_t/k_s$. Process noise는 노면뿐이다. `discretize_input`으로 $r$의 입력 행렬 $G$를 이산화하고

$$x_{k+1} = A x_k + G r_k, \qquad r_k \sim \mathcal N(0, q_r), \qquad Q = q_r\, G G^{\mathsf T}$$

**관측.** $a_z = \dot v_s + e = [-a,\ -b,\ a,\ b]\,x + e$, $e \sim \mathcal N(0, r)$.

**노면 posterior.** 갱신 후 $\hat r_{k-1} = q_r G^{\mathsf T} P_{k|k-1}^{-1}\big(\hat x_{k|k} - \hat x_{k|k-1}\big)$. 공간 재배열 → Golden Car → IRI (참값 없음, estimate-only). 구조와 posterior 복원은 Agebjar et al.을 따른다.

- [Agebjar et al., 2025, IEEE FUSION](https://doi.org/10.23919/FUSION65864.2025.11123970)
- [Doumiati et al., 2011, ACC](https://doi.org/10.1109/ACC.2011.5991595)

**이산화·초기 공분산.** $\Delta t = 0.01$ s, $P_0 = I_4$, $\hat x_0 = 0$.

**파라미터.**

| 기호 | 뜻 | 코드 | 취급 | 범위 | 시작 |
|---|---|---|---|---|---|
| $f$ | sprung 고유진동수 [Hz] | `p[0]` | 추정 | 0.5–3 | 1.3 |
| $\zeta$ | 감쇠비 | `p[1]` | 추정 | 0.05–1.5 | 0.4 |
| $\gamma$ | $k_t/k_s$ | `log_gamma = p[2]` | 추정 | 3–20 | 10 |
| $q_r$ | 노면 입력 분산 | `log_road = p[3]` | 추정 | $e^{-20}$–$e^{-4}$ | $e^{-10}$ |
| $r$ | $a_z$ 잡음 분산 | `log_r = p[4]` | 추정 | $e^{-12}$–$e^{4}$ | $e^{-3}$ |
| $\rho$ | $m_s/m_u$ | 상수 | 고정 | 20/3 | — |

**출력.** $\hat v_s$ → $g\,\hat v_s + c$ → Bounce.

$\gamma$가 상한 20에 도달하므로 wheel-hop parameter를 물리값으로 단정할 수 없다.

## 5. Half-car, RTS와 IRI

```
   (정면에서 본 그림)
          z_s ↑   φ (roll)
      ┌────────────────┐
      │      m_s       │
      └──┬──────────┬──┘
      k_s│c_s    k_s│c_s
       ┌─┴─┐      ┌─┴─┐
       │m_u│      │m_u│      좌/우 unsprung
       └─┬─┘      └─┬─┘
        k_t         k_t
     ~~~~┴~~~ r_L ~~~┴~~~ r_R
```

Quarter-car를 좌우로 두 개 붙이고 차체에 roll 자유도를 준 4-DOF(bounce, roll, 좌·우 바퀴) 모델. 노면은 좌·우 독립 백색 입력 2개. qc2의 5개 파라미터를 그대로 쓰고 roll 관성비 $j$와 $r_{lat}$만 roll 라벨로 추가 fit한다. 출력은 $v_s$ → Bounce, $\dot\phi$ → Roll.

세 회전축의 의미:

| 회전 | 축 | 언제 생기나 | CAN IMU 직접 측정 |
|---|---|---|---|
| Roll $\phi$ | 앞뒤 축 — 차가 옆으로 기움 | 좌/우회전 시 원심력으로 바깥쪽으로 기움, 한쪽 바퀴만 요철을 밟을 때 | 있음 (`IMU_RollRtVal`) |
| Pitch $\theta$ | 좌우 축 — 앞뒤로 끄덕임 | 제동 시 nose-dive, 가속 시 squat, 앞뒤 노면 차이 | 없음 |
| Yaw $\psi$ | 수직 축 — 진행 방향이 돎 | 회전 그 자체 | 있음 (`IMU_YawRtVal`) |

복원 대상은 각도가 아니라 각속도(deg/s)이므로 코너 진입·탈출, 요철처럼 기울기가 *변하는* 순간에 값이 나오고 정상 선회 중에는 0에 가깝다. Roll rate는 CAN IMU에 자이로가 있으므로 애초에 추정 문제가 아니라 보정 문제이고, HC8이 그 채널을 쓰지 않고 $a_z + a_{lat}$만으로 roll을 추정하려 한 것은 문제 설정이 이상했던 것이다.

**상태** (`half_car()`). 좌/우 서스펜션 변형을 $\delta_l = z_s - \phi - z_{u,l}$, $\delta_r = z_s + \phi - z_{u,r}$로 둔다. Half-track $t$로 정규화되어 모델의 $\phi$는 $t\cdot$(roll 각), 단위 m다.

$$x = [z_s, v_s, \phi, \dot\phi, z_{u,l}, v_{u,l}, z_{u,r}, v_{u,r}]^{\mathsf T}$$

**입력.** 좌/우 노면 $r_l, r_r$ — 독립 백색, 각각 분산 $2q_r$ (평균 노면이 QC2와 같은 $q_r$이 되도록).

**연속시간 동역학.** 벡터 표기:

$$x = [z_s,\ v_s,\ \phi,\ \dot\phi,\ z_{u,l},\ v_{u,l},\ z_{u,r},\ v_{u,r}]^{\mathsf T}, \qquad u = [r_l,\ r_r]^{\mathsf T}, \qquad y = [a_z,\ a_{lat}^{HP}]^{\mathsf T}$$

각 측 스프링·댐퍼는 QC2의 절반($a/2$, $b/2$).

$$\dot z_s = v_s$$

$$\dot v_s = -\tfrac a2\,(\delta_l + \delta_r) - \tfrac b2\,(\dot\delta_l + \dot\delta_r)$$

$$\tfrac{d}{dt}\phi = \dot\phi$$

$$\ddot\phi = -\tfrac{a}{j}\,(\delta_r - \delta_l) - \tfrac{b}{j}\,(\dot\delta_r - \dot\delta_l)$$

$$\dot z_{u,l} = v_{u,l}$$

$$\dot v_{u,l} = \rho a\,\delta_l + \rho b\,\dot\delta_l - \rho\gamma a\,(z_{u,l} - r_l)$$

$$\dot z_{u,r} = v_{u,r}$$

$$\dot v_{u,r} = \rho a\,\delta_r + \rho b\,\dot\delta_r - \rho\gamma a\,(z_{u,r} - r_r)$$

$j = 2I_x/(m_s t^2)$ (정규화 roll 관성비). 좌우 대칭이라 $(r_l + r_r)/2$는 bounce만, $r_l - r_r$은 roll만 구동한다 — lateral signal을 추가해도 Bounce가 QC2와 동일한 것이 구조적으로 정상이다.

**관측.** $a_z = \dot v_s + e_z$, $a_{lat}^{HP} = \ddot\phi + e_{lat}$ (`IMU_LatAccelVal`에 0.5 Hz 인과 HP 적용). $R = \mathrm{diag}(r_z, r_{lat})$. 두 번째 관측식이 틀렸다는 것은 아래에.

**이산화·초기 공분산.** $\Delta t = 0.01$ s. $P_0 = I_8$, unsprung 네 성분은 2. $\hat x_0 = 0$.

**파라미터.**

| 기호 | 뜻 | 취급 | 범위 | 시작 |
|---|---|---|---|---|
| $f, \zeta, \gamma, q_r, r_z$ | QC2와 동일 | QC2 fit 값 그대로 고정 | — | — |
| $\rho$ | $m_s/m_u$ | 고정 | 20/3 | — |
| $j$ | roll 관성비 | roll 라벨 NRMSE로 추정 (`hc_objective`) | 0.5–20 | 1960/600 |
| $r_{lat}$ | $a_{lat}$ 잡음 분산 | 추정 | $e^{-12}$–$e^{4}$ | $e^{-3}$ |

**출력.** $\hat v_s$ → Bounce, $\hat{\dot\phi}$ → Roll (각각 자유 $g, c$).

Roll 복원은 실패했다 (held-out roll corr median 0.215, `kf_metrics.csv` — 파형이 공유하는 분산이 5% 수준으로 사실상 정보가 없는 출력). 현재 lateral 관측식은 `H[1] = f[3]`, 즉 $a_{lat} = \ddot\phi + e$로 roll 각가속도(rad/s²)를 단위 이득으로 m/s²에 대응시킨다. 실제 횡가속도는

$$a_{lat} \approx \underbrace{v\,\dot\psi}_{\text{코너링, 1–3 m/s}^2} + \underbrace{g\,\phi}_{\text{중력누설}} + \underbrace{h\,\ddot\phi}_{\text{roll 관성, 작음}}$$

이라 지배항이 코너링인데, 필터는 코너에 들어갈 때마다 $a_{lat}$의 큰 값을 roll 각가속도로 읽고 두 번 적분하므로 실제 roll rate와 모양·타이밍이 전혀 다른 신호가 나온다. 물리를 잘못 쓴 결과이지 roll이 원리적으로 안 되는 것이 아니다. Roll이 필요하면 `IMU_RollRtVal`을 직접 쓰면 되고, 굳이 모델로 가려면 관측식에 $v\dot\psi$(휠속 × `IMU_YawRtVal`, 둘 다 허용 센서)를 넣어 코너링을 빼 주어야 한다. 그 전까지 HC8의 roll 출력은 사용하지 않는다.

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

```
   (옆에서 본 그림)              진행 방향 →
             z_s ↑   θ (pitch, nose-up +)
      ┌────────────────────────┐
      │          m_s, I_y      │      v: 종속도 (상태)
      └──┬──────────────────┬──┘      a_x: 입력 (또는 관측)
     k_r │c_r         k_f   │c_f
       ┌─┴─┐              ┌─┴─┐
       │m_u│  뒤           │m_u│  앞
       └─┬─┘              └─┬─┘
        k_t                 k_t
     ~~~~┴~~~ r_r ~~~~~~~~~~~┴~~~ r_f
        ├── l_r ──┤├── l_f ──┤
```

HC8과 같은 발상이지만 좌우 대신 앞뒤로 붙여 pitch를 얻는 4-DOF(bounce, pitch, 앞·뒤 바퀴) 모델이고, 관측이 완전히 다르다: $a_z$에 더해 **휠속 앞/뒤 평균을 차체 운동의 관측**으로 쓴다. 그래서 종속도 $v$가 상태로 들어온다($\dot v = a_x$). Pitch rate는 CAN IMU에 자이로가 없어 휠속·$a_z$·$a_x$에서 간접적으로 끌어내야 하므로, 이 lab에서 차량 모델이 실제로 필요한 유일한 target이다. 상태에 bounce($z_s, v_s$)도 있지만 `SPECS`가 `target=2, output=3`이라 Pitch만 평가한다.

Target이 `Pitch_rate_6D`인 유일한 모델이다 (`pitch_half_car()`).

**상태** (기본 9개; 변형이 뒤에 덧붙인다).

| 상태 | 뜻 | 단위 |
|---|---|---|
| $z_s, v_s$ | 차체 bounce 변위·속도 | m, m/s |
| $\theta, \dot\theta$ | pitch 각 (nose-up +)·각속도 → Pitch | rad, rad/s |
| $z_{u,f}, v_{u,f}$ / $z_{u,r}, v_{u,r}$ | 앞/뒤 unsprung 변위·속도 | m, m/s |
| $v$ | 종속도 | m/s |
| (+) $r_f, r_r$ | 노면 변위 — `pitch_road*`/`axou`: 2개, `pitch_delay*`: $r_f$만 | m |
| (+) $d$ (, $\dot d$) | 잠재 pitch 모멘트 — `_ou`: 1개, `_osc`: 2개 | rad/s² |
| (+) $a_b, \gamma_g, b_x$ | 잠재 종가속, 구배, $a_x$ bias — `pitch_ax*` | m/s², rad, m/s² |

$$x = [z_s, v_s, \theta, \dot\theta, z_{u,f}, v_{u,f}, z_{u,r}, v_{u,r}, v,\ (\ldots)]^{\mathsf T}$$

**입력.** 기본 $u = a_x^{IMU}$ (`IMU_LongAccelVal` × 9.81). `pitch_tq`: $+\,\tau_f, \tau_r$ (`MCU_Mg2EstTqVal` → 앞, `MCU_Mg1EstTqVal` → 뒤 — 매핑 근거 없음, 가정). `pitch_ax*`: $u = (\tau_f, \tau_r)$만, $a_x$는 관측으로 이동.

**연속시간 동역학.** 벡터 표기 (기본형; 변형은 상태·입력·관측을 덧붙인다):

$$x = [z_s,\ v_s,\ \theta,\ \dot\theta,\ z_{u,f},\ v_{u,f},\ z_{u,r},\ v_{u,r},\ v]^{\mathsf T}, \qquad u = a_x^{IMU}, \qquad y = [v_{w,f},\ v_{w,r},\ a_z]^{\mathsf T}$$

앞/뒤 서스펜션 힘 (질량 정규화):

$$F_f = a_f\,(z_s + l_f\theta - z_{u,f}) + b_f\,(v_s + l_f\dot\theta - v_{u,f})$$

$$F_r = a_r\,(z_s - l_r\theta - z_{u,r}) + b_r\,(v_s - l_r\dot\theta - v_{u,r})$$

$$a_f = \tfrac{\omega^2}{2}(1+\varepsilon), \qquad a_r = \tfrac{\omega^2}{2}(1-\varepsilon)$$

$$b_f = \zeta\omega\,(1+\varepsilon_c), \qquad b_r = \zeta\omega\,(1-\varepsilon_c) \qquad (\varepsilon_c = \varepsilon;\ \texttt{pitch\_eps}만 별도 추정)$$

운동 방정식:

$$\dot z_s = v_s$$

$$\dot v_s = -(F_f + F_r) + w_b$$

$$\tfrac{d}{dt}\theta = \dot\theta$$

$$\ddot\theta = \frac{-l_f F_f + l_r F_r}{j\, l_f l_r} + g_u a_x + d + w_b$$

$$\dot z_{u,f} = v_{u,f}$$

$$\dot v_{u,f} = \rho F_f - \rho\gamma a_f\,(z_{u,f} - r_f)$$

$$\dot z_{u,r} = v_{u,r}$$

$$\dot v_{u,r} = \rho F_r - \rho\gamma a_r\,(z_{u,r} - r_r)$$

$$\dot v = a_x + w_v$$

$w_b$: intensity $q_{body}$ ($v_s$·$\dot\theta$ 행 공통), $w_v$: $q_{long}$. 정규화: $\omega = 2\pi f$ (bounce 고유진동수), $\varepsilon$ (앞뒤 강성 비대칭), $j = I_y/(m_s l_f l_r)$, $\gamma = k_t/k_s$, $\rho = m_s/m_u$. $l_f = 1.45$, $l_r = 1.50$ m, $\rho = 20$ 고정.

- [Rajamani, Vehicle Dynamics and Control, ch. 12](https://doi.org/10.1007/978-1-4614-1433-9)

노면 처리:

| 변형 | $r_f, r_r$ | process noise |
|---|---|---|
| `pitch_hc*` | 상태 없음. 백색 가속으로 $\dot v_{u,i}$에 직접 | $q_{road}\,(\rho\gamma a_i)^2$ |
| `pitch_road*`, `pitch_tq/ax/eps` | 상태 2개, $\dot r_i = w$ (RW 변위) | $q_{road}$ 각각 |
| `pitch_axou` | 상태 2개, $\dot r_i = -\lambda_{road} r_i + w$ | $2q_{road}\lambda_{road}$ |
| `pitch_delay*` | $r_f$만 RW 상태. $r_r(t) = \hat r_f(t - L/v)$를 휠속 누적 거리로 인덱싱해 기지 입력처럼 재생 (공분산 무시) | $q_{road}$ |

외란 $d$ ($\ddot\theta$ 행, §3의 블록 공용): `ou` $\dot d = -\lambda_d d + w$, `osc2` $\ddot d = -\omega_d^2 d - 2\zeta_d\omega_d\dot d + w$ (intensity $4\zeta_d\omega_d^3\sigma_d^2$, 정상 분산 $\sigma_d^2$).

`pitch_ax*` 추가 동역학 ($u = [\tau_f, \tau_r]^{\mathsf T}$, $y$에 $a_x^{IMU}$ 추가):

$$\dot a_b = -\lambda_a\big(a_b - g_v(\tau_f + \tau_r)\big) + w_a \qquad (\text{intensity } 2\sigma_a^2\lambda_a)$$

$$\dot\gamma_g = w_g \qquad (\text{intensity } q_{grade})$$

$$\dot b_x = w_x \qquad (\text{intensity } q_{bias})$$

$$\dot v = a_b$$

$$\ddot\theta \mathrel{+}= g_u a_b + g_f\tau_f + g_r\tau_r$$

**관측.** 휠속은 좌우 평균 [m/s].

$$v_{w,f} = v + \beta\,(v_s + l_f\dot\theta - v_{u,f}) - \lambda_f\dot\theta\ (+\, s_f\tau_f) + e_w$$

$$v_{w,r} = v + \beta\,(v_s - l_r\dot\theta - v_{u,r}) - \lambda_r\dot\theta\ (+\, s_r\tau_r) + e_w$$

$$a_z = \dot v_s + e_z, \qquad (\texttt{pitch\_ax*})\ \ a_x^{IMU} = a_b + g\,\theta + g\,\gamma_g + b_x + e_x$$

$R = \mathrm{diag}(r_{wheel}, r_{wheel}, r_{az}, (r_{ax}))$. $\beta$는 서스펜션 스트로크 속도가 휠 중심 전후 운동으로 새는 기구학 계수, $\lambda$는 pitch 회전 레버다. 이 데이터에서 $v_{w,f}-v_{w,r}$와 $\dot\theta$의 회귀 기울기 −0.2~−0.3 m가 실측된다 (`docs/claude_explanation/pitch_rate_exploration.md`).

**이산화·초기값.** $\Delta t = 0.01$ s. $\hat x_0$: $v$만 첫 샘플 휠속 평균, 나머지 0. $P_0 = I$; $d$ 블록은 정상 공분산, $a_b$는 $\sigma_a^2$, $\gamma_g, b_x$는 0.01.

**파라미터** (기본 14개 + 변형별 추가; 모두 `model_spec`).

| 기호 | 뜻 | index | 취급 | 범위 | 시작 |
|---|---|---|---|---|---|
| $f$ | bounce 고유진동수 [Hz] | 0 | 추정 | 0.5–4 | 1.5 |
| $\zeta$ | 감쇠비 | 1 | 추정 | 0.05–1.5 | 0.4 |
| $\varepsilon$ | 앞뒤 강성 비대칭 | 2 | 추정 | −0.9–0.9 | 0 |
| $j$ | $I_y/(m_s l_f l_r)$ | 3 (log) | 추정 | 0.3–3 | 1 |
| $\gamma$ | $k_t/k_s$ | 4 (log) | 추정 | 3–20 | 10 |
| $g_u$ | 하중이동 이득 [rad/s² per m/s²] | 5 | 추정 | −10–10 | 0.3 |
| $\beta$ | 스트로크→휠속 기구학 계수 | 6 | 추정 | −2–2 | 0.13 |
| $\lambda_f, \lambda_r$ | pitch 회전 레버 [m] | 7, 8 | 추정 | −10–10 | 0.55 |
| $q_{road}$ | 노면 process noise | 9 (log) | 추정 | $e^{-20}$–$e^{-2}$ | $e^{-10}$ |
| $q_{body}$ | 차체 process noise | 10 (log) | 추정 | $e^{-16}$–$e^{4}$ | 0.01 |
| $q_{long}$ | 종속도 process noise | 11 (log) | 추정 | $e^{-12}$–$e^{6}$ | 0.5 |
| $r_{wheel}$ | 휠속 측정잡음 분산 [(m/s)²] | 12 (log) | 추정 | $e^{-14}$–$e^{2}$ | $2\times10^{-3}$ |
| $r_{az}$ | $a_z$ 측정잡음 분산 [(m/s²)²] | 13 (log) | 추정 | $e^{-6}$–$e^{6}$ | 0.6 |
| $l_f, l_r$ | CG–축 거리 [m] | 상수 | 고정 | 1.45, 1.50 | — |
| $\rho$ | $m_s/m_u$ | 상수 | 고정 | 20 | — |
| `_ou`: $\lambda_d$, $\sigma_d^2$ | 외란 | 14, 15 | 추정 | 0.05–100, $e^{-12}$–$e^{8}$ | 2, 1 |
| `_osc`: $\omega_d$, $\zeta_d$, $\sigma_d^2$ | 외란 | 14–16 | 추정 | 0.3–60, 0.05–2, $e^{-12}$–$e^{8}$ | $2\pi$, 0.7, 4 |
| `tq`: $g_f, g_r$ / $s_f, s_r$ | 토크 모멘트 / 슬립 feedthrough | +4 | 추정 | ±0.1 / ±0.02 | 0.01 / $10^{-3}$ |
| `ax`: $\lambda_a, g_v, \sigma_a^2, q_{grade}, q_{bias}, r_{ax}$ | 종가속 잠재·구배·bias·$a_x$ 잡음 | +6 | 추정 | 1–200, ±0.05, $e^{-8}$–$e^{4}$, $e^{-16}$–1, $e^{-16}$–1, $e^{-8}$–$e^{4}$ | 20, 0.01, 0.2, $e^{-8}$, $e^{-10}$, $e^{-3}$ |
| `axou`: $\lambda_{road}$ | 노면 상관 속도 [1/s] | +1 | 추정 | 0.01–50 | 0.5 |
| `eps`: $\varepsilon_c$ | 감쇠 비대칭 | +1 | 추정 | −0.9–0.9 | $\varepsilon$ |

**출력.** $\hat{\dot\theta}$ → $\tfrac{180}{\pi}\hat{\dot\theta} + c$ → Pitch [deg/s] ($c$만 train에서 추정; §5.5 "gain 고정 재적합" 이전에는 $g$도 자유였음).

외란 블록은 1-DOF latent-force 가족(§3)과 공용이다 (`disturbance()` + `augment()`, [Nayek et al., 2019, MSSP](https://doi.org/10.1016/j.ymssp.2019.03.048)). osc2는 $\zeta_d = 1$이면 Matérn 3/2, $\zeta_d < 1$이면 quasi-periodic latent force다. 관측성은 cascade PBH로 확인된다: 외란 극이 plant의 $d \to y$ transmission zero와 겹치지 않으면 관측 가능한데, 이 plant의 구조적 공통 zero는 $s = 0$ 하나뿐이므로(상수 모멘트는 정적 트림만 바꿔 속도·가속도 센서에 안 보임) $\omega_d > 0$ 또는 $\lambda_d > 0$이면 성립한다. 같은 이유로 random-walk 모멘트는 불가하다.

노면 변형의 근거: RW 변위는 ISO 8608의 $f^{-2}$ 변위 스펙트럼에 대응한다. 상수 노면 방향은 $s=0$ blocking zero 때문에 비관측이므로 $\hat r$은 절대 높이가 아닌 상대 프로파일(estimate-only)이다. 독립 RW에서는 앞뒤 반대부호 상수(정적 pitch 트림과 등가) 방향도 비관측이라 $\hat r_f, \hat r_r$이 서로 반대로 drift할 수 있는데(state plot에서 관찰됨), 지연 연결은 이 자유도를 구조적으로 제거한다.

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

당시 관찰: (1) osc2 외란이 지배적 요소다 — 어느 노면 처리 위에서든 parent 대비 +0.10~0.18을 더하며, $\omega_d \approx 9$–11 rad/s (1.4–1.7 Hz), $\zeta_d$ 소로 식별되어 **1–2 Hz 노면 pitch 여기 대역의 quasi-periodic 모멘트**를 흡수한다. (2) 노면 상태 자체(백색 가속 vs RW 변위 vs 지연 연결)는 단독으로는 0.72 수준에서 갈리지 않는다. (3) 물리적으로 올바른 지연 연결(delay_osc 0.807)이 유연한 독립 노면(road_osc 0.905)보다 오히려 낮다.

**위 표와 관찰은 자유 calibration gain 아래에서 얻은 것이며 §5.6의 비식별성 때문에 무효다.** 세 관찰 모두 아래 gain 고정 재적합에서 뒤집힌다.

### gain 고정 재적합 (2026-08-29)

`Pitch_rate_6D`는 deg/s이므로 calibration을 $g = 180/\pi$, $c = \bar y - g\bar{\hat{\dot\theta}}$로 고정했다 (`SPECS[name]["gain"]`, `calibrate(x, y, gain)`). 나머지 조건(train 300 episode, NRMSE, Powell maxiter 300, best-visited)은 동일. 기본 시작점에서 `pitch_hc`는 부호를 넘어 물리 해로 갔지만 `pitch_road`/`pitch_delay`는 잘못된 부호의 골짜기에 갇혀(레버 +0.4~+2.9 m, corr 0.71–0.73) `pitch_hc` 해를 warm start로 주어 재적합했다. 이 규칙은 `run_pitch`에 반영되어 있다 (road/delay ← `pitch_hc`, `_ou`/`_osc` ← parent + $\sigma_d^2 = e^{-8}$).

| Method | 자유 gain corr | **gain 고정 corr** [p10, p90] | RMSE | lag | $f$ [Hz] | $\zeta$ | $j$ | $\varepsilon$ | $\gamma$ | 레버 [m] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pitch_hc | 0.718 | 0.866 [0.67, 0.92] | 2.43 | 10 ms | 1.40 | 0.05↓ | 0.68 | −0.05 | 3.0↓ | −1.01 |
| pitch_hc_ou | 0.793 | 0.878 [0.68, 0.94] | 2.02 | 10 ms | 1.39 | 0.05↓ | 0.67 | −0.02 | 3.2 | −0.73 |
| pitch_hc_osc | 0.879 | **0.884** [0.70, 0.94] | 2.14 | 10 ms | 1.04 | 0.05↓ | 0.41 | −0.08 | 3.0↓ | −0.93 |
| pitch_road | 0.723 | 0.857 [0.59, 0.92] | 2.31 | 20 ms | 1.59 | 0.08 | 0.73 | −0.01 | 6.1 | −0.89 |
| pitch_road_osc | 0.905 | 0.857 [0.59, 0.92] | 2.29 | 20 ms | 1.58 | 0.07 | 0.71 | 0.00 | 5.3 | −0.89 |
| pitch_delay | 0.722 | **0.882** [0.63, 0.93] | 2.23 | 10 ms | 1.60 | 0.09 | 0.77 | −0.05 | 6.5 | −0.92 |
| pitch_delay_osc | 0.807 | **0.882** [0.62, 0.93] | 2.21 | 10 ms | 1.59 | 0.08 | 0.77 | −0.03 | 5.5 | −0.92 |

(↓ = 탐색 하한. 레버 $= \beta(l_f + l_r) - \lambda_f + \lambda_r$, 실측 회귀 기울기 −0.2~−0.3 m. 자유 gain 결과는 `outputs/pitch_freegain/`.)

관찰:

1. **거울상 해가 사라졌다.** 7개 전부 레버가 음수(−0.7~−1.0 m, 실측과 같은 부호), $f$ 1.0–1.6 Hz, $j$ 0.4–0.8, $\varepsilon \approx 0$, road/delay에서는 $\gamma$도 내부값(5–6.5). 파라미터가 처음으로 승용차 물리값 범위에 들어왔다. 남은 경계는 $\zeta = 0.05$ 하한(hc 계열)과 $\gamma = 3$ 하한(hc 계열)뿐이며, 전자는 비선형 감쇠 후보와 일관되고 후자는 wheel-hop이 관측 대역 밖이라 예상된 것이다.
2. **osc2 외란의 기여가 소멸했다.** hc에서 +0.02, road/delay에서 0 (warm start의 $\sigma_d^2 = e^{-8}$ 퇴화점에서 움직이지 않음). 이전의 +0.10~0.18은 노면 여기 모멘트가 아니라 **퇴화한 플랜트(4 Hz, $j = 3$)를 외란이 대신 모델링한 것**이었다. 플랜트가 물리 스케일이면 half-car 자체가 1–1.6 Hz 피치 모드를 잡는다.
3. **지연 연결이 독립 노면을 앞선다** (0.882 vs 0.857). 이전의 반대 결론(0.807 vs 0.905)은 거울상 해 + 외란 자유도의 산물이었다. 물리적으로 올바른 구조가 플랜트가 물리적일 때 이긴다.
4. **상한이 0.905 → 0.884로 내려갔다.** 자유 gain의 0.905는 출력으로는 유효한 예측기였으나 부호가 뒤집힌 $\theta$와 외란이 만든 것이었고, 물리 일관성을 요구한 대가가 corr 0.02다. 비인과 선형 상한(FIR 0.96)과의 격차는 0.08.
5. **2단계(`pitch2`: tq/ax/axou/eps)는 자유 gain의 `pitch_road_osc`를 anchor로 했으므로 무효다.** `pitch_delay` 또는 `pitch_hc_osc`를 anchor로 gain 고정 조건에서 다시 돌려야 한다. $\theta$가 물리 스케일이 되었으므로 $a_x$의 $g\theta$ 항이 이번에는 설계대로 작동할 것으로 기대한다.

### 2단계 확장 (`pitch2`: 토크 입력, $a_x$ 관측, 유색 노면)

**이 절의 결과는 자유 gain의 `pitch_road_osc`(거울상 해)를 anchor로 한 것이라 무효다. gain 고정 조건에서 재실행 전까지 참고용으로만 남긴다.**

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

$q_{grade} = q_{bias} \to 0$은 구배·bias random walk가 상수로 퇴화했다는 뜻이고, $f = 4$ Hz bounce와 $j = 3$은 승용차 물리값(1–1.5 Hz, $j \approx 0.8$–1.0)이 아니다. 이 값들은 §5.6의 평평한 손실 계곡이 경계에서 끝난 자리이지 식별 결과가 아니다.

## 5.6 Pitch 파라미터의 비식별성

**손실의 불변성.** 목적함수는 affine calibration을 포함한다.

$$J(p) = \min_{g,c}\ \frac{\big\|\, g\,\hat{\dot\theta}(p) + c - \dot\theta^{6D} \big\|_2}{\sigma_{6D}}$$

따라서 어떤 $p'$이 $\hat{\dot\theta}(p') = s\,\hat{\dot\theta}(p)$ ($s \ne 0$, 음수 포함)를 만들면 $g' = g/s$로 $J(p') = J(p)$다. $y = abx$에서 $ab$만 식별되는 상황과 같고, 출력 파형과 corr 비교에는 무해하다. 문제는 이 모델이 $\theta \to s\theta$를 거의 자유롭게 실현할 수 있다는 것이다. $\theta$가 데이터와 연결되는 경로를 보면:

| 경로 | $\theta$ 스케일·부호 고정 | 이유 |
|---|:---:|---|
| 휠속 $v_{w,f} = v + \beta(\cdot) + (\beta l_f - \lambda_f)\dot\theta$, $v_{w,r} = \cdots + (-\beta l_r - \lambda_r)\dot\theta$ | ✗ | $\beta, \lambda_f, \lambda_r$ 자유, 부호 무제한 → $\lambda_f' = \beta l_f - (\beta l_f - \lambda_f)/s$ 로 어떤 $s$든 흡수 |
| 피치 관성 $\ddot\theta = \dfrac{-l_f F_f + l_r F_r}{j\, l_f l_r}$ | ✗ | $j' \approx j/s$ 가 응답 스케일을 조정, 고유진동수 변화는 $\omega, \varepsilon$ 이 보상 |
| bounce–pitch 커플링 $\ddot z_s \ni -\dfrac{\omega^2}{2}\big[\varepsilon(l_f + l_r) + (l_f - l_r)\big]\theta = -\dfrac{\omega^2}{2}(2.95\,\varepsilon - 0.05)\,\theta$ | 거의 ✗ | $\varepsilon \in [-0.9, 0.9]$ 로 크기·부호 모두 조절. $s = -1$ 은 $\varepsilon' = -\varepsilon + 0.034$ |
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

**검증 (2026-08-29).** 2번(gain $= 180/\pi$ 고정)을 실행했다 (§5.5 "gain 고정 재적합"). 같은 센서·같은 라벨·같은 optimizer에서 레버 부호가 실측과 일치하고 $f, j, \varepsilon$이 물리값으로 들어왔으며 corr이 오히려 올랐다(hc 0.718 → 0.866). 즉 λ는 찾을 수 있는 값이었고, 자유 $g$가 $\lambda \cdot g$의 곱만 남긴 것이 비식별성의 전부였다. 남은 자유도(1, 3)는 아직 닫지 않았다 — 레버 크기가 실측의 3–4배인데, 실측 기울기가 stroke 항 $\beta(v_s + l_f\dot\theta - v_{u,f})$의 상관까지 합친 총 효과라 직접 비교는 아니다. 부호 대칭은 사라졌지만 Powell이 잘못된 부호 골짜기에 갇히는 일은 여전히 있어(road/delay 기본 시작점), warm start로 우회했다.

## 5.7 차량 모델 요약

DOF는 상태 수가 아니라 강체(질량) 자유도 수다. Pitch와 Bounce를 동시에 평가한 모델은 없다.

| 모델 | 강체 | 상태 수 | 관측 | 평가 target | 새로 얻는 것 | 대가 |
|---|---|---:|---|---|---|---|
| 1-DOF (`oscillator`/`rw`/`ou`/`matern*`) | 차체 | 2 (+0–3) | $a_z$ | Bounce | 가장 단순, 빠름 | 물리 해석 없음 |
| QC2 | 차체 + 바퀴 | 4 | $a_z$ | Bounce (+ 노면 posterior, IRI) | 노면 추정 | $\gamma$ 비식별 |
| HC8 | 차체(bounce+roll) + 바퀴 2 | 8 | $a_z$, $a_{lat}$ | Bounce + Roll | Roll (실패) | 관측식 오류 |
| Pitch HC (`pitch_*`) | 차체(bounce+pitch) + 바퀴 2 + $v$ | 9–17 | 휠속 2, $a_z$, ($a_x$) | Pitch | Pitch | 비식별성(gain 고정으로 해결), 느림 |

차량 모델이 아닌 것: kinematic KF(상수가속 3상태), KF+LSTM hybrid, model-free 6종.

## 5.8 단계적 축소 모델 (`pitch_staged_reconstruction.py`, 2026-08-31)

`methods_gpt.md`의 제안(reduced longitudinal-pitch → bounce-pitch)을 받아, half-car 대신 **센서 체인을 옳게 모델링한 최소 모델**에서 출발해 요소를 하나씩 추가하는 실험. 실측 근거: 휠 파생 $a_x$가 IMU보다 50–60 ms 선행(yaw 채널도 ~30 ms → CAN IMU 체인이 늦음), $\Delta v_w$–$q$ 회귀 기울기 −0.29 m, 토크 매핑 Mg2=앞/Mg1=뒤(VCU cmd와 corr 0.99).

**프로토콜 강화**: test 5명 중 신민철·이강근 봉인(최종 검증 전 미조회), 평가는 **dev-test 3명 128 ep**. Powell은 best-visited 재시작(조기 종료 우회). $g = 9.81$, gain $= 180/\pi$ 고정. `sup`(라벨 NRMSE)과 `ml`(innovation 우도, 라벨 무사용)을 병행하고 free_gain(자유 회귀 이득, 물리적이면 57.3)·signed lag·NIS·ACF(1)를 함께 보고.

| 단계 | 상태 | 추가 요소 | sup corr | RMSE | free_gain | ml corr |
|---|---:|---|---:|---:|---:|---:|
| a_naive | 5 | $[v_x, a_x, \theta, q, \gamma_g]$, $y = [\bar v_w,\ a_x^{IMU} = a_x + g\theta + g\gamma_g]$ | 0.462 | 3.71 | 28 | 0.06 |
| a_lag | 6 | + IMU 1차 지연 상태 $a_I$ ($\tau_I$) | 0.616 | 3.32 | 46 | −0.43 |
| a_full | 6 | + IMU 높이 레버암 $h_I$ ($-h_I\dot q$) | 0.836 | 2.26 | 53 | 0.18 |
| b_full | 9 | + bounce $[z_s, \dot z_s]$, $a_z^{IMU} = \ddot z_s + x_I\dot q$ (지연 공유) | 0.873 | 2.03 | 55 | 0.17 |
| c_wheel | 9 | + $\Delta v_w = \ell q + \kappa a_x$ | 0.887 | 2.01 | 54 | **0.545** |
| **d_torque** | 9 | + $u = [T_f, T_r]$: $\dot a_x = -\lambda_a a_x + b_T\Sigma T$, slip $s_f T_f + s_r T_r$ | **0.936** | **1.48** | **57.1** | 0.542 |
| e_fixgeo | 9 | $x_I = 0.42$ m 고정 (회사 제원: IMU 전방 <0.5 m) | 0.900 | 1.76 | 56 | 0.535 |
| f_fixlever | 9 | + $\ell = -0.29$ m 고정 (실측) | 0.899 | 1.83 | 56 | 0.541 |
| f + joint μ=3 / 10 | 9 | 목적함수 = 우도 + μ·NRMSE (방법 A) | **0.921 / 0.923** | 1.72 / 1.71 | 57 / 56 | (NIS 1.09 / 1.08) |
| f + aug | 9 | 학습 시 라벨을 관측 채널로 (방법 B), r_label fit/1/10 | 0.48–0.60 | ~4.0 | 24–26 | (NIS ≈ 1.06) |
| g_physical (sup / joint μ=3) | 9 | f + GV60 제원·실측 물리 범위 (항목 9), 토크는 정상 이득 $g_a = i/(rm)$ 로 | 0.908 / 0.910 | 1.78 / 1.85 | 54 / 55 | 0.539 |
| 기준: half-car 최고 (dev 재평가) | 9–11 | 14–17 파라미터 | 0.875–0.896 | 2.1–2.4 | 61–67 | 거울상 |

관찰:

1. **성능의 주인은 모델 복잡도가 아니라 센서 체인이다.** 지연(+0.15)·레버암(+0.22)·$\Delta v_w$ 채널·토크 입력이 각각 기여해 9-상태 모델이 half-car 최고치를 +0.04 넘었고, free_gain이 57.1로 **물리 스케일이 라벨 회귀로 재확인**된다.
2. **재현되는 물리 파라미터**: $h_I = -0.358 \to -0.355$ (a_full→b_full), $\kappa = -0.0748$(실측 −0.075), $\ell$ 부호·차수 일치. 이 실험에서 처음으로 "차량 파라미터를 식별했다"고 말할 수 있는 값들.
3. **라벨-프리 ML이 $\Delta v_w$ 채널부터 동작** (0.17 → 0.545, 거울상 소멸, NIS 1.03). 스케일은 아직 틀림 — free_gain 11 ≪ 57.3, 즉 $\hat q$ 가 물리값보다 **약 5배 크고** 고주파 잡음이 많음 (`pitch_staged_compare_*.png` 의 파란 점선). $\ell$·$\sigma_q$ 곱 비식별로 추정. 결합 목적함수 또는 $\ell$ 제원 고정이 다음 열쇠.
4. **남은 문제**: sup의 NIS 0.5–2.2(필터 비일관), 위상 성형 파라미터($\tau_I$, $h_I$, $x_I$, $\ell$)의 경계 표류(τ→5 ms 하한, d_torque에서 $b_T$ 상한·$\lambda_a$ 하한·$\ell \to -0.85$ 이동) — 교환 자유도가 남아 있어 IMU 장착 위치·기하 제원으로 고정하는 것이 정석.

5. **제원 고정 (e, f)**: $x_I$ 고정으로 0.936 → 0.900 (자유 $x_I = -1.35$가 라벨 위상을 흉내 낸 몫 ~0.036), $\ell$ 고정은 공짜(0.899). 제약을 걸수록 NIS가 2.3 → 1.21로 개선 — 물리 제약이 성능이 아니라 비일관성을 깎는다.
6. **결합 목적함수 (방법 A, 2026-09-08)**: $J = \varphi(p) + \mu\cdot\mathrm{NRMSE}$, μ = 3–10에서 **corr 0.921–0.923이면서 NIS 1.08–1.09** — sup(0.899, NIS 1.21)과 ml(0.541, NIS 1.03)을 동시에 능가. 우도 항이 정규화로 작용해 sup 단독이 갇혔던 local minimum($f_p, b_a$ 상한)을 탈출, $f_p = 1.6$ Hz·$h_I = -0.33$ m 내부값. μ = 0.3은 라벨 가중 부족(0.32). 남은 경계: $r_x, r_z$ 하한, $f_z$ 하한.
7. **기준 관측 채널 (방법 B)는 실패**: 학습 시 라벨을 관측식에 넣으면 필터가 라벨로 q를 알아 센서 채널이 pitch 추출을 학습하지 않음(배포 필터 corr 0.48–0.60, free gain ~25). 학습/배포 필터 불일치의 전형. 순수 우도 안에서 라벨을 쓰려면 다른 구조가 필요.
8. **EM 으로 Q, R 만 재추정 (2026-09-09, `run_em_noise_covariance.py`, 교수님 제안 arXiv 2105.00250 §2.2).** `f_fixlever` 플랜트(A, H, B, D, P₀)를 세 적합(sup / ml / joint μ=3)에서 가져와 고정하고, 잡음 공분산만 EM(E-step RTS smoother 모멘트, M-step 논문 식 30–31; `full` 전체 행렬 / `diag` 대각)으로 60회 갱신. 라벨 미사용.

   | 플랜트 출처 | Q,R | 우도 energy (↓) | pitch corr | free gain | NIS |
   |---|---|---:|---:|---:|---:|
   | sup | 원래 (라벨 fit) | −2.77 | 0.899 | 56 | 1.21 |
   | sup | **EM full** | −4.63 | 0.543 | 17 | 1.17 |
   | ml | 원래 (Powell 우도) | −5.21 | 0.541 | 7 | 1.03 |
   | ml | EM full | −5.34 | 0.536 | 7 | 1.07 |
   | joint μ=3 | 원래 | −4.49 | 0.921 | 57 | 1.09 |
   | joint μ=3 | EM full / diag | −4.68 / −4.55 | 0.737 / **0.745** | 24 / 33 | 1.15 |
   | sup | EM full **수렴** (3,000회, 2026-09-13) | −5.08 | 0.555 | 14 | 1.14 |
   | ml | EM full **수렴** (1,484회에서 tol 1e-6 도달) | −5.61 | 0.515 | 7 | 1.05 |
   | joint μ=3 | EM full **수렴** (3,000회) | −5.32 | 0.573 | 8 | 1.09 |

   ⚠ 위 표의 EM 값은 **60회 반복에서 멈춘 과도값**이며 수렴값이 아니다 (항목 10). 수렴까지 돌리면 세 플랜트 모두 0.43–0.54 로 내려간다.

   읽기: (i) EM 은 이론대로 동작 — `full` 은 세 출처 모두 우도가 매 반복 단조 증가, 60회 9초 (그러나 60회는 수렴이 아니다, 항목 10). `diag` 는 Van Loan 이산화의 비대각을 버리는 다른 모델족이라 첫 스텝에서 우도가 떨어질 수 있고(ml, joint 출처) 최종 우도도 낮다. (ii) **우도 최적 Q,R ≠ 라벨 최적 Q,R 을 정량화**: sup 플랜트의 라벨 fit Q,R(0.899)을 EM 으로 우도 최적으로 되돌리면 0.54 — sup 의 잡음 파라미터는 잡음 통계가 아니라 라벨 맞춤용이었다는 증거. ml 플랜트에서는 EM 이 거의 안 움직임(Powell 이 이미 우도 최적 근처, 이 트랙의 $r_x$ 하한은 경계 인공물이 아니라 우도의 실제 요구). (iii) **플랜트와 잡음의 역할 분리**: joint 플랜트(좋은 동역학·기하) + 라벨-프리 EM 잡음 = 0.745 로, 같은 EM 을 sup 플랜트(0.54)나 ml 플랜트(0.54)에 적용한 것보다 훨씬 좋다. 즉 라벨은 플랜트(진동수·감쇠·레버암)를 잡는 데 필요하고, 잡음은 EM 으로 라벨 없이 채워도 0.74 가 나온다 — "제원으로 플랜트 고정 + Q,R 은 EM" 배포 형태의 예행 결과.
9. **물리 범위 적합 (2026-09-13, `g_physical`).** info 파일의 `Vehicle: JW`·`JW1_*.dbc` 로 차량이 제네시스 GV60 (개발코드 JW) 임을 확인하고, 공개 제원 (축거 2.90 m, 탑승 포함 약 2.3 t, 감속비 10.65, 타이어 반경 0.36 m) 과 실측 (자유감쇠 pitch 1.8 Hz·ζ 0.22, bounce 1.65 Hz·ζ 0.24, Δv_w 회귀 κ = −0.075) 으로 탐색 범위를 좁혔다 (`PHYSICAL`): $f_p$ 1.2–2.0 Hz, $\zeta_p$ 0.15–0.4, $b_a$ 0.1–0.4 ($m h_{cg}/I_{yy} \approx 0.25$), $h_I$ ±0.5 m, $\tau_I$ 15–100 ms, $f_z$ 1.2–1.8 Hz, $\zeta_z$ 0.15–0.4, $c_{zp}$ ±15, $\kappa$ −0.12~−0.04 s, $\lambda_a$ 1–20 s⁻¹, 토크는 $b_T$ 대신 정상 이득 $g_a = b_T/\lambda_a \in [0.009, 0.018]$ m/s² per Nm ($i/(rm) = 0.013$), $\sigma_x \ge 0.02$, $\sigma_z \ge 0.03$ m/s². 결과: sup 0.908 (f 0.899 보다 상승 — 상자가 $f_p$·$b_a$ 상한의 나쁜 local minimum 을 제거), joint μ=3 0.910 (f 0.921 에서 −0.011), ml 0.539 (불변). **물리값 범위 안의 모델을 corr 0.01 의 대가로 얻는다.** 다만 상자 안에서도 경계로 가는 방향은 이전과 같다: $\tau_I \to$ 15 ms 하한 (라벨 없는 ml 포함 셋 다 — 실측 50–60 ms 는 IMU 1차 지연이 아니라 채널 간 위상차의 총합), $\zeta_p \to$ 0.15 하한, $\sigma_x, \sigma_z \to$ 하한, $\kappa \to$ −0.04 (slip 항 $s_f, s_r$ 이 일부 대신), $\lambda_a \to$ 20 상한 (a_x 를 토크에 즉시 응답시키고 $g_a$ 로 보상), $c_{zp} \to$ ±15 (sup −15·joint +15 로 부호가 갈림 = 비식별, 경계가 잘라낼 뿐). sup 은 $q_a$ 상한으로 NIS 0.17 (비일관), joint 는 NIS 1.11 → **채택안은 g_physical · joint μ=3**. 같은 플랜트에 라벨 없는 EM 으로 $Q, R$ 만 채우면 60회에서 0.812 / 0.813 이 나오지만 이것은 수렴값이 아니다 (항목 10: 수렴 시 0.43–0.53). 그림: `pitch_staged_compare_g_physical.png`, `pitch_staged_diagnostics_g_physical.png`.
10. **EM 수렴 확인 (2026-09-13, `outputs/em_long_run_g_physical.{json,png}`).** 항목 8·9 의 EM 수치는 60회에서 멈춘 값이고 그 시점에 우도는 매 반복 오르고 있었다 (`em()` 의 상대 tol 1e-5 는 발동하지 않았고 반복 상한 60 이 먼저 걸림). g_physical 의 sup / joint 플랜트에서 100회 단위로 3,000회까지 이어 돌린 결과 우도 (energy) 는 단조 감소, pitch corr 은 단조 하강한다: sup 플랜트 0.813 (60회) → 0.59 (300) → 0.43 (1,000 이후 수렴, energy −5.29); joint 플랜트 0.812 (60) → 0.72 (300) → 0.55 (1,000) → 0.53 (3,000, 아직 −1e-4/회 하강 중). 수렴점에서 $R$ 은 거의 0 ($\sigma_w$ 0.6 mm/s, $\sigma_x$ 0.007, $\sigma_z$ 0.006 m/s²) 이고 분산이 전부 full $Q$ 로 옮겨간다 — 유색 측정잡음을 모델링하지 않은 모델에서 우도가 선호하는 해이며, ml (Powell) 이 $r_x, r_z$ 하한에 붙던 것과 같은 현상이다. **결론: 플랜트가 무엇이든 수렴한 EM 의 pitch corr 은 ml 과 같은 0.43–0.54 이고, 0.74–0.81 은 sup 잡음 근처에서 일찍 멈춘 과도값이었다.** 정정 대상: 항목 8 의 0.745 / 0.543 / 0.536, 항목 9 의 0.812 / 0.813. 수렴 설정 (`run_em_noise_covariance.py` 3,000회, tol 1e-6) 재계산: f_fixlever 플랜트 sup / ml / joint → 0.555 / 0.515 / 0.573, g_physical 플랜트 → 0.432 / 0.575 / 0.534 (`outputs/em_noise_covariance_*` 는 마지막 실행인 g_physical 것). EM 은 반복당 0.15 s 로 싸지만 능선을 따라 수천 회가 필요하므로 이후 EM 은 `--em-iters 2000` 이상 또는 상대 tol 1e-6 으로 돌린다 (`run_em_noise_covariance.py`, `plot_pitch_staged_waveforms.py`, alt 목적함수 모두 반영).
11. **교대 최적화 (2026-09-13, 목적함수 `alt:<n>`): 플랜트는 sup, $Q, R$ 은 EM.** 잡음 파라미터를 탐색에서 빼고 [EM (플랜트 고정, 라벨 없음) → 플랜트만 sup] 을 n회 교대, 마지막은 EM 으로 끝내 배포 필터 = "sup 플랜트 + EM 잡음" 을 만든다. g_physical 물리 범위, sup 플랜트 warm start, EM 2,000회. 결과 **0.536** (NRMSE 1.11, free gain 10) 으로 ml (0.539) 과 같다. 각 라운드의 sup 단계는 EM 잡음 아래에서 NRMSE 1.75 → 1.62 → 1.11 로 평균 예측 (1.0) 보다 나은 플랜트를 찾지 못했고, 파라미터는 경계로 흩어졌다 ($h_I$ 부호 반전 +0.32 m, $f_p$ 하한, $\lambda_a$ 하한, $\zeta_p$·$b_a$·$f_z$ 상한). 시작값 플랜트에서 출발한 첫 시도 (warm start 없음, EM 60회) 는 0.298. 해석: 수렴한 EM 의 $Q, R$ ($R \approx 0$, full $Q$) 아래에서는 물리 범위 안의 어떤 플랜트도 pitch 를 복원하지 못한다. pitch 추출의 대역·위상을 정하는 것은 플랜트만이 아니라 $Q, R$ 인데, 우도에는 그 정보가 없기 때문이다 (§5.6, §5.8-3, §5.8-10). **결론: 라벨 정보는 플랜트뿐 아니라 $Q, R$ 에도 들어가야 하고, 그것을 하나의 목적함수로 묶는 joint 가 현재 동작하는 유일한 결합 방식이다.**
12. **식별되는 항목만 EM (2026-09-14, `run_em_noise_covariance.py --variants structured`, `outputs/em_noise_covariance_structured_*`).** 항목 10–11 의 진단 (우도는 $R$ 과 pitch 관련 $Q$ 항목을 정할 정보가 없다) 에 따라, $R$ 전체와 $q_p$ (pitch 구동), $q_g$ (구배) 는 적합값에 고정하고 **$q_a$ (몸체 가속 구동), $q_z$ (bounce 구동) 두 개만** EM 으로 갱신한다. M-step 의 해당 대각 원소 비율로 연속시간 세기를 갱신하고 같은 플랜트로 Van Loan 재이산화하므로 $Q$ 의 구조 (비대각 포함) 가 유지된다 (generalized EM 근사). g_physical, 200 에피소드, tol 1e-6.

   | 플랜트 | $q_a$ | $q_z$ | energy (fit) | pitch corr | NIS | 반복 |
   |---|---:|---:|---:|---:|---:|---:|
   | sup 플랜트 (적합값) | 54.6 (상한) | 66.9 | −2.05 | 0.908 | 0.17 | — |
   | sup 플랜트 + structured EM | 10.6 | 7.3 | −3.09 | 0.874 | 0.60 | 14 |
   | joint μ=3 플랜트 (적합값) | 9.8 | 0.55 | −4.34 | 0.910 | 1.11 | — |
   | joint μ=3 플랜트 + structured EM | 9.4 | 0.54 | −4.34 | 0.910 | 1.12 | 7 |

   읽기: (i) sup 플랜트에서 EM 은 sup 이 상한에 붙여 놓은 $q_a$ 를 1/5 로 내리고 우도를 크게 올리며 (−2.05 → −3.09) NIS 를 0.17 → 0.60 으로 정상 쪽으로 옮긴다. pitch 는 0.908 → 0.874 로 0.03 만 잃는다. full EM (항목 10, 0.43) 과 달리 pitch 가 유지되는 이유는 pitch 의 나눔 규칙 ($q_p$, $q_g$, $R$) 을 EM 손에서 뺐기 때문이다. (ii) joint 플랜트에서는 EM 이 거의 움직이지 않는다 ($q_a$ 9.8 → 9.4): joint 의 $q_a, q_z$ 는 이미 우도 최적이었다는 뜻이고, joint 가 "라벨이 정하는 항목은 라벨로, 우도가 정하는 항목은 우도로" 를 한 목적함수 안에서 이미 해 놓았다는 확인이다. (iii) 14회·7회 만에 수렴하는 것은 두 파라미터만 남아 우도 능선이 없기 때문이다. **결론: "sup + EM" 을 올바르게 하는 형태는 이것이다 — 우도가 결정할 수 있는 잡음만 EM 에 맡긴다. 성능은 joint 와 같고 (0.874–0.910), 방식은 표준 EM 의 부분집합이라 설명이 쉽다.**
13. **합성 데이터 식별성 검증 (2026-09-14, `check_em_identifiability_synthetic.py`, `outputs/em_identifiability_synthetic_g_physical_sup_*.json`).** "EM 이 틀린 것인가, 이 관측 구성에서 $Q, R$ 이 식별되지 않는 것인가" 를 가르기 위해 g_physical sup 의 플랜트·$Q$·$R$ 을 참값으로 두고 실제 토크 입력으로 센서 200 에피소드를 생성한 뒤, $Q \times 3, R \times 1/3$ 과 $Q \times 1/3, R \times 3$ 두 시작점에서 full EM 을 돌렸다 (tol 1e-6, 465·474회 수렴).

   | | energy | corr($\hat q$, 참 $q$) | $Q$ 회복 비율 [$a_x$, $q$, 구배, $w_s$] | $R$ 회복 비율 [$\bar v_w$, $a_–x$, $a_z$, $\Delta v_w$] |
   |---|---:|---:|---|---|
   | 참 $Q, R$ (기준) | −0.3772 | 0.734 | 1 | 1 |
   | 시작 $Q\times3, R/3$ → EM | −0.3772 | 0.732 | 1.02, 0.97, 1.5, 1.00 | 1.00, **0.33**, **0.33**, 1.00 |
   | 시작 $Q/3, R\times3$ → EM | −0.3771 | 0.734 | 1.00, 1.01, 0.63, 1.00 | 1.00, **4.2**, **3.6**, 1.00 |

   읽기: (i) 모델이 맞는 데이터에서는 EM 이 우도를 참값 수준까지 올리고 pitch 구동 $q_p$ 를 1–3 % 안에서 회복하며, EM 의 $Q, R$ 로 만든 필터가 참 $Q, R$ 필터만큼 pitch 를 복원한다 (0.732–0.734 vs 0.734). **EM 과 모델 구조는 맞다.** (ii) 그러나 IMU 측정잡음 $r_x, r_z$ 는 시작값에 그대로 남는다 (1/3 → 0.33, 3 → 4.2): 우도가 이 두 항목에 대해 평탄하다는 뜻이고, 모델이 맞아도 원리적으로 식별되지 않는 항목이다 (IMU 지연 상태 $a_I$ 와 잡음이 센서 예측에서 구별되지 않음). 구배 잡음 $q_g$ 도 1e-6 수준이라 식별되지 않는다 (1.5 / 0.63). (iii) 합성에서는 $r_x, r_z$ 가 틀려도 pitch 가 유지되는데 실제 데이터에서는 $r_x, r_z \to 0$ 과 함께 pitch 가 무너진다. 차이는 데이터뿐이므로 **실제 데이터에서의 실패는 EM 이 아니라 모델 불일치 (유색 측정잡음·미모델 동역학) 탓**이다: 실제 잔차의 시간 상관을 백색잡음 모델이 설명하려면 $R \to 0$ 에 full $Q$ 로 옮기는 수밖에 없고, 그 과정에서 항목 10 의 pitch 붕괴가 일어난다. (iv) 따라서 항목 12 의 처방 (식별 안 되는 $R$, $q_p$, $q_g$ 는 고정, 식별되는 $q_a, q_z$ 만 EM) 이 합성 검증과 정확히 일치한다.
14. **교대 최적화를 structured EM 으로 (2026-09-14, 목적함수 `alts:<n>`).** 항목 11 의 alt 에서 EM 을 항목 12 의 structured EM 으로 바꾼 것: $R$, $q_p$, $q_g$ 는 sup 적합값에 고정, [$q_a, q_z$ 만 EM → 플랜트만 sup] 을 3회 교대, sup 플랜트 warm start. 라운드별 (EM energy → sup NRMSE, $q_a$, $q_z$): (−3.07 → 0.496, 10.9, 7.5), (−3.06 → 0.505, 7.2, 3.3), (−3.06 → 0.505, 7.1, 3.2) — 2회 만에 자리를 잡는다.

   | g_physical | pitch corr | NIS | energy (dev) | 비고 |
   |---|---:|---:|---:|---|
   | sup | 0.908 | 0.17 | −2.03 | $q_a$ 상한 |
   | structured EM 단독 (sup 플랜트, 항목 12) | 0.874 | 0.60 | — | 플랜트 그대로 |
   | **alts:3** | **0.893** | 0.61 | −3.02 | 플랜트 재적합: $b_a$ 0.17 → 0.26, $f_z$ 상한, $\lambda_a$ 하한 |
   | alt:3 (full EM, 항목 11) | 0.536 | 1.29 | −4.26 | 실패 |
   | joint μ=3 | 0.910 | 1.11 | −4.11 | |

   읽기: (i) full EM 교대 (0.536) 와 달리 structured EM 교대는 동작한다. sup 단계가 EM 잡음 아래에서 NRMSE 0.50 을 찾고 (alt 는 1.1), 플랜트를 다시 맞춰 structured EM 단독보다 +0.02 를 회복한다. (ii) 그래도 joint 보다 0.017 낮고 NIS 는 0.61 로 joint (1.11) 보다 비일관하다. 이유는 $R$, $q_p$, $q_g$ 가 sup 값 (라벨 맞춤용 성형값, $q_a$ 상한 시절의 짝) 에 묶여 있기 때문이다. joint 는 그 항목까지 우도와 라벨이 함께 정한다. (iii) 정리하면 "sup + EM" 의 올바른 형태 (structured, 교대) 는 0.89 / NIS 0.6, joint 는 0.91 / NIS 1.1 — 분리 절차가 필요한 상황 (플랜트는 한 번 식별하고 잡음만 현장에서 갱신) 이면 alts, 그렇지 않으면 joint.
15. **bounce 라벨을 목적함수에 추가 (2026-09-14, 목적함수 `joint2:<μ_p>:<μ_b>`).** $J = \varphi + \mu_p\,\mathrm{NRMSE}_{pitch} + \mu_b\,\mathrm{NRMSE}_{bounce}$. bounce 항은 bounce 속도 상태 $w_s$ 를 `Bounce_rate_6D` 와 자유 이득·오프셋으로 맞춘 오차 (라벨 단위 미확정). g_physical, $\mu_p = 3$, $\mu_b = 3 / 10$. 이번부터 모든 행에 `bounce_corr` (dev-test, $w_s$ 자유 이득 회귀) 를 기록한다.

   | g_physical | pitch corr | bounce corr | NIS |
   |---|---:|---:|---:|
   | sup | 0.908 | 0.250 | 0.17 |
   | ml | 0.539 | 0.082 | 1.08 |
   | joint μ=3 | 0.910 | 0.253 | 1.11 |
   | alts:3 | 0.893 | 0.211 | 0.61 |
   | joint2 μ_p=3, μ_b=3 | 0.910 | 0.302 | 1.11 |
   | joint2 μ_p=3, μ_b=10 | 0.909 | 0.305 | 1.11 |
   | 참고: 1-DOF bounce KF (§3) / 표준 3-상태 수직채널 | — | 0.918 / 0.938 | |

   읽기: bounce 라벨 항을 넣어도 bounce corr 은 0.25 → 0.30 에 그치고 ($\mu_b$ 를 3 → 10 으로 올려도 변화 없음), pitch 는 0.910 그대로다. 파라미터도 joint 와 거의 같다 ($f_z$ 1.5 Hz, $\zeta_z$ 0.4 상한, $c_{zp}$ −15). 즉 물리 범위 (1.2–1.8 Hz 진동자) 안의 bounce 속도 $w_s$ 로는 라벨 파형을 만들 수 없다. `Bounce_rate_6D` 는 bounce 속도가 아니라 칩이 $a_z$ 를 0.77 Hz 고역통과 후 적분한 파생 신호이고, 그것을 맞추는 모델은 사실상 같은 필터인 1-DOF KF (0.918) 나 3-상태 수직채널 (0.938) 이다. **결론: 이 모델은 pitch 전용으로 두고, bounce 는 기존 1-DOF KF 가 담당한다. bounce 추정 불가가 아니라 라벨 정의 불일치다.**
16. **Abbeel 외 2005 "Discriminative Training of Kalman Filters" 재현 (2026-09-14, 목적함수 `res:<source>`, `reso:<source>`, `pred:<P>:<source>`, `--optimizer coord`).** 논문 설정 = 플랜트 (f, g) 고정, 잡음 공분산만 부분 라벨 (논문은 GPS 위치, 우리는 pitch rate) 기준으로 학습. 우리 sup 과의 차이는 (i) 플랜트까지 학습하느냐, (ii) 손실 형태 (논문 Res 는 offset 없는 제곱합, 우리는 평균 정렬 NRMSE — 정규화는 상수배라 argmin 동일), (iii) 공분산까지 보는 Pred 기준, (iv) 최적화기 (논문 좌표 상승 vs Powell). 재현: g_physical 의 joint / sup 플랜트 고정, 잡음 세기 8개 ($q_a, q_p, q_g, q_z, r_w, r_x, r_z, r_d$, 물리 범위 유지) 만 학습, 출처 적합값에서 출발.

   $$\mathrm{Res}: \ \frac{1}{NT}\sum (q^{6D}_t - 57.3\,\hat q_t)^2 \qquad \mathrm{Pred}: \ \frac{1}{NT}\sum \Big[\log \Omega_t + \frac{(q^{6D}_t - 57.3\,\hat q_t)^2}{\Omega_t}\Big], \quad \Omega_t = 57.3^2\, P_{t|t}[q,q] + P$$

   $P_{t|t}$ 는 선형 시불변이라 데이터와 무관한 Riccati 재귀 (`posterior_variance`), $P$ (기준 센서 잡음 분산) 는 논문대로 고정 (0 또는 1 (deg/s)²). 새 지표 `label_logloss` = dev-test 에서 $P = 0$ 인 Pred 손실 (필터가 말하는 pitch 분산이 실제 오차와 맞는가, 낮을수록 정직).

   | g_physical | 플랜트 | 학습 | pitch corr | RMSE | NIS | label_logloss | 센서 energy (dev) | 비고 |
   |---|---|---|---:|---:|---:|---:|---:|---|
   | sup | 자유 | 전부 | 0.908 | 1.78 | 0.17 | 3.50 | −2.03 | 필터 σ_q 5.2 deg/s vs RMSE 1.8: 불확실성 3배 과대 |
   | joint μ=3 | 자유 | 전부 | 0.910 | 1.85 | 1.11 | **2.35** | **−4.11** | σ_q 2.0 ≈ RMSE 1.85: 보정됨 |
   | ml | 자유 | 전부 | 0.539 | 9.48 | 1.08 | 32.8 | −4.66 | σ_q 2.0 vs RMSE 9.5: 과신 |
   | Res (joint 플랜트) | 고정 | 잡음 8 | 0.910 | 1.79 | 0.36 | 2.68 | −2.32 | $q_a$ 9.8→27, $q_z$ 0.55→28, $r_d$ 2배: 센서 불신 방향 |
   | Res + offset (reso) | 고정 | 잡음 8 | 0.910 | 1.79 | 0.37 | 2.68 | −2.29 | offset 유무 무관 (GPT 우려 해소) |
   | Pred P=0 (joint 플랜트) | 고정 | 잡음 8 | 0.911 | 1.79 | 0.95 | **2.35** | −1.00 | $q_z \to 403$ 상한, $\sigma_w$ 절반 |
   | Pred P=1 (joint 플랜트) | 고정 | 잡음 8 | 0.910 | 1.79 | 1.40 | 2.54 | −0.67 | |
   | Res @coord / Pred @coord | 고정 | 잡음 8 | 0.911 / 0.910 | 1.79 | 0.46 / 0.98 | 2.52 / 2.37 | −2.60 / −0.91 | 좌표 상승 = Powell (손실 3.721 vs 3.719, 2.150 vs 2.149) |
   | Res (sup 플랜트) | 고정 | 잡음 8 | 0.908 | 1.78 | 0.18 | 3.48 | −2.04 | sup 잡음에서 안 움직임 (이미 Res 최적) |
   | Pred P=0 (sup 플랜트) | 고정 | 잡음 8 | 0.906 | 1.82 | 1.12 | 2.41 | −2.26 | |

   읽기: (i) **pitch 정확도는 라벨 기준이 무엇이든 0.906–0.911 로 같다.** 좋은 플랜트 위에서는 sup / joint / Res / Pred 가 같은 천장에 닿는다. (ii) **Pred 는 논문대로 공분산 보정에서 최고** (label_logloss 2.35, NIS 0.95): Res (2.68) 와 sup 플랜트의 Res (3.48 = sup) 보다 낫다. 논문의 "Res 는 오차, Pred 는 로그손실" 이 재현된다. joint 도 같은 2.35 — 우도 항이 pitch 공분산 보정을 공짜로 해 준다. (iii) **그러나 Res·Pred 모두 센서 우도를 망친다** (energy −2.3 / −1.0 vs joint −4.1): 라벨이 안 보는 방향 ($q_z$ → 상한 403, $q_a$·$r_d$ 팽창) 이 평탄해서 파라미터가 흘러간다. 부분 라벨 판별 학습의 구조적 문제이며, 논문에서는 라벨 (GPS 위치) 을 센서가 직접 관측해 이 방향이 작았을 것이라는 것이 우리 가설이다 (논문의 주장은 아님). joint 는 이 방향을 우도 항으로 잡는다. (iv) offset 과 최적화기는 결과에 영향 없음. **결론: Abbeel 재현은 성공적이고 (Pred 의 보정 효과 확인), 우리 joint 는 "Res/Pred 를 플랜트 식별로 확장 + 우도 정규화로 라벨 사각지대를 채운 것" 으로 위치 짓는다.**
17. **완전 EM — 논문 (arXiv 2105.00250) 식 (27)–(32) 그대로, 플랜트까지 자유 (2026-09-14, `run_em_noise_covariance.py --variants system`, `outputs/em_noise_covariance_system_*`).** $[A\ B]$, $[H\ D]$ 를 스무더 모멘트의 최소제곱 해로, $Q, R$ 을 식 (30)–(31) 로 매 반복 갱신. 모든 원소 자유 (물리 구조 없음), $m_0$ (에피소드별 초기값) 와 $P_0$ 만 고정. 라벨 미사용. 상태가 닮음변환까지만 정해지므로 pitch 는 두 가지로 평가: 4번째 상태를 그대로 (`corr`), train 라벨로 회귀한 9-상태 선형 판독 (`corr_readout`, 라벨 사용).

   | 출발 플랜트 | 반복 (tol 1e-6) | energy (fit) | \|ΔA\|/\|A\| | pitch corr (4번째 상태) | 선형 판독 corr | NIS |
   |---|---:|---:|---:|---:|---:|---:|
   | g_physical joint (출발값) | 0 | −4.34 | 0 | 0.910 | 0.915 | 1.11 |
   | → 완전 EM | 1,112 | **−6.32** | 1.56 | 0.483 | 0.702 | 1.15 |
   | g_physical sup (출발값) | 0 | −2.05 | 0 | 0.908 | 0.904 | 0.17 |
   | → 완전 EM | 485 | **−6.36** | 1.34 | 0.583 | 0.734 | 1.16 |

   읽기: (i) 센서 우도는 지금까지 중 압도적 최고 (−6.3; 물리 구조 안의 최고 ml 이 −4.7). 자유 9-상태 선형 모델은 4개 센서를 거의 무잡음 ($R \sim 10^{-5}$) 으로 설명하는 부분공간 모델로 수렴한다. (ii) 플랜트는 출발점의 흔적이 없을 만큼 바뀌고 (\|ΔA\|/\|A\| > 1), 4번째 상태는 더 이상 pitch 가 아니다 (0.48–0.58). (iii) 라벨로 9-상태의 최적 선형 판독을 붙여도 0.70–0.73 이다. 우도가 고른 잠재 상태는 센서 예측용 압축 표현이지 pitch 를 담도록 만들어진 것이 아니고, 물리 모델의 pitch 상태 (0.91) 는 물론 인과 FIR 상한 (0.96) 에도 못 미친다. (iv) 두 출발점이 다른 국소해로 갔지만 (energy −6.32 / −6.36, 판독 0.70 / 0.73) 결론은 같다. **결론: 논문 §2.2 를 글자 그대로 (플랜트 포함) 적용하면 "센서를 가장 잘 설명하는 선형 모델" 을 얻지만 pitch 추정기로는 쓸 수 없다. 우리가 EM 을 $Q, R$ 에 한정하고 플랜트를 물리 구조로 묶은 이유가 여기서 수치로 확인된다.**
18. **구배 상태 $\gamma_g$ 제거 — 보고용 기준 모델 `h_nograde` (2026-09-15, `methods_gpt.md` §3-1).** 9-상태 모델의 관측 불가능 방향은 (θ 상수 ↔ $\gamma_g$) 하나뿐이고 pitch rate 는 functionally observable (Fernando–Trinh–Jennings 2010: $\mathrm{rank}[\mathcal O; e_q^T] = \mathrm{rank}\,\mathcal O$) 이지만, 설명을 단순하게 하기 위해 $\gamma_g$ 를 뺀 8-상태 (fully observable) 를 보고 기준으로 삼는다. 구현은 $q_g = 0$, $P_0[\gamma_g] = 0$ (γ_g ≡ 0, 8-상태와 동치).

   먼저 g_physical joint 플랜트에서 필터만 다시 돌려 $\gamma_g$ 의 몫을 잰 결과:

   | 설정 | pitch corr | RMSE | NIS | θ 의 에피소드 평균 오프셋 |
   |---|---:|---:|---:|---:|
   | 적합값 그대로 ($q_g$ = 1.4e-5) | 0.910 | 1.85 | 1.11 | 0.0009 rad |
   | $q_g \to 0$ (에피소드마다 상수, 초기값만 추정) | 0.905 | 1.95 | 1.13 | 0.0013 |
   | $\gamma_g$ 제거 ($q_g = 0$, $P_0 = 0$) | 0.884 | 2.47 | 1.22 | 0.0037 |
   | $q_g \times 100$ | 0.808 | 2.45 | 1.06 | 0.0011 |

   읽기: $\gamma_g$ 의 역할은 IMU 종가속의 저주파 오프셋 (경사·바이어스) 을 θ 대신 받아 주는 것이다. 없애면 오프셋이 θ 로 가고 (평균 오프셋 4배) 그 변동이 q 로 새어 corr 0.03·RMSE 30 % 를 잃는다. 에피소드당 상수 하나면 거의 손실이 없고, 반대로 $q_g$ 를 키우면 pitch 의 저주파까지 가져간다. 재적합 결과 (sup / ml / joint / Res / Pred / alts / EM) 는 아래에 추가.

   **재적합 시 주의 (cold start 실패).** $\gamma_g$ 를 뺀 모델을 기본 시작값에서 적합하면 sup 0.708 ($r_w$ 상한 1 로 휠속을 꺼 버림), ml −0.248 (부호 반전), joint 0.134 로 전부 무너진다. 같은 모델에서 g_physical 의 파라미터를 그대로 쓰면 0.884 이므로 모델의 한계가 아니라 최적화 지형의 문제다: $\gamma_g$ 가 있으면 IMU 오프셋을 받아 주는 자리가 있어 시작값에서도 옳은 골짜기로 들어가지만, 없으면 초기에 그 오프셋을 어디에 둘지 갈피를 못 잡는다. 따라서 `h_nograde` 는 `--warm g_physical` (같은 목적함수의 g_physical 적합값에서 출발) 로 적합한다. 또 $\gamma_g$ 의 분산을 정확히 0 으로 두면 RTS 스무더 (EM) 의 예측 공분산이 특이해지므로 무시 가능한 ε (초기 분산 1e-10, 구동 잡음 1e-16) 만 남겨 얼린다.

   **재적합 결과: $\gamma_g$ 제거는 채택하지 않는다.** g_physical 적합값에서 warm start 해도 h_nograde 는 sup 0.694 (RMSE 3.05, $r_w \to 1$, NIS 15.5), ml −0.218, joint 0.817 (RMSE 3.05, free gain 36), Res 0.806, Pred 0.791, alts 0.645, EM full 0.52–0.69 로 g_physical (0.908–0.911, RMSE 1.8) 보다 크게 나쁘다. 위 "필터만 재실행" 표의 0.884 가 오해였다: `metrics()` 의 corr 은 에피소드별 평균을 뺀 값이라 **$\hat q$ 의 에피소드별 상수 오프셋을 숨긴다.** $\gamma_g$ 가 없으면 IMU 의 저주파 오프셋 (경사·바이어스) $g\theta_0$ 를 θ 가 떠안아야 하는데, θ 를 상수 $\theta_0$ 에 붙들어 두려면 pitch 식 $\dot q = -\omega_p^2\theta - 2\zeta_p\omega_p q + \dots$ 에서 $\hat q \approx -\omega_p\theta_0/(2\zeta_p)$ 의 **상수 pitch-rate 편향**이 생긴다 (θ₀ = 1° 이면 ω_p = 10, ζ_p = 0.15 에서 33 deg/s). 같은 파라미터에서 fit NRMSE 가 g_physical 0.48 → γ_g 제거 1.41 로 뛰는 것이 그 증거이고, corr 은 이를 못 본다. 최적화기는 이 편향을 줄이려고 휠속 채널을 끄거나 ($r_w \to 1$: a_x 가 오프셋을 대신 흡수) 부호를 뒤집는 나쁜 골짜기로 간다. 결론: **$\gamma_g$ (또는 동치인 IMU 종가속 오프셋 상태) 는 필수**이며, 관측 불가능 방향 (θ 상수 ↔ γ_g) 은 pitch rate 와 직교하므로 functional observability 로 설명하고 유지한다. 최소 모델 실험 (항목 19) 도 γ_g 를 포함한 채 진행한다.

   **$c_{zp}$ (pitch → bounce 연성) 은 무관한 파라미터** (아래 표, 이 결론은 항목 19 의 사다리로 확장됨). g_physical 의 joint / sup 플랜트에서 $c_{zp}$ 를 적합값 (±15, 경계) → 0 → 부호 반전으로 바꿔 필터만 다시 돌리면 corr / RMSE / NIS / energy 가 소수 셋째 자리까지 같다 (joint 0.910 / 1.85 / 1.11 / −4.11 불변). 앞뒤 강성 비대칭 $(k_f l_f - k_r l_r)/m$ 에 해당하는 항이지만 이 데이터에서는 식별도 안 되고 (sup −15, joint +15 로 갈림) 결과에도 영향이 없으므로, 보고용 모델에서는 $c_{zp} = 0$ 으로 두어 두 진동자를 완전히 분리한 형태로 적는다.
19. **기능 제거 사다리 → 5-상태 최소 모델 (2026-09-15, `build_m`, stage `m`/`n`/`p`, `methods_gpt.md` §3-2).** §3 (g_physical) 의 joint 플랜트에서 요소를 하나씩 끄고 필터만 재실행한 스크리닝 (corr: $c_{zp}$ 0 → 0.910, slip $s_f,s_r$ 0 → 0.910, 토크 입력 제거 → 0.908, $b_a$ 0 → 0.909, $x_I$ 0 → 0.911, $a_z$ 채널 끔 → 0.910, $\tau_I$ → 5 ms → 0.910, $\kappa$ 0 → 0.904, $\Delta v_w$ 끔 → 0.825, $h_I$ 0 → −0.518) 에 따라, bounce 진동자·$a_z$ 채널·토크 입력·$b_a$·$c_{zp}$ 를 한꺼번에 뺀 6-상태 [$v_x, a_x, \theta, q, \gamma_g, a_I$], 관측 [$\bar v_w, a_x^{IMU}, \Delta v_w$] 를 기준형 (m6) 으로 두고 재적합으로 확인했다. 전부 g_physical 적합값에서 warm start, 물리 범위 유지.

   | 모델 | 상태 | 파라미터 | sup corr / RMSE | joint corr / RMSE / NIS | 판정 |
   |---|---:|---:|---|---|---|
   | g_physical (§3) | 9 | 21 | 0.908 / 1.78 | 0.910 / 1.85 / 1.11 | 출발점 |
   | m6 (bounce·$a_z$·토크·$b_a$·$c_{zp}$ 제거) | 6 | 11 | 0.903 / 1.84 | 0.912 / 1.85 / 1.07 | 손실 없음 |
   | m6 + $b_a$ | 6 | 12 | 0.898 / 1.84 | 0.910 / 1.87 / 1.07 | 불필요 |
   | m6 + $\lambda_a$ | 6 | 12 | 0.904 / 1.85 | 0.912 / 1.86 / 1.07 | 불필요 |
   | m6 − $\kappa$ | 6 | 10 | 0.902 (NIS 68) | 0.905 / 1.86 / 1.04 | 소폭 손실, sup 불안정 |
   | **m5_nodelay = m6 − IMU 지연 상태** | **5** | **10** | **0.908 / 1.76** | **0.916 / 1.80 / 1.08** | **최고, 보고 기준** |
   | m6 − $\Delta v_w$ 채널 | 6 | 9 | 0.817 / 2.40 | 0.824 / 2.46 / 1.09 | 필수 |
   | m6 − $\gamma_g$ | 6 | 10 | 0.843 / 2.63 (NIS 11) | 0.471 / 3.50 | 필수 (항목 18) |
   | m5_nodelay − $\kappa$ | 5 | 9 | 0.899 / 1.85 | 0.912 / 1.82 / 1.05 | 소폭 손실 (−0.004), 유지 권장 |
   | m5_nodelay − $\gamma_g$ (4-상태) | 4 | 9 | 0.871 / 2.44 | 0.237 / 3.85 | 필수 재확인 (m5 값에서 warm start 해도) |

   읽기: (i) 성능을 만드는 것은 셋뿐이다 — IMU 종가속의 높이 레버암 $h_I$ (pitch 각가속도를 종가속에 드러냄), 휠속 앞뒤 차이 $\Delta v_w$ (q 의 부호·위상), 구배 상태 $\gamma_g$ (IMU 저주파 오프셋의 배출구). (ii) IMU 1차 지연 상태는 $h_I$ 가 있으면 필요 없다. $-h_I\dot q$ 항이 q 보다 90° 앞선 성분을 주어 지연이 만드는 위상을 대신 흡수한다 (사다리 초기의 지연 +0.15 는 $h_I$ 가 없을 때의 값). (iii) bounce·$a_z$·토크·$b_a$·$c_{zp}$·$\lambda_a$ 는 제원 범위 안에서는 기여가 없다. 성능 상승을 위해 붙였던 것들이 사실은 자유 $x_I$ 등이 라벨 위상을 흉내 낼 때만 도움이 됐던 것이다. (iv) 5-상태 모델의 Abbeel Res / Pred (joint 플랜트 고정): 0.912 / 0.915, label_logloss 2.34 / 2.31.

   **5-상태 모델의 방법별 결과 (보고 표, dev-test 128 에피소드):**

   | m5_nodelay | pitch corr | RMSE | NIS | label_logloss | 비고 |
   |---|---:|---:|---:|---:|---|
   | sup | 0.908 | 1.76 | 0.17 | 3.50 | 공분산 과대 |
   | ml (라벨 없음) | 0.574 | 9.71 | 1.03 | 68.8 | free gain 7.6: q̂ 7배 과대 |
   | joint μ=3 | **0.916** | 1.80 | 1.08 | 2.32 | 채택 |
   | Abbeel Res / Pred (joint 플랜트, 잡음만) | 0.912 / 0.915 | 1.76 / 1.79 | 1.31 / 1.25 | 2.34 / 2.31 | |
   | EM full 수렴 (sup / joint 플랜트) | 0.562 / 0.571 | | 1.03 / 1.03 | | 1,590 / 2,084회, R → 0 으로 퇴화 (§3 과 같은 결론) |
   | 참고: 사다리 변형들의 ml | m6 0.529, m6−κ 0.515, m6−Δv_w −0.151, m6−γ_g −0.071 | | | | Δv_w·γ_g 없으면 라벨-프리는 부호조차 못 잡음 |

   그림: `outputs/pitch_staged_compare_m5_nodelay.png`, `outputs/pitch_staged_diagnostics_m5_nodelay.png`, `outputs/pitch_staged_compare_g_physical_vs_m5_nodelay_joint3.png`.
20. **pitch ⊕ bounce 결합 (2026-09-15, `build_pb`, `--models pb1_block,pb2_*`, `methods_gpt.md` §3-3).** 5-상태 pitch 모델 (§3-2) 옆에 bounce 블록을 붙이고 관측에 $a_z^{IMU}$ 를 더한다. 목적함수 joint2 (우도 + 3·pitch NRMSE + 3·bounce NRMSE), pitch 파라미터는 m5 적합값에서 출발. bounce 는 자유 이득으로 `Bounce_rate_6D` 와 대조. 두 버전:
   - **1번 `pb1_block`**: 기존 1-DOF bounce KF 구조 그대로 (진동자 $f_b, \zeta_b$ 넓은 범위 + random-walk 외란 $d_b$ 가 $\ddot z_s$ 구동), 교차항 0 (블록 대각). bounce 출력 = $\dot z_s$. 칩 필터를 파라미터로 흉내 내는 방식.
   - **2번 `pb2_*`**: 물리 bounce 진동자 ($f_b$ 1.2–1.8 Hz, $\zeta_b$ 0.15–0.4, 백색잡음 구동) 에 **칩 처리 체인을 상태 하나로 붙임**: $\dot b = -\omega_c b + \ddot z_s$, $\omega_c = 2\pi\cdot0.77$ (회귀값 고정). 즉 `Bounce_rate_6D` $\approx K\cdot\mathrm{HP}_{\omega_c}[\int a_z]$ 를 모델 출력 $b$ 로 만들어 라벨과 직접 비교 → 물리 bounce 상태가 라벨로 검증된다. 기본형에서 하나씩 추가하고 뚜렷한 이득이 있는 것만 남긴다.

   | 모델 | 상태 | 파라미터 | pitch corr / RMSE | bounce corr | NIS | 판정 |
   |---|---:|---:|---|---:|---:|---|
   | m5_nodelay (pitch 만, 참고) | 5 | 10 | 0.916 / 1.80 | — | 1.08 | |
   | 기존 1-DOF bounce KF 단독 (참고, §3) | 3 | 5 | — | 0.918 | | |
   | 1번 pb1_block | 8 | 15 | 0.916 / 1.80 | 0.905 | 1.11 | 두 필터를 따로 돌린 것과 동치. pitch 불변 |
   | **2번 pb2_basic (물리 진동자 + 칩 체인)** | **8** | **14** | **0.916 / 1.80** | **0.923** | 1.10 | **채택.** $f_b$ 1.8 Hz (상한), $\zeta_b$ 0.4 (상한) |
   | 2번 + 교차항 $c_{zp}, c_{pz}$ | 8 | 16 | 0.918 / 1.75 | 0.923 | 1.11 | $c_{zp}$ −14.5·$c_{pz}$ +15 로 둘 다 경계, $f_b$ 하한으로 이동. +0.002 는 이득 아님. 제외 |
   | 2번 + $\omega_c$ 자유 | 8 | 15 | 0.916 / 1.80 | 0.923 | 1.10 | $f_c$ = 0.75 Hz (회귀 0.77 재현), 변화 없음. 고정 유지 |
   | 2번 + IMU 전방 레버 $x_I$ = 0.42 ($a_z^{IMU}$ 에만) | 8 | 14 | 0.902 / 2.05 | 0.813 | 1.12 | 둘 다 악화 (free gain 70, $h_I$ 하한). 제외 |

   읽기: (i) 물리 bounce 진동자에 칩 체인 출력을 붙이면 `Bounce_rate_6D` 를 0.923 으로 재현한다 — 칩 필터를 흉내 내던 1-DOF KF (0.905–0.918) 와 같거나 낫고, 이제 bounce 상태 $z_s, \dot z_s$ 는 물리량이다. 라벨 = "차체 bounce 가속도를 0.77 Hz 고역통과 후 적분" 이라는 회사 설명이 모델 안에서 그대로 성립한다. (ii) 동역학적 결합 (교차항) 은 식별되지 않고 (경계로 감) 이득도 없다. 승용차의 분리 조건 ($k_f l_f \approx k_r l_r$) 과 일치. 최종 모델은 pitch 블록과 bounce 블록이 동역학적으로 독립이고, 관측만 나눠 갖는다. (iii) IMU 전방 레버 $x_I \dot q$ 를 $a_z^{IMU}$ 에 넣으면 나빠진다. 라벨에 레버 흔적이 없었던 것과 같은 방향의 사실이며, IMU 의 $a_z$ 가 pitch 각가속도에 우리 모델대로 반응하지 않거나 (부호·크기) 다른 처리가 있다는 뜻. 제원 확인 항목. (iv) $\omega_c$ 는 자유로 두어도 0.75 Hz 로 회귀값을 재현하므로 고정한다. (v) bounce 진동자의 $f_b$, $\zeta_b$ 가 물리 범위 상한 (1.8 Hz, 0.4) 에 붙는다. 실측 자유감쇠 (1.65 Hz, 0.24) 보다 조금 높은 쪽을 원하며, 범위를 넓히면 더 움직일 수 있다 — 물리 해석의 한계로 기록.

   **결론.** 보고용 결합 모델 = `pb2_basic`: 상태 8개 $[v_x, a_x, \theta, q, \gamma_g, z_s, \dot z_s, b]$, 관측 4개 $[\bar v_w, a_x^{IMU}, \Delta v_w, a_z^{IMU}]$, 파라미터 14개, 출력 pitch rate $= (180/\pi) q$, `Bounce_rate_6D` $= K b$. dev-test pitch 0.916 / bounce 0.923. 관측 가능성은 §3-2 와 같고 ($\theta$ 상수 ↔ $\gamma_g$ 만 marginal, 두 출력 모두 그 방향과 직교), 식별은 pitch 블록 (§3-2) 과 bounce 블록 (3개 + 잡음 2개) 이 관측을 나눠 가져 서로 독립이다.
21. **기하 고정 half-car 로 결합 재검 (2026-09-15, `pb2_halfcar`).** 항목 20 의 "교차항 자유" 시험은 half-car 가 아니라 임의 교차항이었으므로, §4 의 2-자유도 pitch-plane half-car (sprung mass) 파라미터화로 다시 확인했다. 기하·질량을 GV60 제원으로 고정 ($l_f = l_r = 1.45$ m, $m = 2{,}300$ kg, $I_{yy} = m l_f l_r$) 하고 축별 휠레이트 $k_f, k_r$ 과 댐핑 $c_f, c_r$ 만 자유. 진동수·감쇠·교차항 (강성·댐핑 모두) 이 전부 이 넷에서 유도되므로 구조적으로 식별된다. 나머지 (센서 체인, $\gamma_g$, 칩 체인, 잡음) 는 §3-3 과 같고 파라미터 수도 14 로 같다. joint2, pb2_basic 값에서 warm start.

   | | pitch corr / RMSE | bounce corr | NIS | label_logloss | 적합값 |
   |---|---|---:|---:|---:|---|
   | pb2_basic (독립 진동자, §3-3) | 0.916 / 1.80 | 0.923 | 1.10 | 2.32 | $f_p$ 1.52, $f_b$ 1.8 (상한), $\zeta_b$ 0.4 (상한) |
   | pb2_halfcar (기하 고정 half-car) | 0.917 / 1.75 | 0.923 | 1.10 | 2.31 | $k_f$ 85, $k_r$ 113 kN/m; $c_f$ 7.4 kN·s/m, $c_r$ 0.3 (하한) → $f_b = f_p$ 1.48 Hz, $\zeta$ 0.18, $c_{zp}$ +17.9, 댐핑 교차 +4.5 |

   두 출력의 세부 (dev-test): pitch corr p10 0.803, lag +10 ms; bounce (칩 체인 $b$) corr p10 0.861, lag 0 ms, 0.4–1.5 Hz 대역 corr 0.985, 이득 10.05; 물리 $\dot z_s$ 를 라벨과 직접 비교하면 0.143 (독립 모델 0.259) — 라벨이 칩 출력이므로 낮은 것이 정상. 고정값의 출처는 `methods_gpt.md` §4-1 Parameters 표.

   읽기: (i) 성능은 독립 진동자와 같다. 결합을 물리적으로 정확히 파라미터화해도 pitch·bounce 어느 쪽도 오르지 않는다. (ii) 데이터가 정하는 것은 합 ($k_f + k_r$ → 1.48 Hz, $c_f + c_r$ → $\zeta$ 0.18) 이고, 차이는 약하게만 정해진다: $k_f/k_r$ = 0.75 (교차항 +18) 는 나왔지만 항목 20 에서 교차항이 결과에 영향이 없던 것을 고려하면 신뢰 구간이 넓고, 댐핑은 $c_r$ 이 하한 (0.3 kN·s/m) 으로 가서 앞뒤 분배가 전혀 식별되지 않는다. 즉 구조적으로는 식별 가능하나 실용적으로는 합만 식별된다. (iii) $I_{yy} = m l_f l_r$ (dynamic index 1) 과 $l_f = l_r$ 을 함께 두면 $f_p = f_b$ 가 강제된다. 독립 모델은 $f_p$ 1.52 / $f_b$ 1.8 로 갈라 놓았는데 half-car 는 1.48 로 묶였고 성능은 같으므로, 이 데이터가 두 진동수의 차이를 요구하지 않는다는 뜻이다. $I_{yy}$ 를 풀면 갈라지겠지만 $k$ 와의 교환이 다시 생긴다. **결론: 보고 모델은 §3-3 (독립 진동자) 유지. half-car 는 "분리 가정을 데이터가 뒷받침한다" 의 근거로만 쓴다 — 결합 계수를 물리로 묶어도 이득이 없고, 앞뒤 분배는 이 센서 구성으로 식별되지 않는다.** k, c 의 절대값이 필요하면 회사 제원으로 고정해야 한다.

결과: `outputs/pitch_staged_metrics.csv`, `outputs/pitch_staged_models_*.png`.

## 5.9 파라미터 추정 목적함수 정리

이 lab 에서 파라미터를 맞추는 데 쓴 방법은 다섯 가지다: `sup`, `ml`, `joint`, `aug`, EM. 모두 같은 필터 (§2) 위에서 돌고, **무엇을 최소화하느냐**만 다르다. 수치는 §5.8 에 있고 여기서는 정의와 성질만 정리한다.

### 공통 기호

- 파라미터 벡터 $p = (p_A, p_Q)$. 플랜트 파라미터 $p_A$ (고유진동수 $f_p$, 감쇠 $\zeta_p$, 레버암 $h_I, x_I, \ell$, IMU 지연 $\tau_I$, 하중이동 이득 $b_a$ 등) 가 $A, B, H, D$ 를 정하고, 잡음 파라미터 $p_Q$ ($\log q_\ast$, $\log r_\ast$) 가 $Q, R$ 을 정한다.
- 관측 $y_k$ (휠속·IMU 등 배포 시 쓸 수 있는 채널만), 입력 $u_k$ (토크), 라벨 $q^{6D}_k$ (`Pitch_rate_6D`, deg/s). 학습 에피소드 $n = 1, \dots, N$, 길이 $T$.
- 필터가 내는 것: one-step 예측 $\hat y_{k|k-1} = H x_{k|k-1} + D u_k$, innovation $\nu_k = y_k - \hat y_{k|k-1}$, innovation 공분산 $S_k = H P_{k|k-1} H^T + R$, 상태 추정 $x_{k|k}$, 그중 pitch rate $\hat q_k(p) = e_q^T x_{k|k}$ (rad/s, $q = \dot\theta$).
- **innovation 우도 (energy).** Kalman filter 의 prediction error decomposition (Särkkä & Svensson 2023, Thm 16.9) 으로 관측의 marginal likelihood 는 innovation 만으로 쓰인다. 샘플당 음의 로그우도를

$$\varphi(p) = -\frac{1}{N T'} \log p\big(y^{(1:N)}_{1:T} \mid p\big) = \frac{1}{N T'} \sum_{n=1}^{N} \sum_{k > k_0} \frac{1}{2}\Big[\nu_k^T S_k^{-1} \nu_k + \log\det S_k\Big] + \text{const}$$

  로 두고 부른다. 첫 $k_0 = 100$ 샘플 (초기 과도) 은 버리고 $T' = T - k_0$. 낮을수록 필터가 다음 샘플의 센서를 잘 예측한다. 코드: `innovation_metrics(...)["energy"]`.

- **라벨 오차 (NRMSE).** 출력 이득은 단위 변환 $g = 180/\pi$ 로 고정하고 (§5.6) offset $c$ 만 학습 평균으로 둔다.

$$\mathrm{NRMSE}(p) = \frac{1}{\sigma(q^{6D})}\sqrt{\frac{1}{NT}\sum_{n,k}\big(g\,\hat q_k(p) + c - q^{6D}_k\big)^2}, \qquad c = \overline{q^{6D}} - g\,\overline{\hat q(p)}$$

### 1. `sup` — 라벨 오차 최소화 (supervised)

$$p^{\mathrm{sup}} = \arg\min_p\ \mathrm{NRMSE}(p)$$

플랜트와 잡음 파라미터를 전부 라벨 오차로 움직인다. 시스템 식별 용어로는 필터를 하나의 출력 예측기로 보고 목표 신호에 대한 output error 를 줄이는 것이며, Kalman filter 의 확률 모델 (잡음 통계) 과는 무관한 기준이다. 따라서

- $Q, R$ 은 잡음 분산이 아니라 필터의 대역폭·위상을 라벨에 맞추는 성형 손잡이가 된다 (§5.8-8 에서 EM 으로 잡음 통계로 되돌리면 corr 이 떨어지는 이유).
- innovation 은 백색이 아니고 $S_k$ 는 실제 예측 오차 크기를 대표하지 않는다 (`outputs/pitch_staged_diagnostics_*.png`). 필터가 내놓는 $P$ 를 다른 모듈이 믿고 쓰면 안 된다.
- pitch rate 자체는 이 기준을 직접 최소화하므로 가장 좋다. 라벨이 있는 차량에서만 가능하다.

Optimizer: bounded Powell + best-visited 재시작 (`pitch_staged_reconstruction.fit`).

### 2. `ml` — innovation 우도 최대화 (maximum likelihood)

$$p^{\mathrm{ml}} = \arg\min_p\ \varphi(p)$$

라벨을 쓰지 않는다. 선형 Gaussian 상태공간 모델의 표준 ML 식별 (prediction error method; Ljung 1999 Ch. 7, Särkkä & Svensson 2023 Ch. 16) 이다. 플랜트와 잡음을 같이 움직이며, 최적점에서 innovation 이 백색이고 $\mathrm{NIS} \approx 1$ 인 **일관된 필터**를 준다. 한계는 우도가 센서 채널 전체를 설명하는 기준이라는 것이다. pitch 가 센서에 남기는 정보량이 작으면 (§5.6, §5.8-3) $\hat q$ 의 스케일과 잡음 수준이 우도로는 결정되지 않는다. 라벨 없이 어느 차량에나 적용 가능하다.

### 3. `joint` — 벌점 우도 (penalized likelihood)

$$p^{\mu} = \arg\min_p\ \varphi(p) + \mu\,\mathrm{NRMSE}(p)$$

$\mu$ 는 라벨 항의 가중치 (무차원, $\mu = 3$–$10$ 사용). 해석은 두 방향이다. (i) 우도에 "이 필터의 용도는 pitch rate 복원" 이라는 사전 정보를 벌점으로 넣은 것 (identification for the intended use). (ii) `sup` 에 우도 정규화를 붙여 비물리 해 (파라미터 경계 표류, 잡음 성형) 를 막은 것. $\mu$ 에는 통계적 의미가 없고 검증 데이터로 고르는 손잡이다. 라벨이 필요하며, 배포 필터는 `sup` 과 같은 형태 (라벨 없이 $y_k$ 만) 다.

### 4. `aug` — 학습 시 라벨을 기준 관측으로 추가 (reference-sensor calibration)

학습 중에만 관측을 확장한다.

$$y^{+}_k = \begin{bmatrix} y_k \ q^{6D}_k \end{bmatrix},\qquad H^{+} = \begin{bmatrix} H \ g\,e_q^T \end{bmatrix},\qquad R^{+} = \begin{bmatrix} R & 0 \ 0 & r_{\mathrm{label}} \end{bmatrix},\qquad p^{\mathrm{aug}} = \arg\min_p\ \varphi^{+}(p)$$

$\varphi^{+}$ 는 확장 관측 $y^{+}$ 의 innovation 우도다. 배포 시에는 라벨 행을 떼고 $H, R$ 로 돌린다. 기준 센서를 같이 관측해 플랜트를 식별하는 교정 절차와 같은 구조이지만 이 lab 에서는 실패했다 (§5.8-7). 학습 중 필터가 라벨 행에서 $q$ 를 직접 읽어 버려 센서 행이 pitch 를 추출하도록 학습되지 않고, 라벨을 뗀 배포 필터는 학습된 필터와 다른 필터가 된다 (학습/배포 불일치).

### 5. EM — 플랜트 고정, $Q, R$ 만 (expectation-maximization)

`ml` 과 같은 우도 $\log p(y \mid Q, R)$ 를 최대화하되, 상태열 $x_{0:T}$ 를 잠재변수로 두고 완전 데이터 로그우도 $\log p(x_{0:T}, y_{1:T} \mid Q, R)$ 의 조건부 기대값을 반복 최대화한다 (Shumway & Stoffer 1982; arXiv 2105.00250 §2.2). 일반 EM-KF 는 $\theta = \{A, C, m_0, Q, R, P_0\}$ 전부에 닫힌 M-step 을 준다 (그 논문 식 27–32: $A, C$ 는 스무더 모멘트의 최소제곱 해). 단, 그 $A$ 갱신은 모든 원소가 자유인 행렬에만 성립하고, 그 논문의 실험도 $A, C$ 는 기지의 등가속도 모델로 두고 $Q, R, m_0, P_0$ 만 추정한다 (식 43 근방). 이 lab 도 플랜트 $A, B, H, D$ 를 제원이나 앞선 적합에서 가져와 고정한다. 이유는 아래 "`ml` 과의 관계" (ii) 와, 플랜트까지 자유로 둔 완전 EM 이 pitch 를 잃는다는 실측 (§5.8-17: 우도 −6.3 으로 최고이나 pitch 0.48–0.58, 라벨 판독으로도 0.70–0.73).

**E-step.** 현재 $Q, R$ 로 RTS smoother 를 돌려 $x_{k|T}$, $P_{k|T}$, 인접 시각 교차공분산 $P_{k,k-1|T} = P_{k|T} J_{k-1}^T$ 를 얻는다. $J_{k-1} = P_{k-1|k-1} A^T P_{k|k-1}^{-1}$ 은 smoother gain 이다.

**M-step (닫힌 형태).** $r_k = x_{k|T} - A x_{k-1|T} - B u_{k-1}$, $e_k = y_k - H x_{k|T} - D u_k$ 로

$$\hat Q = \frac{1}{T-1}\sum_{k=2}^{T}\Big[r_k r_k^T + P_{k|T} - P_{k,k-1|T} A^T - A P_{k,k-1|T}^T + A P_{k-1|T} A^T\Big]$$

$$\hat R = \frac{1}{T}\sum_{k=1}^{T}\Big[e_k e_k^T + H P_{k|T} H^T\Big]$$

에피소드가 여러 개면 합을 에피소드 평균으로 바꾼다. 성질:

- **단조성.** 매 반복 $\log p(y \mid Q, R)$ 가 감소하지 않는다 (Jensen 부등식).
- **`ml` 과의 관계.** Fisher's identity $\nabla_\theta \log p(y \mid \theta) = \mathbb{E}\big[\nabla_\theta \log p(x, y \mid \theta) \mid y\big]$ 에 의해 EM 의 고정점은 $\varphi$ 의 정류점과 같다. 두 방법은 **같은 기준의 다른 optimizer** 다. 차이는 (i) EM 은 $Q, R$ 에 대해 M-step 이 닫힌 형태라 미분·경계 없이 수 초에 수렴하고, Powell 의 경계 인공물 (하한에 붙은 $r_x$ 등) 이 없다. (ii) $A, H$ 가 $f_p, \zeta_p, h_I$ 등에 비선형이라 그 파라미터에는 닫힌 M-step 이 없다. 그래서 이 lab 의 EM 은 플랜트를 고정하고, `ml` 은 Powell 로 플랜트까지 움직인다.
- 라벨을 쓰지 않으므로 pitch 정보 부족은 `ml` 과 같다. 좋은 플랜트 (라벨이나 제원으로 잡은 것) 위에 EM 잡음을 얹는 것이 배포 형태의 표준 절차다.
- **수렴 속도.** 반복당 비용은 작지만 (300 에피소드 0.15 s) 우도 능선을 따라 느리게 움직여 수천 회가 필요하다. 60회에서 멈춘 값은 초기 $Q, R$ 근처의 과도값이며 (§5.8-10), 수렴 판정은 상대 변화 1e-6 이하로 한다.
- **식별되는 항목만 (structured EM).** 우도가 정할 수 없는 항목 ($R$, pitch 구동 $q_p$, 구배 $q_g$) 은 적합값·제원에 고정하고 $q_a, q_z$ 만 EM 으로 갱신하면 pitch 를 잃지 않으면서 (0.874–0.910) 우도·NIS 를 개선한다 (§5.8-12). 이것이 이 lab 에서 "sup + EM" 의 올바른 형태다.

코드: `run_em_noise_covariance.py` (`full` 전체 행렬 / `diag` 대각만).

### 6. `alt` — 플랜트는 `sup`, $Q, R$ 은 EM (교대 최적화)

$$p_A^{(k+1)} = \arg\min_{p_A}\ \mathrm{NRMSE}\big(p_A;\ Q^{(k)}, R^{(k)}\big), \qquad \big(Q^{(k+1)}, R^{(k+1)}\big) = \mathrm{EM}\big(p_A^{(k+1)}\big)$$

잡음 파라미터 $p_Q$ 는 탐색에서 빠지고 EM 행렬로 대체된다. 배포 형태 ("라벨로 플랜트 식별, 잡음은 라벨 없이 EM") 와 정확히 같은 절차이고 $\mu$ 같은 가중치가 없다. 그러나 두 단계가 다른 함수를 최소화하므로 하나의 목적함수가 아니며 수렴 보장이 없다. 이 lab 에서는 실패했다 (§5.8-11, corr 0.536 = `ml` 수준): 우도가 고른 $Q, R$ 아래에서는 어떤 플랜트도 pitch 를 복원하지 못한다.

### 7. `res`, `pred` — Abbeel 외 2005 의 판별 학습 (플랜트 고정, 잡음만)

$$\mathrm{Res}(p_Q) = \frac{1}{NT}\sum_{n,k}\big(q^{6D}_k - g\,\hat q_k\big)^2, \qquad \mathrm{Pred}(p_Q) = \frac{1}{NT}\sum_{n,k}\Big[\log \Omega_k + \frac{(q^{6D}_k - g\,\hat q_k)^2}{\Omega_k}\Big], \quad \Omega_k = g^2 P_{k|k}[q,q] + P$$

플랜트 $p_A$ 는 고정하고 잡음 $p_Q$ 만 움직인다. Res 는 `sup` 의 플랜트 고정판 (정규화·offset 차이는 argmin 에 무관), Pred 는 필터가 내놓는 pitch 분산 $P_{k|k}[q,q]$ 까지 라벨로 보정한다 ($P$ = 기준 센서 잡음 분산, 고정). 논문 원형은 GPS 위치 라벨로 EKF 의 Q, R 을 학습한 것이고, 부분 라벨 설정이 우리와 같다. 결과 (§5.8-16): pitch 는 `joint` 와 같고, Pred 는 공분산 보정 (`label_logloss`) 에서 `joint` 와 동급으로 최고이나, 라벨이 안 보는 잡음 방향은 평탄해 센서 우도를 망친다. `joint` 는 그 방향을 $\varphi$ 로 채운 것이다.

### 요약

| 방법 | 최소화 대상 | 라벨 | 움직이는 파라미터 | 필터 일관성 (백색 innovation, NIS≈1) | pitch rate 정확도 | 언제 쓰나 |
|---|---|:---:|---|:---:|:---:|---|
| `sup` | $\mathrm{NRMSE}$ | 필요 | $p_A, p_Q$ | ✗ | 최고 | 라벨 있는 차량에서 출력 정확도만 중요할 때 |
| `ml` | $\varphi$ | 불필요 | $p_A, p_Q$ | ○ | 낮음 | 라벨이 없거나, 공분산 $P$ 를 다른 모듈이 쓸 때 |
| `joint` | $\varphi + \mu\,\mathrm{NRMSE}$ | 필요 | $p_A, p_Q$ | ○ (근사) | 최고 수준 | 정확도와 일관성을 같이 원할 때 |
| `aug` | $\varphi^{+}$ | 학습 시만 | $p_A, p_Q$ | — | 실패 | 채택 불가 |
| EM | $\varphi$ ($Q, R$ 만) | 불필요 | $p_Q$ | ○ | `ml` 수준 | 플랜트가 정해진 뒤 잡음만 채울 때 (배포 형태) |
| `alt` | $\mathrm{NRMSE}$ ($p_A$) ↔ $\varphi$ ($p_Q$) 교대 | 필요 | $p_A$ (sup), $p_Q$ (full EM) | ○ | 실패 (`ml` 수준) | 채택 불가 — 라벨 정보가 $Q, R$ 에 못 들어감 |
| `alts` | 위와 같되 EM 은 $q_a, q_z$ 만 ($R, q_p, q_g$ 는 sup 값 고정) | 필요 | $p_A$ (sup), $q_a, q_z$ (EM) | △ (NIS 0.6) | `joint` −0.02 | 플랜트는 한 번 식별하고 잡음만 현장 갱신해야 할 때 (§5.8-14) |
| `joint2` | $\varphi + \mu_p\,\mathrm{NRMSE}_{pitch} + \mu_b\,\mathrm{NRMSE}_{bounce}$ | 필요 (2종) | $p_A, p_Q$ | ○ | `joint` 와 같음 | bounce 는 0.30 에 그침 — 라벨 정의 불일치 (§5.8-15), 채택 안 함 |
| `res` (Abbeel Res) | 라벨 제곱오차 | 필요 | $p_Q$ (플랜트 고정) | ✗ (NIS 0.36, 센서 우도 악화) | `joint` 와 같음 | 논문 재현 baseline (§5.8-16) |
| `pred` (Abbeel Pred) | 라벨 예측 로그우도 (필터 분산 포함) | 필요 | $p_Q$ (플랜트 고정) | △ (pitch 분산은 보정, 센서 우도 악화) | `joint` 와 같음 | 필터의 pitch 공분산을 써야 할 때. `joint` 가 같은 보정을 공짜로 줌 |

세 척도가 서로 다른 것을 잰다는 점이 핵심이다. pitch rate corr 은 "목표 신호를 얼마나 복원했나", $\varphi$ 는 "센서를 얼마나 잘 예측했나", NIS 와 innovation 자기상관은 "필터가 자기 불확실성을 정직하게 말하나" 다. `sup` 은 첫째만, `ml` 과 EM 은 둘째·셋째만, `joint` 는 셋을 같이 잡는다. 그림: `outputs/pitch_staged_compare_<model>.png` (첫째), `outputs/pitch_staged_diagnostics_<model>.png` (둘째·셋째).

## 6. 실제 데이터 기반 1-DOF KF + LSTM

Hybrid baseline은 train objective가 1-DOF 후보 중 가장 낮은 Matern 3/2이다. Test 성능을 보고 baseline을 선택하지 않았다.

$$x_k = [z_k, v_k, d_k, \dot d_k]^{\mathsf T}$$

KF의 one-step prediction과 innovation은 다음과 같다.

$$\hat x_{k|k-1} = A\,\hat x_{k-1|k-1}$$

$$\nu_k = a_{z,k} - H\,\hat x_{k|k-1}$$

Causal LSTM 입력은 현재와 과거의 다음 다섯 값이다.

$$u_k = [\hat z_k, \hat v_k, \hat d_k, \hat{\dot d}_k, \nu_k]$$

KF의 calibrated Bounce와 학습 target은 다음과 같다.

$$\hat b^{KF}_k = g\,\hat v_k + c$$

$$e_k = b_k - \hat b^{KF}_k$$

2-layer causal LSTM은 $e_k$만 출력한다.

$$\hat b^{hybrid}_k = \hat b^{KF}_k + \mathrm{LSTM}(u_{0:k})$$

이 구조에는 합성 data가 없다. LSTM 입력도 모두 `IMU_VerAccelVal`에서 online KF로 계산하며 wheel speed나 다른 sensor를 추가하지 않았다. Bi-directional layer, future window, centered convolution도 없으므로 sample 기준 online causal method이다.

Data 분리는 다음과 같다.

- KF parameter fitting, affine calibration, LSTM fitting: 실제 train 2,429 episode
- Early stopping: 실제 validation driver `조현석` 300 episode
- 최종 평가: 기존 driver-held-out test 162 episode

Validation driver는 non-test driver 중 전체 train의 10%에 가장 가까운 episode 수를 가진 driver로 결정한다. KF parameter도 validation을 제외하고 다시 fitting하므로 validation Bounce label이 KF feature 생성에 사용되지 않는다.

이 방법은 full state를 LSTM으로 복원하는 기존 2-DOF LSTM-KF가 아니다. KF가 physics feature를 만들고 LSTM이 최종 Bounce residual을 보정하는 현재 데이터용 hybrid이다. 따라서 LSTM 개선이 $z$, $d$, $\dot d$의 정확도 개선을 뜻하지는 않는다.

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
- Pitch 모델은 자유 calibration gain 아래에서 $\theta$ 의 스케일·부호가 비식별이었다 (§5.6). gain 고정으로 해소했고 파라미터가 물리 범위로 들어왔다. 2단계(`pitch2`) 결과는 아직 자유 gain 기준이라 무효.
- Pitch 의 affine calibration 은 이제 $g = 180/\pi$ 고정, offset 만 train 에서 추정한다. Bounce 는 여전히 $g$ 자유 (라벨 단위 m/s 와 KF $v$ 의 스케일 관계가 확인되지 않음: gain 8–15).
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
- `outputs/pitch_staged_metrics.csv`, `outputs/pitch_staged_models_*.png`: 단계적 축소 모델 (§5.8) metric·파라미터·파형
- `outputs/em_noise_covariance_metrics.csv`, `_history.json`, `_convergence.png`: EM 결과와 우도 수렴 이력 (§5.8-8)
- `outputs/pitch_staged_compare_<model>.png`, `outputs/pitch_staged_diagnostics_<model>.png`: sup/ml/joint/EM 파형 비교, innovation 자기상관·NIS·±2σ 대역 (§5.9)
