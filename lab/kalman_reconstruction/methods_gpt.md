# Pitch-rate 추정을 위한 Kalman Filter 차량 모델 정리

모든 모델은 최종적으로

$$
x_{k+1}=A_dx_k+B_du_k+E_dd_k+G_dw_k
$$

$$
y_k=C_dx_k+D_du_k+F_dd_k+v_k
$$

형태로 이산화하여 Kalman filter에 사용한다.

- $q=\dot\theta$: 추정 대상 pitch rate
- $\bar v_w$: wheel speed로부터 계산한 longitudinal vehicle speed
- $u$: known input, 예: front/rear motor torque
- $d$: unknown disturbance, 예: road input
- $w_k$, $v_k$: process / measurement noise

| 모델 | State $x$ | 입력 $u$ | 측정 $y$ | Observability / Detectability |
|---|---|---|---|---|
| **1. Reduced longitudinal-pitch** | $[v_x,a_x,\theta,q]^T$ | $[T_f,T_r]^T$ 선택적 | Wheel speed, Long. IMU | **Fully observable / detectable** |
| **2. Reduced bounce-pitch** | $[v_x,a_x,z_s,\dot z_s,\theta,q]^T$ | $[T_f,T_r]^T$ 선택적 | Wheel speed, Long./Vertical IMU | **Fully observable / detectable** |
| **3. Sensor-chain bounce-pitch (9-state, 구현·검증 완료)** | $[v_x,a_x,\theta,q,\gamma_g,a_I,z_s,\dot z_s,a_{Iz}]^T$ | $[T_f,T_r]^T$ | Wheel speed **평균+앞뒤 차이**, Long./Vertical IMU (**1차 지연·장착 레버암 모델링**) | Fully observable ($\theta$ DC ↔ 구배만 marginal, rate 무관). **실측 dev-test corr 0.936, free gain 57.1 ≈ 180/π** |
| **4. 2-DOF physical half-car** | $[v_x,a_x,z_s,\dot z_s,\theta,q]^T$ | $[T_f,T_r]^T$ | Wheel speed, Long./Vertical IMU | Road known: parameter별 rank 확인. Road unknown: unknown-input observability 문제 |
| **5. 4-DOF half-car** | $[v_x,a_x,z_s,\dot z_s,\theta,q,z_{uf},\dot z_{uf},z_{ur},\dot z_{ur}]^T$ | $[T_f,T_r]^T$ | Wheel speed, Long./Vertical IMU | Full observability는 parameter-dependent. Detectability 별도 확인 필요 |

---

# 모델 간 관계

$$
[v_x,a_x,\theta,q]
\;\Rightarrow\;
[v_x,a_x,z_s,\dot z_s,\theta,q]
\;\Rightarrow\;
\underbrace{\text{9-state sensor-chain}}_{\text{권장, corr 0.936}}
\;\Rightarrow\;
\text{2-DOF half-car}
\;\Rightarrow\;
\text{4-DOF half-car}
$$

9-state 이후 단계는 회사 제원으로 $k, c$를 고정할 수 있을 때만 진행한다.

---

# 1. Reduced longitudinal-pitch model

## Diagram

```text
(옆에서 본 그림)                                   진행 방향 →
             θ (pitch, nose-up +)
      ┌────────────────────────┐
      │       sprung body      │
      │       states: θ, q     │
      └────────────────────────┘
                 ↑
                 │  longitudinal acceleration coupling
                 │  b_a a_x
                 │
     wheel speed ─────────→ v_x
                              │
                              └→ a_x

측정:
- wheel speed   → v_x
- IMU LongAccel → a_x + s_x g θ
```

## State / Control input

$$
x=[v_x,\ a_x,\ \theta,\ q]^T,
\qquad
u=[T_f,\ T_r]^T
$$

Torque를 사용하지 않으면 $u$를 생략하고 $a_x$ dynamics를 random walk로 둘 수 있다.

## Dynamics

$$
\dot v_x=a_x
$$

$$
\dot a_x=-\lambda_a a_x+b_fT_f+b_rT_r+w_a
$$

$$
\dot\theta=q
$$

$$
\dot q=-\omega_p^2\theta-2\zeta_p\omega_pq+b_a a_x+w_p
$$

따라서

$$
A_c=
\begin{bmatrix}
0&1&0&0\\
0&-\lambda_a&0&0\\
0&0&0&1\\
0&b_a&-\omega_p^2&-2\zeta_p\omega_p
\end{bmatrix}
$$

이다.

Torque를 사용하지 않는 경우에는 $\lambda_a=b_f=b_r=0$으로 두고

$$
\dot a_x=w_a
$$

로 둔다.

## Measurement

$$
\bar v_w\simeq v_x
$$

$$
a_{x,\mathrm{IMU}}\simeq a_x+s_xg\theta
$$

따라서

$$
y=
\begin{bmatrix}
\bar v_w\\
a_{x,\mathrm{IMU}}
\end{bmatrix}
=
Cx+v
$$

$$
C=
\begin{bmatrix}
1&0&0&0\\
0&1&s_xg&0
\end{bmatrix}
$$

## Observability / Detectability

$$
\operatorname{rank}(\mathcal O)=4
$$

따라서

$$
\boxed{\text{Fully observable}}
\qquad
\boxed{\text{Detectable}}
$$

이다.

---

# 2. Reduced bounce-pitch model

## Diagram

```text
(옆에서 본 그림)                                   진행 방향 →
             z_s ↑   θ (pitch, nose-up +)
      ┌────────────────────────┐
      │       sprung body      │
      │   z_s, ż_s, θ, q       │
      └────────────────────────┘
                 │
                 ├── longitudinal coupling: a_x → q
                 │
     wheel speed ─────────→ v_x
                              │
                              └→ a_x

측정:
- wheel speed   → v_x
- IMU LongAccel → a_x + s_x g θ
- IMU VerAccel  → z̈_s
```

## State / Control input

$$
x=[v_x,\ a_x,\ z_s,\ \dot z_s,\ \theta,\ q]^T,
\qquad
u=[T_f,\ T_r]^T
$$

## Dynamics

$$
\dot v_x=a_x
$$

$$
\dot a_x=-\lambda_a a_x+b_fT_f+b_rT_r+w_a
$$

$$
\ddot z_s=-\omega_z^2z_s-2\zeta_z\omega_z\dot z_s+w_z
$$

$$
\dot\theta=q
$$

$$
\dot q=-\omega_p^2\theta-2\zeta_p\omega_pq+b_a a_x+w_p
$$

따라서

$$
A_c=
\begin{bmatrix}
0&1&0&0&0&0\\
0&-\lambda_a&0&0&0&0\\
0&0&0&1&0&0\\
0&0&-\omega_z^2&-2\zeta_z\omega_z&0&0\\
0&0&0&0&0&1\\
0&b_a&0&0&-\omega_p^2&-2\zeta_p\omega_p
\end{bmatrix}
$$

이다.

Road/bump excitation은 $w_z,w_p$에 포함한다.

## Measurement

$$
y=
\begin{bmatrix}
\bar v_w\\
a_{x,\mathrm{IMU}}\\
\tilde a_{z,\mathrm{IMU}}
\end{bmatrix}
$$

$$
\bar v_w\simeq v_x
$$

$$
a_{x,\mathrm{IMU}}\simeq a_x+s_xg\theta
$$

$$
\tilde a_{z,\mathrm{IMU}}
\simeq
\ddot z_s
=
-\omega_z^2z_s-2\zeta_z\omega_z\dot z_s
$$

따라서

$$
C=
\begin{bmatrix}
1&0&0&0&0&0\\
0&1&0&0&s_xg&0\\
0&0&-\omega_z^2&-2\zeta_z\omega_z&0&0
\end{bmatrix}
$$

## Observability / Detectability

일반적인 $\omega_z\neq0$, $g\neq0$에 대해

$$
\operatorname{rank}(\mathcal O)=6
$$

따라서

$$
\boxed{\text{Fully observable}}
\qquad
\boxed{\text{Detectable}}
$$

이다.

---

# 3. Sensor-chain bounce-pitch model (9-state)

Model 2에 **센서 체인**(IMU 1차 지연, 장착 위치 레버암)과 **휠속 앞뒤 차이 채널**, **모터 토크 입력**을 추가한 모델.
`lab/kalman_reconstruction/pitch_staged_reconstruction.py`의 `d_torque`로 구현·검증되었다.

## Diagram

```text
(옆에서 본 그림)                                   진행 방향 →
             z_s ↑   θ (pitch, nose-up +)
      ┌────────────────────────┐
      │       sprung body      │    v_x, a_x: 종방향 상태
      │   z_s, ż_s, θ, q       │    γ_g: 도로 구배 (random walk)
      └────────────────────────┘
                 ↑ b_a a_x (하중이동)
                 │
  motor torque u=[T_f,T_r] ──► ȧ_x = −λ_a a_x + b_T ΣT

  wheel speed (빠른 채널, 지연 기준점)
     ├─ 평균  v̄_w  = v_x
     └─ 차이  Δv_w = ℓ q + κ a_x + s_f T_f + s_r T_r   ← q의 부호·위상을 핀

  IMU 유닛 ── [1차 지연 τ_I] ── 장착 위치 (h_I 높이, x_I 전후)
     ├─ a_x_IMU = a_x + g θ + g γ_g − h_I q̇    (지연 상태 a_I 로 관측)
     └─ a_z_IMU = z̈_s + x_I q̇                  (지연 상태 a_Iz 로 관측)

측정에 반영된 실측 사실:
- 휠 파생 a_x가 IMU보다 50–60 ms 선행 (yaw 채널 ~30 ms 교차 확인) → τ_I
- Δv_w–q 회귀 기울기 −0.29 m → ℓ 시작값
- Mg2=앞축, Mg1=뒤축 (VCU cmd와 corr 0.99)
```

## State / Control input

$$
x=[v_x,\ a_x,\ \theta,\ q,\ \gamma_g,\ a_I,\ z_s,\ \dot z_s,\ a_{Iz}]^T,
\qquad
u=[T_f,\ T_r]^T
$$

## Dynamics

$$
\dot v_x=a_x
$$

$$
\dot a_x=-\lambda_a a_x+b_T(T_f+T_r)+w_a
$$

$$
\dot\theta=q
$$

$$
\dot q=-\omega_p^2\theta-2\zeta_p\omega_pq+b_a a_x+w_p
$$

$$
\dot\gamma_g=w_g
$$

$$
\ddot z_s=-\omega_z^2z_s-2\zeta_z\omega_z\dot z_s+c_{zp}\theta+w_z
$$

$$
\dot a_I=\frac{1}{\tau_I}\big(a_x+g\theta+g\gamma_g-h_I\dot q-a_I\big)
$$

$$
\dot a_{Iz}=\frac{1}{\tau_I}\big(\ddot z_s+x_I\dot q-a_{Iz}\big)
$$

($\dot q$, $\ddot z_s$는 위 동역학 식으로 치환하여 선형 유지. $g=9.81$ 고정 — 스케일 앵커.)

## Measurement

$$
y=
\begin{bmatrix}
\bar v_w\\
a_{x,\mathrm{IMU}}\\
a_{z,\mathrm{IMU}}\\
\Delta v_w
\end{bmatrix}
=
\begin{bmatrix}
v_x\\
a_I\\
a_{Iz}\\
\ell q+\kappa a_x
\end{bmatrix}
+
\begin{bmatrix}
0&0\\0&0\\0&0\\s_f&s_r
\end{bmatrix}u
+v
$$

## Observability / Detectability

$\theta$ DC ↔ $\gamma_g$ 방향 하나만 marginal(둘 다 상수 오프셋, pitch **rate**에는 무관)이고 나머지는 fully observable.

## 실측 결과

프로토콜: driver-held-out **dev-test 3명 128 episode** (최종 검증용 2명은 봉인해 미조회), 100 Hz, 출력 gain $=180/\pi$ 고정(offset만 train). 목적함수 두 가지 병행 — `sup` = 라벨 NRMSE, `ml` = innovation 우도(**라벨 미사용**). *free gain* = 이득을 자유로 풀어 회귀했을 때의 값으로, 모델이 물리 스케일이면 $180/\pi = 57.3$이 나와야 한다.

요소를 하나씩 추가하며 어느 것이 성능을 만드는지 분리했다 (corr = episode별 correlation의 median):

| 실험 | 추가 요소 (검증한 가설) | sup corr | ml corr | free gain |
|---|---|---:|---:|---:|
| a_naive | Model 1 원형 — 지연·레버암 없음 | 0.462 | 0.06 | 28 |
| a_lag | + IMU 1차 지연 $\tau_I$ (실측 60 ms 위상차가 실패 원인인가) | 0.616 | −0.43 | 46 |
| a_full | + IMU 높이 레버암 $h_I$ ($-h_I\dot q$) | 0.836 | 0.18 | 53 |
| b_full | + heave·$a_z$ 채널, 전후 레버암 $x_I$ | 0.873 | 0.17 | 55 |
| c_wheel | + $\Delta v_w = \ell q + \kappa a_x$ (휠속 앞뒤 차이) | 0.887 | **0.545** | 54 |
| **d_torque** | + 토크 입력·slip — **본 §3 모델** | **0.936** | 0.542 | **57.1** |
| e_fixgeo | $x_I = 0.42$ m 고정 (회사 제원 검증) | 0.900 | 0.535 | 56 |
| f_fixlever | + $\ell = -0.29$ m 고정 (실측 레버) | 0.899 | 0.541 | 56 |
| f + joint (μ=3 / 10) | 목적함수 = 우도 + μ·라벨오차 (방법 A) | **0.921 / 0.923** | — | 57 / 56 |
| f + aug (r_label fit / 1 / 10) | 학습 시 라벨을 관측 채널로 (방법 B) | 0.48 / 0.56 / 0.60 | — | 24–26 |
| 기준 | 기존 4-DOF half-car 최고 (파라미터 14–17개) | 0.896 | 거울상 | 67 |

d_torque 상세: RMSE 1.48 deg/s, signed lag +10 ms.

읽는 법:

- **성능은 차량 물리가 아니라 센서 체인에서 나왔다**: 지연 +0.15, $h_I$ +0.22, $a_z$/$x_I$ +0.04, 토크 +0.05.
- $\Delta v_w$ 채널은 sup에는 +0.01뿐이지만 **라벨-프리 ml을 0.17 → 0.55로** 올린다 — $q$의 부호·위상을 센서만으로 핀하는 유일한 채널.
- 파라미터 검증: $h_I \approx -0.36$ m와 $x_I = +0.42$ m(회사 제원 "전방 <0.5 m" 일치)는 **a_full→b_full 구간에서만** 안정 재현된다. $\Delta v_w$·토크 채널이 붙는 c/d에서는 둘 다 표류한다($x_I \to -1.35$, 부호 반전) — $h_I, x_I, \tau_I, \ell$이 서로 위상 역할을 나눠 갖는 교환 자유도의 증거. $\kappa = -0.075$만 끝까지 회귀 실측과 일치.
- 물리 제약의 바닥: $x_I$ 고정(e) 0.900 → $\ell$까지 고정(f) 0.899 — d_torque의 0.936 중 ~0.036은 비물리 $x_I$가 라벨 위상을 흉내 낸 몫이고, **물리값을 걸어도 half-car 최고(0.896)는 유지**된다. f에서 NIS가 2.3 → 1.21로 처음 정상 범위에 접근. 남은 표류: $f_p$·$b_a$ 상한, $\tau_I$ 하한 — $h_I$(IMU 높이)와 IMU 지연 제원이 다음 열쇠.
- **결합 목적함수(방법 A)가 sup과 ml을 동시에 이긴다**: μ = 3–10에서 corr 0.921–0.923 (sup 0.899보다 높음) **이면서** NIS 1.08–1.09 (sup 1.21, ml 1.03). 우도 항이 정규화 역할을 해 sup 단독이 갇혔던 나쁜 local minimum($f_p$·$b_a$ 상한)을 벗어났고, $f_p = 1.6$ Hz·$h_I = -0.33$ m(b_full의 −0.355 재현)로 파라미터도 내부값. 남은 경계: $r_x, r_z$ 하한(센서를 무잡음 취급, ml 특성), $f_z$ 하한.
- **방법 B(학습 시 라벨 관측 채널)는 실패** (corr 0.48–0.60, free gain ~25): 학습 중 필터가 라벨로 q를 알아버려 센서 채널이 pitch를 추출하도록 학습되지 않음 — 학습/배포 필터 불일치의 전형. 이 형태로는 채택 불가.
- 미해결: 결합 μ의 통계적 의미 부재, $r_x, r_z$ 하한(유색 측정잡음 미모델링), 봉인 hold-out 최종 검증.

---

# 4. 2-DOF physical half-car model

## Diagram

```text
(옆에서 본 그림)                                   진행 방향 →
             z_s ↑   θ (pitch, nose-up +)
      ┌────────────────────────┐
      │          m_s, I_y      │      v_x: 종속도 상태
      └──┬──────────────────┬──┘      a_x: 종가속도 상태
     k_r │c_r         k_f   │c_f
         │                  │
         │                  │
     ~~~~┴~~~ r_r ~~~~~~~~~~┴~~~~ r_f
        ├── l_r ──┤├── l_f ──┤

측정:
- wheel speed   → v_x
- IMU LongAccel → a_x + s_x g θ
- IMU VerAccel  → z̈_s  또는 z̈_s + x_I q̇
```

## State / Control input

$$
x=[v_x,\ a_x,\ z_s,\ \dot z_s,\ \theta,\ q]^T,
\qquad
u=[T_f,\ T_r]^T
$$

Unknown road input은

$$
d=[z_{rf},\ \dot z_{rf},\ z_{rr},\ \dot z_{rr}]^T
$$

로 둔다.

## Dynamics

Front / rear suspension deflection:

$$
\delta_f=z_s+l_f\theta-z_{rf},
\qquad
\delta_r=z_s-l_r\theta-z_{rr}
$$

$$
\dot\delta_f=\dot z_s+l_fq-\dot z_{rf},
\qquad
\dot\delta_r=\dot z_s-l_rq-\dot z_{rr}
$$

Suspension force:

$$
F_f=-k_f\delta_f-c_f\dot\delta_f
$$

$$
F_r=-k_r\delta_r-c_r\dot\delta_r
$$

Longitudinal dynamics:

$$
\dot v_x=a_x
$$

$$
\dot a_x=-\lambda_a a_x+b_fT_f+b_rT_r+w_a
$$

Bounce / pitch dynamics:

$$
m_s\ddot z_s=F_f+F_r
$$

$$
\dot\theta=q
$$

$$
I_y\dot q=l_fF_f-l_rF_r-m_sh\,a_x
$$

따라서

$$
\dot x=A_cx+B_cu+E_cd+G_cw
$$

이다.

## Measurement

$$
y=
\begin{bmatrix}
\bar v_w\\
a_{x,\mathrm{IMU}}\\
\tilde a_{z,\mathrm{IMU}}
\end{bmatrix}
$$

$$
\bar v_w\simeq v_x
$$

$$
a_{x,\mathrm{IMU}}\simeq a_x+s_xg\theta
$$

IMU가 CG로부터 longitudinal 방향으로 $x_I$만큼 떨어져 있으면

$$
\tilde a_{z,\mathrm{IMU}}
\simeq
\ddot z_s+x_I\dot q
$$

이고, CG에 가까우면

$$
\tilde a_{z,\mathrm{IMU}}\simeq\ddot z_s
$$

이다.

Road input이 known일 때는

$$
y=Cx+Du+Fd+v
$$

형태의 선형 measurement model로 정리할 수 있다.

## Observability / Detectability

$C$는 위 measurement 관계를 state-space 식으로 정리하여 구성한다.

### Road known

$$
\operatorname{rank}(\mathcal O)
=
\operatorname{rank}
\begin{bmatrix}
C\\
CA\\
\vdots\\
CA^{5}
\end{bmatrix}
$$

를 실제 차량 parameter에 대해 계산해야 한다.

- **Full observability:** parameter-dependent
- **Detectability:** parameter-dependent

### Road unknown

$$
\dot x=Ax+Bu+Ed
$$

이므로 standard observability가 아니라 **unknown-input observability**를 확인해야 한다.

- **Full state/road observability:** 일반적으로 보장되지 않음
- **Detectability:** suspension parameter와 measurement configuration에 따라 확인 필요

---

# 5. 4-DOF half-car model

## Diagram

```text
(옆에서 본 그림)                                   진행 방향 →
             z_s ↑   θ (pitch, nose-up +)
      ┌────────────────────────┐
      │          m_s, I_y      │      v_x: 종속도 상태
      └──┬──────────────────┬──┘      a_x: 종가속도 상태
     k_r │c_r         k_f   │c_f
       ┌─┴─┐              ┌─┴─┐
       │m_u│  뒤           │m_u│  앞
       └─┬─┘              └─┬─┘
        k_t                 k_t
     ~~~~┴~~~ r_r ~~~~~~~~~~~┴~~~ r_f
        ├── l_r ──┤├── l_f ──┤

unsprung states:
- rear : z_ur, ż_ur
- front: z_uf, ż_uf

측정:
- wheel speed   → v_x
- IMU LongAccel → a_x + s_x g θ
- IMU VerAccel  → z̈_s  또는 z̈_s + x_I q̇
```

## State / Control input

$$
x=
[v_x,\ a_x,\ z_s,\ \dot z_s,\ \theta,\ q,\ z_{uf},\ \dot z_{uf},\ z_{ur},\ \dot z_{ur}]^T
$$

$$
u=[T_f,\ T_r]^T,
\qquad
d=[z_{rf},\ z_{rr}]^T
$$

## Dynamics

Front suspension force:

$$
F_{sf}
=
-k_{sf}(z_s+l_f\theta-z_{uf})
-c_{sf}(\dot z_s+l_fq-\dot z_{uf})
$$

Rear suspension force:

$$
F_{sr}
=
-k_{sr}(z_s-l_r\theta-z_{ur})
-c_{sr}(\dot z_s-l_rq-\dot z_{ur})
$$

Tire force:

$$
F_{tf}=k_{tf}(z_{rf}-z_{uf}),
\qquad
F_{tr}=k_{tr}(z_{rr}-z_{ur})
$$

Longitudinal:

$$
\dot v_x=a_x
$$

$$
\dot a_x=-\lambda_a a_x+b_fT_f+b_rT_r+w_a
$$

Sprung mass:

$$
m_s\ddot z_s=F_{sf}+F_{sr}
$$

$$
\dot\theta=q
$$

$$
I_y\dot q=l_fF_{sf}-l_rF_{sr}-m_sh\,a_x
$$

Unsprung masses:

$$
m_{uf}\ddot z_{uf}=F_{tf}-F_{sf}
$$

$$
m_{ur}\ddot z_{ur}=F_{tr}-F_{sr}
$$

따라서

$$
\dot x=A_cx+B_cu+E_cd+G_cw
$$

이다.

## Measurement

$$
y=
\begin{bmatrix}
\bar v_w\\
a_{x,\mathrm{IMU}}\\
\tilde a_{z,\mathrm{IMU}}
\end{bmatrix}
$$

$$
\bar v_w\simeq v_x
$$

$$
a_{x,\mathrm{IMU}}\simeq a_x+s_xg\theta
$$

$$
\tilde a_{z,\mathrm{IMU}}
\simeq
\ddot z_s+x_I\dot q
$$

Motor torque는 measurement가 아니라 $u=[T_f,T_r]^T$로 사용한다.

Wheel rotational speed는 $z_{uf},z_{ur}$의 vertical velocity를 직접 측정하지 않는다.

## Observability / Detectability

$C$는 위 measurement 관계를 10-state vector에 대해 정리하여 구성한다.

$$
\mathcal O=
\begin{bmatrix}
C\\
CA\\
\vdots\\
CA^9
\end{bmatrix}
$$

- **Full observability:** 실제 vehicle parameter와 sensor location에 따라 rank 확인 필요
- **Detectability:** $A$의 unobservable mode의 안정성을 확인하여 판단
- **Unknown road 포함:** standard observability만으로 충분하지 않으며 unknown-input observability 문제로 바뀜

현재 sensor set으로 road, front/rear unsprung state까지 동시에 독립적으로 복원하는 것은 일반적으로 강한 가정 없이 보장하기 어렵다.

---

# Wheelbase preview를 추가하는 경우

Half-car의 front/rear road input은 동일한 road profile의 시간 지연 형태로 둘 수 있다.

$$
\tau_k=\frac{L}{v_{x,k}}
$$

$$
z_{rr}(t)\simeq z_{rf}(t-\tau)
$$

따라서 $z_{rf},z_{rr}$ 두 unknown input을 하나의 road profile로 줄일 수 있으며, Model 4 또는 Model 5의 unknown-input estimation 문제를 단순화할 수 있다.

