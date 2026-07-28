# 6D 신호 재구성

배포 가용 채널 $x(t)$ (IMU 5ch, 휠속 4ch, 토크 4ch)로부터 6D 신호 $y(t) = [\dot z,\ \dot\phi,\ \dot\theta]$ (bounce/roll/pitch rate)를 추정한다. 학습 모델은 물리 근사 $\phi(x)$를 앵커로 한 잔차 구조를 공유한다:

$$\hat y = \phi(x) + f([\,x;\ \phi(x)\,])$$

## 1. Physics

채널별 독립 신호 체인. 학습 파라미터 없음.

**Roll** — 강체의 각속도는 측정 위치와 무관하므로 자이로 통과:

$$\hat{\dot\phi} = \omega_{roll}^{IMU}$$

**Bounce** — 수직 가속도 적분 후 드리프트 제거 (2차 Butterworth, 영위상):

$$\hat{\dot z} = \mathrm{HP}_{0.3\,\mathrm{Hz}}\!\left[\int_0^t \big(a_z - \bar a_z\big)\, d\tau\right]$$

**Pitch** — 종축 비력의 중력 누설 $a_x = \dot v + g\sin\theta$ 에서 각도를 복원, 미분:

$$\hat\theta = \frac{a_x - \dot{\bar v}_{whl}}{g}, \qquad \hat{\dot\theta} = -\,\mathrm{LP}_{2\,\mathrm{Hz}}\!\left[\frac{d\hat\theta}{dt}\right]$$

한계: bounce는 HP 컷오프 아래 저주파 손실, pitch는 노면 경사와 차체 pitch를 원리적으로 구분 불가.

## 2. Kalman (half-car gray-box)

상태 $s = [z,\ \dot z,\ \theta,\ \dot\theta,\ \eta,\ b,\ \gamma]$ — heave 변위/속도, pitch 각/각속도, 유색 노면 여기, $a_z$ 바이어스, 노면 경사.

**동역학** — 감쇠 진동자 2개(heave, pitch) + random walk 3개($\eta$, $b$, $\gamma$):

$$\ddot z = -\omega_z^2 z - 2\zeta_z\omega_z \dot z + \eta + \kappa\, \dot{\bar v}_{whl}$$

$$\ddot\theta = -\omega_\theta^2 \theta - 2\zeta_\theta\omega_\theta \dot\theta + g_1 \dot{\bar v}_{whl} + g_2 \tau_{mot} + g_3\, \Delta\dot v_{f\!-\!r} + c_{z\theta}\, \dot z + w_\theta$$

$$\dot\eta = w_\eta$$

노면 여기 $\eta$를 백색 노이즈가 아닌 random walk 상태로 두는 것은 ISO 8608 노면 스펙트럼(변위 PSD $\propto f^{-2}$)에 근거하며, 식별에서도 유색 경로가 지배한다($q_z \to 0$). $\kappa$는 종방향 힘의 수직 성분(anti-lift), $c_{z\theta}$는 앞뒤 서스펜션 비대칭의 pitch–heave 커플링, $\Delta\dot v_{f\!-\!r}$는 앞뒤 휠속 미분 차(노면 pitch 이벤트의 직접 신호).

pitch에도 대칭으로 유색 여기 $\eta_\theta$를 두는 확장과 초기 시변 게인 워밍업을 시험했으나 기본 설정에서는 모두 비활성이다: 전자는 식별이 $q_{\eta\theta} \to 0$으로 기각했고(노면 pitch는 $\Delta\dot v_{f\!-\!r}$로 이미 설명됨), 후자는 초기 스파이크가 peak 통계를 오염시켰다.

**측정** — 레버암 $\ell$ 만큼 pitch가 섞인 수직 가속도, 그리고 중력 누설:

$$a_z^{IMU} = \ddot z + \ell\,\ddot\theta + b + v_1$$

$$a_x^{IMU} - \dot{\bar v}_{whl} = g(\theta + \gamma) + v_2$$

경사 $\gamma$(느린 random walk)와 차체 pitch $\theta$(고유진동수 $\omega_\theta$의 진동자)가 **동역학 사전지식으로 분리**되는 것이 physics 대비 핵심 이득.

**추정** — Euler 이산화 후 표준 칼만 재귀 (정상상태 게인 $K$는 Riccati 수렴해):

$$s_{n|n-1} = A\,s_{n-1} + B\,u_n, \qquad s_n = s_{n|n-1} + K\big(y_n - H s_{n|n-1} - D u_n\big)$$

$$P \leftarrow APA^{\!\top} + Q, \quad K = PH^{\!\top}(HPH^{\!\top}+R)^{-1}, \quad P \leftarrow (I-KH)P$$

파라미터 $\psi = \{\omega_z, \zeta_z, \omega_\theta, \zeta_\theta, \ell, g_1, g_2, \kappa, c_{z\theta}, g_3, Q, R\}$ 는 train driver 에피소드에서 타깃과의 상관 최대화로 식별한다 (gray-box). 출력은 $\hat{\dot z} = s_2$, $\hat{\dot\theta} = -s_4$, roll은 자이로 통과.

**행렬 표현** — 기본 모드(워밍업·$\eta_\theta$ 비활성), $s = [z,\ \dot z,\ \theta,\ \dot\theta,\ \eta,\ b,\ \gamma]^\top$, $u = [\dot{\bar v}_{whl},\ \tau_{mot},\ \Delta\dot v_{f\!-\!r}]^\top$, $y = [a_z^{IMU},\ a_x^{IMU} - \dot{\bar v}_{whl}]^\top$, $\Delta t = 1/f_s$:

$$
A = \begin{bmatrix}
1 & \Delta t & 0 & 0 & 0 & 0 & 0 \\
-\omega_z^2 \Delta t & 1 - 2\zeta_z\omega_z\Delta t & 0 & 0 & \Delta t & 0 & 0 \\
0 & 0 & 1 & \Delta t & 0 & 0 & 0 \\
0 & c_{z\theta}\Delta t & -\omega_\theta^2\Delta t & 1 - 2\zeta_\theta\omega_\theta\Delta t & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
\qquad
B = \begin{bmatrix}
0 & 0 & 0 \\
\kappa\Delta t & 0 & 0 \\
0 & 0 & 0 \\
g_1\Delta t & g_2\Delta t & g_3\Delta t \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
H = \begin{bmatrix}
-\omega_z^2 & -2\zeta_z\omega_z + \ell c_{z\theta} & -\ell\omega_\theta^2 & -2\ell\zeta_\theta\omega_\theta & 1 & 1 & 0 \\
0 & 0 & g & 0 & 0 & 0 & g
\end{bmatrix}
\qquad
D = \begin{bmatrix}
\kappa + \ell g_1 & \ell g_2 & \ell g_3 \\
0 & 0 & 0
\end{bmatrix}
$$

$$
Q = \mathrm{diag}(0,\ q_z,\ 0,\ q_\theta,\ q_\eta,\ q_b,\ q_\gamma)\,\Delta t
\qquad
R = \mathrm{diag}(r_z,\ r_x)
$$

$H$의 1행은 측정식 $a_z = \ddot z + \ell\ddot\theta + b$에 동역학의 $\ddot z,\ \ddot\theta$를 대입해 상태의 선형결합으로 푼 것이고, 같은 대입에서 입력에 걸리는 항이 $D$의 1행이다. $A$의 random walk 세 상태($\eta, b, \gamma$)는 항등 행, $Q$에서 위치 상태($z, \theta$)는 노이즈 없음.

## 3. FIR (Wiener)

비인과 다채널 선형 필터, 커널 길이 $2L{+}1$ ($L = 50$, ±0.5 s):

$$\hat y_c[n] = \sum_{ch} \sum_{\tau=-L}^{L} h_{c,ch}[\tau]\; x_{ch}[n+\tau]$$

MSE 최소화 해는 Wiener 필터로 수렴하므로, **선형 시불변 매핑의 성능 상한**을 정의한다.

## 4. U-Net

다중 해상도 인코더-디코더 (100→50→25 Hz), skip 연결로 고주파 보존. $\downarrow$ = 평균 풀링 ×2, $\uparrow$ = 최근접 업샘플 ×2, $C_i$ = conv-ReLU 스택:

$$e_1 = C_1(x), \quad e_2 = C_2(\downarrow e_1), \quad e_3 = C_3(\downarrow e_2)$$

$$d_2 = C_4([\,\uparrow e_3;\ e_2\,]), \quad d_1 = C_5([\,\uparrow d_2;\ e_1\,]), \quad f = W d_1$$

잔차 목적함수 $\mathbb{E}\,\|\phi + f - y\|^2$ 를 최소화. 깊은 경로가 ~1 s 스케일의 느린 동역학을, skip이 피크 형상을 담당한다.

## 결과 (held-out driver 5명, 파형 상관 중앙값)

| | Bounce | Roll | Pitch |
|---|---|---|---|
| physics | 0.739 | 0.932 | 0.435 |
| kalman | 0.965 | 0.932 | 0.886 |
| fir | 0.971 | 0.978 | 0.881 |
| unet | **0.991** | **0.985** | **0.993** |

kalman은 인과·해석가능 방법 중 최고로, bounce는 유색 노면 상태 $\eta$ (백색 버전 0.53 → 0.96), pitch는 앞뒤 휠속차 입력 $\Delta\dot v_{f\!-\!r}$ (0.85 → 0.88, 비인과 FIR과 동급)이 결정적이었다. 식별 파라미터 일부는 필터 정형화 역할을 겸하므로(heave 모드 주파수 등) 서스펜션 물리량으로 직해석해서는 안 된다. 실험: `lab/reconstruction/`.
