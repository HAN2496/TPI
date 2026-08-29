# Pitch rate 복원 탐색 기록 (2026-08-20, 코드 삭제됨)

`Pitch_rate_6D` 를 IMU + 휠속 + 모터 토크만으로 복원하기 위해 시도한 내용의 기록. 관련 스크립트와 결과 파일은 모두 삭제했고, 이 문서만 남긴다.

## 1. 구현했던 것

**(a) 신호 탐색** — 조현석 60–120 ep 로 pitch rate 와 각 채널의 상관·coherence·전달함수를 확인.
**(b) 채널별 선형 정보량** — train driver 로 FIR(비인과 ±0.3 s / 인과 0.6 s)을 최소제곱 fit, held-out driver 162 ep 상관 측정.
**(c) 프로토타입 KF** — 아래 §3 모델을 기존 `KalmanFilter` 틀에 넣어 Powell 로 fit.

## 2. 데이터에서 관찰한 것 (논문 아님, 이 데이터의 경험적 사실)

- Pitch rate 파워 91%가 2 Hz 아래, PSD 피크 ≈ 1 Hz. 부호는 nose-up 양수 (가속 시작 +4.7, 제동 시작 −6.2 deg/s).
- 휠속 미분 $\dot v_w$ 가 IMU $a_x$ 보다 60–80 ms 선행, 토크 과도 시 휠 가속 스파이크가 차체의 2–3배. → 중력누설 $a_x - \dot v_w$ 는 pitch 각 proxy 로 나쁨. $\dot v_w$ 를 60 ms 지연만 시켜도 physics 상관 0.43 → 0.70.
- $a_x$ 회귀에 $\theta,\ \ddot\theta$ 항을 넣으면 R² 0.29 → 0.81, 레버암 $|h| \approx 0.15\text{–}0.2$ m. pitch 모드(≈2 Hz)에서 $h\omega^2 \gg g$ 이므로 레버암 항이 중력누설보다 큼.
- 저주파(< 0.8 Hz)에서 $\dot\theta \approx -a_z / v$ (기울기 −0.9~−1.0, 상관 −0.8): 차체가 노면 기울기를 따라감.
- $v_{wf} - v_{wr}$ 가 $\dot\theta$ 와 2–5 Hz 상관 −0.62, 회귀 기울기 −0.2~−0.3 m.
- 인과 0.6 s FIR held-out 상관: vmean 0.86 / +토크 0.91 / +$v_f{-}v_r$ 0.93 / +$a_x, a_z$ **0.96**. roll·yaw·$a_y$ 는 무기여. (선형 인과 방법의 상한 근사)

## 3. 프로토타입 KF 수식

상태 $x = [\theta,\ \dot\theta,\ a_b,\ d,\ b_x]$, 입력 $u = [\tau,\ \Delta\dot v_{fr}]$, 관측 $y = [a_x^{IMU},\ \dot v_w]$.

$$\ddot\theta = -\omega_\theta^2\theta - 2\zeta_\theta\omega_\theta\dot\theta + k_a a_b + k_\tau \tau + k_{dv}\Delta\dot v_{fr} + w_\theta$$

$$\dot a_b = -\lambda_a\,(a_b - g_\tau\tau - d) + w_a, \qquad \dot d = w_d, \qquad \dot b_x = w_b$$

$$a_x^{IMU} = a_b + g\theta + h\,\ddot\theta + b_x + v_x, \qquad \dot v_w = a_b + \kappa_w\,(g_\tau\tau - a_b) + v_w$$

held-out pitch 상관 (train 200 ep, Powell 3000 fev 동일 예산): 기존 reduced_kf 0.744 / 위 모델에서 잠재 $a_b$ 끈 ablation 0.816 / 켠 것 0.842. 최적화가 $k_a, k_{dv}$ 경계에서 멈춰 미완성이었고, 인과 FIR 상한 0.96과 차이가 컸다.

## 4. 근거 구분

**논문·교과서 근거가 있는 부분**
- pitch 2차 진동자 (decoupled pitch mode, $\omega_\theta^2 = (k_f a^2 + k_r b^2)/I_y$): Gillespie, *Fundamentals of Vehicle Dynamics* ch. 5; Rajamani, *Vehicle Dynamics and Control* ch. 11–12.
- 하중이동 모멘트 $\propto$ 종가속, anti-squat/anti-dive 로 토크가 직접 pitch 모멘트를 만듦: Rajamani ch. 4; Gillespie ch. 7.
- 관측 $a_x = a_b + g\theta$ (가속도계 중력누설로 기울기 읽기) 와 구배를 random walk 로 두는 것: 차량 pitch/구배 추정 문헌 (Tseng·Xu·Hrovat 2007 VSD; Lingman & Schmidtbauer 2002 VSD; Sebsadji·Glaser·Mammar 2008 ACC).
- 가속도계 레버암 항 $h\ddot\theta$: strapdown INS 표준 (Titterton & Weston; Groves).
- 미지 외력을 RW/OU/Matérn latent 상태로: Nayek et al. 2019 MSSP; Branlard et al. 2020 (기존 bounce 1-DOF 와 동일 계보).
- 뒷바퀴 노면 = 앞바퀴의 $L/v$ 지연 (wheelbase filtering; 미구현 확장에서 언급): Gillespie; Doumiati et al. 2011 ACC; Agebjar et al. 2025 FUSION.

**내가 추가한 가정 (직접적인 논문 근거 없음)**
- $\dot a_b = -\lambda_a(a_b - g_\tau\tau - d)$: "차체 가속이 토크에 1차 지연으로 따라간다"는 형태 자체는 타이어 relaxation length (Pacejka) 와 휠 회전 동역학에서 *동기만* 얻은 임의 축약. 문헌은 휠 동역학 $J\dot\omega = i\tau - RF_x$ 를 명시적으로 두지, 이런 1차 근사를 쓰지 않는다.
- 휠속 관측의 과응답 항 $\kappa_w(g_\tau\tau - a_b)$: 전적으로 임의. "반력이 전달되기 전 휠이 먼저 가속한다"를 한 파라미터로 뭉갠 것.
- $v_{wf} - v_{wr} = c_\theta\dot\theta + \dots$ 관측 (제안만, 미구현): 서스펜션 측면 기구학·하중 의존 유효반경으로 설명했지만 특정 논문을 확인하지 못함. Toyota 휠속 기반 pitch/bounce 제진을 근거로 들었으나 정확한 출처 불확실.
- $\Delta\dot v_{fr}$ 를 pitch 모멘트 입력으로 (기존 reduced_kf 에서 승계): 경험적으로 효과가 있었을 뿐 물리 유도 없음.
- fit 을 상관/대역 MSE 로 하자는 것, 파라미터를 정적 이득 $G_a = k_a/\omega_\theta^2$ 로 재매개화하자는 것: 식별 실무 판단이지 문헌 인용 아님.

## 5. 남은 결론

1-DOF 급 기본형부터 다시 가는 것이 맞다: 상태 $[\theta, \dot\theta, d_\theta(, \gamma)]$, $\ddot\theta = -\omega_\theta^2\theta - 2\zeta_\theta\omega_\theta\dot\theta + d_\theta$, 관측 $a_x - \dot v_w = g(\theta + \gamma)$, $d_\theta$ 는 RW/OU/Matérn 비교 — 기존 bounce 1-DOF 와 같은 계보(Nayek/Branlard)이며 새 가정이 들어가지 않는다. bounce 와 달리 관측이 같은 자유도의 가속도가 아니라 중력누설이라 성능 기대치는 낮게 잡아야 한다.
