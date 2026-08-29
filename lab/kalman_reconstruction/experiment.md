# Bounce-rate reconstruction experiments

## 1. Scope

이 문서는 다음 네 종류의 실험만 정리한다.

1. Ornstein-Uhlenbeck Kalman filter (KF)
2. Ornstein-Uhlenbeck KF + LSTM
3. Online model-free: LSTM, GRU, Transformer
4. Offline model-free: Bi-LSTM, Offline Transformer

모든 모델의 최종 목표는 사용 가능한 센서만으로 `Bounce_rate_6D` 시계열을 복원하는 것이다. 실험에는 합성 데이터가 아니라 프로젝트의 실제 주행 데이터만 사용했다. `Bounce_rate_6D`는 학습 및 평가 정답으로만 사용하며, 추론 시 입력에는 포함하지 않는다.

OU KF는 물리적인 quarter-car suspension model이 아니다. Bounce 운동을 1-DOF oscillator로 나타내고, 관측되지 않는 외란 가속도를 평균회귀하는 Ornstein-Uhlenbeck 상태로 모델링한 확장형이다.

---

## 2. Common experimental protocol

### 2.1 Data split

| Split | Drivers | Episodes |
|---|---|---:|
| Training | 조현석과 held-out test 운전자를 제외한 나머지 운전자 | 2,429 |
| Validation | 조현석 | 300 |
| Test | 김재호, 김진명, 김태근, 신민철, 이강근 | 162 |

- Sampling frequency: 100 Hz
- Episode length: 1,000 samples, or 10 s
- Short episodes: edge padding
- Long episodes: first 1,000 samples retained
- Random seed: 42

Validation 운전자는 모델 선택과 early stopping에만 사용했다. 다섯 명의 test 운전자는 정규화, KF 파라미터 추정, 학습, early stopping에 전혀 사용하지 않았다.

### 2.2 Input channels

Model-free 모델에 사용한 입력 채널은 다음 13개이다.

| Group | Channels |
|---|---|
| IMU | `IMU_RollRtVal`, `IMU_VerAccelVal`, `IMU_YawRtVal`, `IMU_LatAccelVal`, `IMU_LongAccelVal` |
| Wheel speed | `WHL_SpdFLVal`, `WHL_SpdFRVal`, `WHL_SpdRLVal`, `WHL_SpdRRVal` |
| Estimated motor torque | `MCU_Mg1EstTqVal`, `MCU_Mg2EstTqVal` |
| Commanded motor torque | `VCU_MotTqCmdFrntVal`, `VCU_MotTqCmdRearVal` |

OU KF는 이 중 `IMU_VerAccelVal` 하나만 관측한다. Hybrid LSTM도 원본 13개 채널을 직접 사용하지 않고, 같은 vertical-acceleration observation으로 KF가 생성한 state와 innovation만 입력받는다. Target은 모든 모델에서 `Bounce_rate_6D`이다.

### 2.3 Evaluation metrics

각 test episode에서 correlation, RMSE, temporal lag를 계산한 뒤 162개 episode의 10th percentile, median, 90th percentile을 보고한다.

Correlation은 다음과 같다.

$$
\rho = \frac{\sum_{t=1}^{T}(b_t-\bar b)(\hat b_t-\bar{\hat b})}{\sqrt{\sum_{t=1}^{T}(b_t-\bar b)^2}\sqrt{\sum_{t=1}^{T}(\hat b_t-\bar{\hat b})^2}}
$$

RMSE는 다음과 같다.

$$
\operatorname{RMSE}=\sqrt{\frac{1}{T}\sum_{t=1}^{T}(b_t-\hat b_t)^2}
$$

여기서 기호의 의미는 다음과 같다.

| Symbol | Meaning |
|---|---|
| $T$ | episode의 sample 수, 본 실험에서는 1,000 |
| $b_t$ | 시각 $t$의 실제 `Bounce_rate_6D` |
| $\hat b_t$ | 시각 $t$의 복원값 |
| $\bar b$ | 실제값의 episode 평균 |
| $\bar{\hat b}$ | 복원값의 episode 평균 |

Lag는 제한된 시차 범위(±0.5 s)에서 실제값과 복원값의 cross-correlation이 최대가 되는 시차의 절댓값으로 계산하며, 100 Hz 기준 sample 단위 lag를 ms로 환산한다. Correlation은 클수록, RMSE와 lag의 절댓값은 작을수록 좋다. Lag는 진단용으로 보고만 하며 어떤 후보정에도 쓰지 않는다. `Bounce_rate_6D` 자체가 IMU와 위상 정렬이 안 된 파생 신호이므로 이 lag는 추정기 고유 지연이 아니라 target과의 상대 위상이다 (`methods.md` §10).

---

## 3. Ornstein-Uhlenbeck Kalman filter

### 3.1 Purpose and observation

KF가 직접 관측하는 값은 IMU의 vertical acceleration이다. 원 데이터가 중력가속도 단위로 기록되었다고 보고 다음과 같이 변환한다.

$$
y_k=a_{z,k}=9.81\left(\texttt{IMU\_VerAccelVal}_k-1\right)
$$

`Bounce_rate_6D`는 KF의 measurement가 아니다. KF가 vertical acceleration으로부터 bounce velocity를 추정한 뒤, training label로 affine calibration하여 최종 bounce-rate 복원값을 만든다.

### 3.2 State definition

연속시간 상태는 다음과 같다.

$$
\mathbf{x}(t)=\begin{bmatrix}z(t)&v(t)&d(t)\end{bmatrix}^{\mathsf T}
$$

| Symbol | Meaning |
|---|---|
| $z(t)$ | 차체의 latent vertical displacement |
| $v(t)=\dot z(t)$ | 차체의 latent vertical velocity이며 bounce-rate의 직접적인 기반값 |
| $d(t)$ | 관측되지 않는 disturbance acceleration |
| $f$ | oscillator natural frequency in Hz |
| $\omega=2\pi f$ | oscillator natural angular frequency |
| $\zeta$ | damping ratio |
| $q_v$ | velocity equation에 들어가는 white process-noise intensity |
| $\lambda$ | disturbance가 0으로 회귀하는 속도 |
| $q_f$ | OU disturbance의 stationary variance $\sigma_d^2$. 코드의 white-noise intensity는 $2 q_f \lambda$ |
| $r$ | accelerometer measurement-noise variance |

### 3.3 Continuous-time dynamics

Bounce oscillator는 다음과 같이 둔다.

$$
\dot z(t)=v(t)
$$

$$
\dot v(t)=-\omega^2z(t)-2\zeta\omega v(t)+d(t)+w_v(t)
$$

OU disturbance는 다음 1차 stochastic differential equation으로 나타낸다.

$$
\dot d(t)=-\lambda d(t)+w_d(t)
$$

그 결과 전체 연속시간 모델은 다음과 같다.

$$
\dot{\mathbf{x}}(t)=\mathbf{F}\mathbf{x}(t)+\mathbf{w}(t)
$$

$$
\mathbf{F}=\begin{bmatrix}
0&1&0\\
-\omega^2&-2\zeta\omega&1\\
0&0&-\lambda
\end{bmatrix}
$$

구현에 사용한 continuous process-noise spectral-density matrix는 다음과 같다.

$$
\mathbf{Q}_c=\operatorname{diag}\left(10^{-10},q_v,2q_f\lambda\right)
$$

$2 q_f \lambda$는 OU 상태의 stationary variance가 $q_f$가 되도록 정한 intensity이다. OU에서는 disturbance dynamics로부터 stationary covariance를 계산해 초기 covariance에 사용한다. $10^{-10}$은 displacement state의 수치적 안정성을 위한 매우 작은 process noise이다.

### 3.4 Measurement equation

Vertical acceleration은 oscillator 식의 오른쪽 항이므로 measurement matrix는 $\mathbf{F}$의 두 번째 행과 같다.

$$
y_k=\mathbf{H}\mathbf{x}_k+\epsilon_k
$$

$$
\mathbf{H}=\begin{bmatrix}-\omega^2&-2\zeta\omega&1\end{bmatrix}
$$

$$
\epsilon_k\sim\mathcal{N}(0,r)
$$

따라서 센서가 측정한 acceleration 하나로 $z$, $v$, $d$를 동시에 확률적으로 추정한다.

### 3.5 Continuous-to-discrete conversion

Sampling interval은 $\Delta t=0.01$ s이다. 단순 Euler approximation 대신 matrix exponential과 Van Loan 방법으로 discrete transition matrix $\mathbf{A}$와 process covariance $\mathbf{Q}$를 계산한다.

$$
\mathbf{M}=\begin{bmatrix}\mathbf{F}&\mathbf{Q}_c\\\mathbf{0}&-\mathbf{F}^{\mathsf T}\end{bmatrix}\Delta t
$$

$$
\exp(\mathbf{M})=\begin{bmatrix}\mathbf{E}_{11}&\mathbf{E}_{12}\\\mathbf{0}&\mathbf{E}_{22}\end{bmatrix}
$$

$$
\mathbf{A}=\mathbf{E}_{11},\qquad \mathbf{Q}=\mathbf{E}_{12}\mathbf{A}^{\mathsf T}
$$

Discrete state-space model은 다음과 같다.

$$
\mathbf{x}_k=\mathbf{A}\mathbf{x}_{k-1}+\mathbf{q}_{k-1},\qquad \mathbf{q}_{k-1}\sim\mathcal{N}(\mathbf{0},\mathbf{Q})
$$

### 3.6 Kalman-filter recursion

각 episode 시작 시 state mean은 다음과 같이 0으로 초기화한다.

$$
\hat{\mathbf{x}}_{0|0}=\mathbf{0}
$$

초기 covariance는 기본적으로 identity matrix를 사용하며, OU disturbance state $d$의 stationary variance를 사용한다.

Prediction step은 다음과 같다.

$$
\hat{\mathbf{x}}_{k|k-1}=\mathbf{A}\hat{\mathbf{x}}_{k-1|k-1}
$$

$$
\mathbf{P}_{k|k-1}=\mathbf{A}\mathbf{P}_{k-1|k-1}\mathbf{A}^{\mathsf T}+\mathbf{Q}
$$

Innovation과 innovation covariance는 다음과 같다.

$$
\nu_k=y_k-\mathbf{H}\hat{\mathbf{x}}_{k|k-1}
$$

$$
\mathbf{S}_k=\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^{\mathsf T}+r
$$

Kalman gain과 state update는 다음과 같다.

$$
\mathbf{K}_k=\mathbf{P}_{k|k-1}\mathbf{H}^{\mathsf T}\mathbf{S}_k^{-1}
$$

$$
\hat{\mathbf{x}}_{k|k}=\hat{\mathbf{x}}_{k|k-1}+\mathbf{K}_k\nu_k
$$

Covariance update에는 수치적으로 더 안정적인 Joseph form을 사용한다.

$$
\mathbf{P}_{k|k}=(\mathbf{I}-\mathbf{K}_k\mathbf{H})\mathbf{P}_{k|k-1}(\mathbf{I}-\mathbf{K}_k\mathbf{H})^{\mathsf T}+\mathbf{K}_kr\mathbf{K}_k^{\mathsf T}
$$

현재 시각 $k$의 measurement만 사용하므로 이 KF는 online causal estimator이다.

### 3.7 Bounce-rate calibration

KF가 얻은 velocity state $\hat v_k$는 `Bounce_rate_6D`와 좌표계, scale, offset이 정확히 같다고 보장할 수 없다. 따라서 training split에서 다음 affine calibration을 함께 추정한다.

$$
\hat b_k=g\hat v_k+c
$$

여기서 $g$는 gain, $c$는 offset이다. 두 값은 KF가 생성한 training velocity와 training `Bounce_rate_6D` 사이의 least-squares fit으로 계산한다. Test label을 이용한 후처리는 하지 않는다.

### 3.8 Parameter fitting

추정 대상은 다음 여섯 개이다.

$$
\boldsymbol{\theta}=\begin{bmatrix}f&\zeta&\log q_v&\log q_f&\log r&\log\lambda\end{bmatrix}^{\mathsf T}
$$

계산량을 제한하기 위해 training split에서 시간 순서대로 균등하게 고른 최대 300개 실제 episode를 사용했다. 각 후보 파라미터에 대해 KF를 실행하고, training label로 $g$와 $c$를 fit한 뒤 normalized RMSE를 최소화했다. Optimizer는 Powell method이며 최대 iteration은 40이다.

| Parameter | Search range |
|---|---:|
| $f$ | 0.5 to 8.0 Hz |
| $\zeta$ | 0.05 to 5.0 |
| $\log q_v$ | -12 to 3 |
| $\log q_f$ | -12 to 5 |
| $\log r$ | -12 to 4 |
| $\log\lambda$ | $\log 0.05$ to $\log 100$ |

Initial point는 다음과 같다.

$$
\boldsymbol{\theta}_0=\begin{bmatrix}1.3&0.3&-5&0&-3&\log 2\end{bmatrix}^{\mathsf T}
$$

현재 저장된 split에서 얻은 fitted values는 다음과 같다.

| Quantity | Fitted value |
|---|---:|
| $f$ | 4.004278 Hz |
| $\zeta$ | 0.298955 |
| $q_v$ | $1.1066\times10^{-3}$ |
| $q_f$ | 148.413159 |
| $r$ | 0.330639 |
| $\lambda$ | 13.484995 |

$f$는 기존 상한 3 Hz를 넘어 4 Hz 부근으로 이동했지만, 확장한 새 상한에는 도달하지 않았다. 반면 $q_f = 148.41 = e^{5}$는 $\log q_f$의 상한에 정확히 도달한 값이다 (Matern 3/2 bound sensitivity에서도 $\sigma_d^2 = e^5$, $q_v = e^{-12}$가 모든 stage에서 경계에 고정된다). 외란 분산이 상한, 속도 process noise가 하한이라는 것은 $d$가 $a_z$를 통째로 흡수하고 KF가 사실상 인과 2차 bandpass 적분기로 동작한다는 뜻이다. 이 값들은 reconstruction objective로 얻은 유효 모델 파라미터이며 실제 suspension parameter의 식별 결과로 해석해서는 안 된다.

### 3.9 OU KF result

| Metric | P10 | Median | P90 |
|---|---:|---:|---:|
| Correlation | 0.8854 | 0.9234 | 0.9562 |
| RMSE | 0.1657 | 0.2557 | 0.3755 |

Median lag는 10 ms였다. 이 결과는 single vertical-acceleration observation, linear oscillator, OU disturbance라는 강한 가정만으로 얻은 baseline이다.

### 3.10 No-disturbance two-state ablation

OU disturbance state를 제거한 2-state oscillator도 같은 split과 fitting 조건으로 평가했다.

$$
\mathbf{x}(t)=\begin{bmatrix}z(t)&v(t)\end{bmatrix}^{\mathsf T}
$$

$$
\dot z=v,\qquad \dot v=-\omega^2z-2\zeta\omega v+w_v
$$

$$
\mathbf{C}=\begin{bmatrix}-\omega^2&-2\zeta\omega\end{bmatrix}
$$

| Model | Corr. P10 | Corr. median | Corr. P90 | RMSE P10 | RMSE median | RMSE P90 | Median lag |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2-state KF | 0.6019 | 0.7140 | 0.8639 | 0.2735 | 0.4444 | 0.6787 | 10 ms |
| OU KF | 0.8854 | 0.9234 | 0.9562 | 0.1657 | 0.2557 | 0.3755 | 10 ms |

2-state KF의 fitted frequency는 2.4897 Hz였고, damping ratio는 lower bound인 0.05에 도달했다. OU KF보다 correlation과 RMSE가 모두 크게 나빠졌으므로, 현재 실제 데이터에서는 acceleration을 oscillator dynamics만으로 설명하기보다 latent disturbance state $d$를 함께 두는 편이 적절하다. 비교 figure는 [`oscillator_models.png`](../../lab/kalman_reconstruction/outputs/oscillator_models.png)이다.

---

## 4. OU KF + LSTM

### 4.1 Motivation

OU KF는 interpretable한 state와 안정적인 baseline을 제공하지만, 실제 차량의 nonlinear response, 운전자별 차이, 다른 센서와의 관계를 직접 표현하지 못한다. Hybrid model은 KF 출력을 버리지 않고 LSTM이 그 residual만 학습하도록 구성했다.

$$
\hat b_k^{\mathrm{hybrid}}=\hat b_k^{\mathrm{KF}}+\Delta\hat b_k^{\mathrm{LSTM}}
$$

LSTM의 target residual은 다음과 같다.

$$
e_k=b_k-\hat b_k^{\mathrm{KF}}
$$

### 4.2 LSTM input

각 시각의 LSTM input은 KF가 추정한 세 state와 innovation을 합친 4차원 vector이다.

$$
\mathbf{u}_k=\begin{bmatrix}\hat z_k&\hat v_k&\hat d_k&\nu_k\end{bmatrix}^{\mathsf T}
$$

Innovation은 current measurement와 one-step prediction의 차이이다.

$$
\nu_k=y_k-\mathbf{H}\mathbf{A}\hat{\mathbf{x}}_{k-1|k-1}
$$

이 구성에서는 LSTM이 원본 13개 센서를 직접 보지 않는다. KF state와 KF가 설명하지 못한 acceleration residual만 이용한다.

### 4.3 Network and training configuration

| Item | Setting |
|---|---|
| Recurrent network | 2-layer unidirectional LSTM |
| Input dimension | 4 |
| Hidden dimension | 64 |
| Output head | Linear 64 to 32, SiLU, Linear 32 to 1 |
| Trainable parameters | 53,313 |
| Loss | residual MSE |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Batch size | 32 episodes |
| Maximum epochs | 30 |
| LR scheduler | ReduceLROnPlateau, factor 0.2, patience 3 |
| Early stopping | patience 8, minimum improvement $10^{-5}$ |
| Gradient clipping | norm 5.0 |

KF feature와 residual target의 mean과 standard deviation은 training split에서만 계산한다. Validation driver는 조현석이며, 저장된 run은 30 epoch까지 학습되었다. Best validation MSE는 0.256338이고, training residual의 standard deviation은 0.270113이었다.

Unidirectional LSTM에는 미래 sample이 입력되지 않으므로 KF와 residual network 모두 online causal 구조이다.

### 4.4 Hybrid result

| Metric | P10 | Median | P90 |
|---|---:|---:|---:|
| Correlation | 0.9601 | 0.9828 | 0.9917 |
| RMSE | 0.0795 | 0.1226 | 0.1889 |

Median lag는 0 ms였다. OU KF 단독보다 median correlation과 median RMSE가 모두 개선되었다. 다만 LSTM correction 때문에 물리 state와 최종 출력 사이의 직접적인 해석 가능성은 KF 단독보다 낮다.

---

## 5. Model-free experiments

### 5.1 Common formulation

Model-free 모델은 명시적인 vehicle dynamics나 KF state 없이 13-channel sensor sequence에서 bounce rate를 직접 추정한다.

$$
\hat b_{1:T}=f_{\boldsymbol{\phi}}\left(\mathbf{s}_{1:T}\right),\qquad \mathbf{s}_t\in\mathbb{R}^{13}
$$

Online 모델은 시각 $t$의 출력에 현재와 과거 입력만 사용한다.

$$
\hat b_t=f_{\boldsymbol{\phi}}\left(\mathbf{s}_{1:t}\right)
$$

Offline 모델은 전체 episode를 받은 뒤 과거와 미래 context를 함께 사용할 수 있다.

$$
\hat b_t=f_{\boldsymbol{\phi}}\left(\mathbf{s}_{1:T}\right)
$$

두 종류의 차이는 배포 조건이다. Online 모델은 sample이 들어오는 즉시 추론할 수 있지만, offline 모델은 episode 또는 분석 window가 끝날 때까지 기다려야 한다.

### 5.2 Common preprocessing and training

각 input channel은 training split에서 계산한 mean과 standard deviation으로 z-score normalization한다. Target은 모든 training sample에서 계산한 하나의 mean과 standard deviation으로 정규화한다.

$$
s'_{t,j}=\frac{s_{t,j}-\mu_j}{\sigma_j},\qquad b'_t=\frac{b_t-\mu_b}{\sigma_b}
$$

Network는 normalized target을 출력하며, 평가 전에 원래 scale로 되돌린다.

$$
\hat b_t=\sigma_b\hat b'_t+\mu_b
$$

공통 학습 설정은 다음과 같다.

| Item | Setting |
|---|---|
| Sequence length | 1,000 |
| Loss | sample-wise MSE |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Maximum epochs | 30 |
| LR scheduler | ReduceLROnPlateau, factor 0.2, patience 3 |
| Early stopping | patience 8, minimum improvement $10^{-5}$ |
| Gradient clipping | norm 5.0 |
| Mixed precision | enabled when CUDA is available |
| Model selection | lowest validation MSE checkpoint |

Normalization statistics 역시 test split을 보지 않고 training split에서만 계산한다.

---

## 6. Online model-free models

### 6.1 LSTM

LSTM은 13-channel input을 2-layer unidirectional recurrent network에 입력하고, 각 시각의 마지막 hidden representation을 linear head로 `Bounce_rate_6D` 하나에 mapping한다.

| Item | Setting |
|---|---|
| Input dimension | 13 |
| Recurrent layers | 2 unidirectional LSTM layers |
| Hidden dimension | 64 |
| Output head | Linear 64 to 1 |
| Batch size | 32 episodes |
| Trainable parameters | 53,569 |
| Selected epoch | 28 |
| Best validation MSE | 0.029143 |

Unidirectional recurrence 때문에 hidden state at time $t$에는 $1$부터 $t$까지의 입력만 포함된다.

### 6.2 GRU

GRU 실험은 LSTM과 동일한 입력, hidden dimension, layer 수, output head를 사용하되 recurrent cell만 GRU로 교체했다. 따라서 gating 구조의 차이를 비교하는 실험이다.

| Item | Setting |
|---|---|
| Input dimension | 13 |
| Recurrent layers | 2 unidirectional GRU layers |
| Hidden dimension | 64 |
| Output head | Linear 64 to 1 |
| Batch size | 32 episodes |
| Trainable parameters | 40,193 |
| Selected epoch | 29 |
| Best validation MSE | 0.024944 |

GRU는 LSTM보다 parameter가 적지만, 현재 split에서는 online model 중 가장 낮은 validation MSE를 보였다.

### 6.3 Transformer

각 13-dimensional input을 64-dimensional token으로 projection하고 learned positional embedding을 더한다. Transformer encoder의 self-attention에는 strictly future position을 가리는 upper-triangular causal mask를 적용했다.

$$
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases}
$$

| Item | Setting |
|---|---|
| Input projection | Linear 13 to 64 |
| Positional encoding | learned, maximum length 1,000 |
| Encoder layers | 3 |
| Attention heads | 4 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Activation and normalization | GELU, pre-layer normalization |
| Output head | LayerNorm, Linear 64 to 1 |
| Batch size | 8 episodes |
| Trainable parameters | 165,505 |
| Selected epoch | 30 |
| Best validation MSE | 0.043610 |

구현 후 길이 40의 test tensor에서 midpoint 이후의 입력만 바꾸고 앞부분 출력을 비교했다. 앞부분 출력의 최대 차이가 0이어서 future leakage가 없음을 확인했다.

### 6.4 Online held-out results

| Model | Corr. P10 | Corr. median | Corr. P90 | RMSE P10 | RMSE median | RMSE P90 | Median lag |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.9764 | 0.9908 | 0.9952 | 0.0550 | 0.0900 | 0.1620 | 0 ms |
| GRU | 0.9771 | **0.9918** | 0.9960 | 0.0525 | **0.0841** | 0.1515 | 0 ms |
| Transformer | 0.9667 | 0.9863 | 0.9933 | 0.0750 | 0.1081 | 0.1811 | 0 ms |

현재 설정에서는 GRU가 세 online model 중 가장 좋은 median correlation과 RMSE를 얻었다. 이는 GRU가 일반적으로 항상 우수하다는 의미가 아니라, 현재 dataset, split, parameter budget, 최대 30 epoch 조건에서의 결과이다.

---

## 7. Offline model-free models

### 7.1 Bi-LSTM

Bi-LSTM은 forward LSTM과 backward LSTM이 각각 episode를 처리한 뒤, 두 방향의 hidden representation을 연결한다. 따라서 시각 $t$의 출력이 $t$ 이후의 sensor sample에도 의존한다.

| Item | Setting |
|---|---|
| Input dimension | 13 |
| Recurrent layers | 2 bidirectional LSTM layers |
| Hidden dimension | 64 per direction |
| Output dimension before head | 128 |
| Output head | Linear 128 to 1 |
| Batch size | 32 episodes |
| Trainable parameters | 139,905 |
| Selected epoch | 27 |
| Best validation MSE | 0.021307 |

### 7.2 Offline Transformer

Network size와 training setup은 Transformer와 동일하다. 유일한 구조적 차이는 causal mask를 제거하여 각 query position이 episode의 모든 key position을 볼 수 있게 한 것이다.

| Item | Setting |
|---|---|
| Input projection | Linear 13 to 64 |
| Positional encoding | learned, maximum length 1,000 |
| Encoder layers | 3 |
| Attention heads | 4 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Attention mask | none |
| Activation and normalization | GELU, pre-layer normalization |
| Output head | LayerNorm, Linear 64 to 1 |
| Batch size | 8 episodes |
| Trainable parameters | 165,505 |
| Selected epoch | 29 |
| Best validation MSE | 0.056634 |

Mask를 제거했기 때문에 이 모델은 과거와 미래 sample 사이의 long-range dependency를 모두 학습할 수 있지만 real-time inference에는 사용할 수 없다.

### 7.3 Offline held-out results

| Model | Corr. P10 | Corr. median | Corr. P90 | RMSE P10 | RMSE median | RMSE P90 | Median lag |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bi-LSTM | 0.9807 | **0.9927** | 0.9967 | **0.0467** | **0.0790** | **0.1384** | 0 ms |
| Offline Transformer | 0.9586 | 0.9819 | 0.9917 | 0.0744 | 0.1203 | 0.2278 | 0 ms |

현재 설정에서는 Bi-LSTM이 두 offline model 중 가장 좋은 결과를 보였다. Future context를 사용할 수 있다는 사실만으로 성능 향상이 보장되지는 않았다. Offline Transformer는 해당 hyperparameter와 30-epoch budget에서 Bi-LSTM뿐 아니라 일부 online model보다도 낮았다.

---

## 8. Consolidated comparison

| Category | Model | Input at inference | Future context | Corr. median | RMSE median |
|---|---|---|---:|---:|---:|
| Model-based | KF | vertical acceleration | No | 0.9234 | 0.2557 |
| Hybrid | KF + LSTM | KF state and innovation derived from vertical acceleration | No | 0.9828 | 0.1226 |
| Model-free online | LSTM | 13 sensors | No | 0.9908 | 0.0900 |
| Model-free online | GRU | 13 sensors | No | **0.9918** | **0.0841** |
| Model-free online | Transformer | 13 sensors | No | 0.9863 | 0.1081 |
| Model-free offline | Bi-LSTM | 13 sensors | Yes | **0.9927** | **0.0790** |
| Model-free offline | Offline Transformer | 13 sensors | Yes | 0.9819 | 0.1203 |

전체 수치만 보면 Bi-LSTM이 가장 높은 median correlation과 가장 낮은 median RMSE를 보였고, real-time 사용이 가능한 모델 중에서는 GRU가 가장 좋았다. KF + LSTM은 GRU와 Bi-LSTM보다 오차가 크지만, vertical acceleration 하나로 동작하고 KF state를 중간 표현으로 유지한다는 차별점이 있다.

### 8.1 Interpretation boundaries

- KF와 hybrid는 vertical acceleration 하나를 기반으로 하지만 model-free 모델은 13개 sensor channel을 사용한다. 따라서 이 표는 동일 입력에서 architecture만 비교한 ablation이 아니다.
- Online과 offline의 구분은 정보 접근 범위이다. Offline model이 더 많은 정보를 볼 수 있어도 optimization과 inductive bias가 맞지 않으면 성능이 자동으로 좋아지지 않는다.
- 각 model의 parameter 수와 batch size가 동일하지 않다. 특히 Transformer는 recurrent online model보다 parameter가 많다.
- 결과는 하나의 driver split과 seed 42에 대한 것이다. 일반화 결론을 위해서는 multiple split 또는 leave-one-driver-out 반복이 필요하다.
- `Bounce_rate_6D`는 supervised training label로 사용되었다. 즉, 추론 시 금지 신호를 사용하지 않는 reconstruction이지, label 없이 학습하는 unsupervised estimation은 아니다.
- KF fitting의 파라미터는 reconstruction objective로 최적화한 유효 파라미터이므로 실제 차량의 고유 물성으로 해석할 수 없다.
- Validation MSE는 normalized learning target에서 계산된다. Hybrid는 normalized residual을, model-free 모델은 normalized bounce target을 학습하므로 두 계열의 validation MSE를 서로 직접 비교해서는 안 된다.

---

## 9. Reproduction and artifacts

OU KF + LSTM experiment:

```powershell
python -m lab.kalman_reconstruction.run ou-hybrid
```

Model-free experiments:

```powershell
python -m lab.kalman_reconstruction.run model-free
```

Selected model-free experiments can be run with the internal model names `lstm_online`, `gru_online`, `transformer_online`, `bilstm_offline`, `transformer_offline`, and `unet_offline` through the `--models` option.

Relevant implementation files:

- [`state_space.py`](../../lab/kalman_reconstruction/state_space.py), [`models.py`](../../lab/kalman_reconstruction/models.py): OU state-space model, discretization, KF, metric
- [`hybrid.py`](../../lab/kalman_reconstruction/hybrid.py): residual LSTM, normalization, training and prediction
- [`model_free.py`](../../lab/kalman_reconstruction/model_free.py): model-free architectures and common trainer
- [`run.py`](../../lab/kalman_reconstruction/run.py): data split, parameter fitting, evaluation, plotting and artifact saving

Result files:

- [`ou_lstm_metrics.csv`](../../lab/kalman_reconstruction/outputs/ou_lstm_metrics.csv)
- [`model_free_metrics.csv`](../../lab/kalman_reconstruction/outputs/model_free_metrics.csv)
- [`ou_lstm_models.png`](../../lab/kalman_reconstruction/outputs/ou_lstm_models.png)
- [`model_free_models.png`](../../lab/kalman_reconstruction/outputs/model_free_models.png)

