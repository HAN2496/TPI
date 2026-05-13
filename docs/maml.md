# MAML (Model-Agnostic Meta-Learning)

---

## 1. 개요

MAML은 **"학습하는 법을 학습"** 하는 메타 학습 알고리즘입니다. 소수의 샘플만으로 새로운 task에 빠르게 적응할 수 있는 **초기 파라미터 $\theta$** 를 학습하는 것이 핵심입니다.

본 구현에서는 **학습 시 보지 않은 새로운 운전자(test driver)의 소량 데이터**만으로 bump 통과 위험도 판별 모델을 빠르게 적응시키는 데 적용합니다.

**핵심 아이디어**:
- 학습 운전자 각각을 하나의 **task**로 취급
- 모든 task에서 몇 번의 gradient step만으로 적응할 수 있는 **범용 초기 파라미터**를 meta-learning
- Test 시 새 운전자의 소량 데이터(context)로 inner loop 적응 → 나머지 데이터(holdout)로 평가

**원 논문**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017

---

## 2. 파이프라인

```
Input: 운전자별 시계열 데이터 (IMU, Pitch, Bounce 등)

↓

[1] Build
- 학습 운전자별 (item_ids, labels) 구성
- 전체 데이터 Z-score 정규화
- MetaCNNModel 초기화

↓

[2] Meta-Training (Outer + Inner Loop)
- 에폭마다 운전자 중 task 랜덤 샘플링
- Inner loop: support set으로 fast adaptation
- Outer loop: query set loss로 meta-gradient 계산 및 초기 파라미터 업데이트

↓

[3] Sequential Evaluation (Test Driver)
- X_test 앞 절반: context (t=1 → split_idx)
- X_test 뒤 절반: 고정 holdout
- t 증가에 따라 inner loop 적응 후 holdout AUROC 측정
```

---

## 3. 데이터 구조

### Task 구성

각 학습 운전자 $u$의 데이터를 하나의 task $\mathcal{T}_u$로 정의합니다.

$$\mathcal{T}_u = \{(x_i, y_i)\}_{i=1}^{n_u}, \quad y_i \in \{0, 1\}$$

각 에폭에서 task를 샘플링한 뒤 support/query set으로 균형 분할합니다:

$$\mathcal{S}_u = \{x_i^s, y_i^s\}_{i=1}^{N_s}, \quad \mathcal{Q}_u = \{x_i^q, y_i^q\}_{i=1}^{N_q}$$

**균형 샘플링**: positive/negative 비율을 맞춤

$$N_s^+ = N_s^- = \frac{N_s}{2}, \quad N_q^+ = N_q^- = \frac{N_q}{2}$$

**설정**: `n_support=20`, `n_query=20`, `n_tasks_per_epoch=20`

### 정규화

$$\bar{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$

where $\mu_{\text{train}}, \sigma_{\text{train}}$은 학습 운전자 전체 데이터의 axis=(sample, time) 평균/표준편차

---

## 4. MetaCNNModel

### 아키텍처

입력 $x \in \mathbb{R}^{T \times D}$ (시간 길이 $T$, 피처 차원 $D$)에 대해:

**Conv Block**:

$$h = \text{LeakyReLU}(\text{Conv1D}_{D \to h}(x^\top)) \in \mathbb{R}^{h \times T}$$

$$h = \text{LeakyReLU}(\text{Conv1D}_{h \to h}(h)) \in \mathbb{R}^{h \times T}$$

(kernel\_size=3, padding=1 → 시간 축 길이 보존)

**Global Max Pooling**:

$$h = \text{MaxPool1D}(h, T) \in \mathbb{R}^h$$

**MLP Head**:

$$\hat{r} = \text{Linear}_{h \to 1}(\text{LeakyReLU}(\text{Linear}_{h \to h}(h))) \in \mathbb{R}$$

전체 forward:

$$f_\theta(x) = \hat{r} \in \mathbb{R} \quad \text{(logit)}$$

$$P(y=1 \mid x) = \sigma(f_\theta(x))$$

**설정**: `hidden_dim=64`

---

## 5. Meta-Training

### 손실 함수

$$\mathcal{L}(\mathcal{D}; \theta) = \frac{1}{|\mathcal{D}|} \sum_{(x,y) \in \mathcal{D}} \text{BCE}(f_\theta(x), y)$$

$$= -\frac{1}{|\mathcal{D}|} \sum_{(x,y)} \left[ y \log \sigma(f_\theta(x)) + (1-y) \log(1 - \sigma(f_\theta(x))) \right]$$

### Inner Loop (Fast Adaptation)

Task $\mathcal{T}_u$에 대해 support set $\mathcal{S}_u$로 $K$번 SGD step:

$$\theta_u^{(0)} = \theta$$

$$\theta_u^{(k)} = \theta_u^{(k-1)} - \alpha \nabla_{\theta_u^{(k-1)}} \mathcal{L}(\mathcal{S}_u;\, \theta_u^{(k-1)}), \quad k = 1, \ldots, K$$

- $\alpha$: inner learning rate
- $K$: inner steps
- **원본 파라미터 $\theta$는 변경되지 않음** → `higher` 라이브러리로 differential inner loop 구현

**설정**: `inner_lr=0.005`, `inner_steps=3`

### Outer Loop (Meta-Update)

에폭당 $N_T$개 task를 샘플링하여 적응된 파라미터 $\theta_u^{(K)}$로 query loss 계산:

$$\mathcal{L}_{\text{meta}} = \frac{1}{N_T} \sum_{u \sim p(\mathcal{T})} \mathcal{L}\!\left(\mathcal{Q}_u;\, \theta_u^{(K)}\right)$$

$$\theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{\text{meta}}$$

**핵심**: outer gradient $\nabla_\theta \mathcal{L}_{\text{meta}}$는 inner loop를 **통해서** 역전파됩니다 — 즉, "support set 적응 후 query loss를 최소화하는 초기 파라미터"를 직접 학습합니다.

$$\nabla_\theta \mathcal{L}_{\text{meta}} = \nabla_\theta \mathcal{L}(\mathcal{Q}_u;\, \theta_u^{(K)}) \cdot \frac{\partial \theta_u^{(K)}}{\partial \theta}$$

where $\dfrac{\partial \theta_u^{(K)}}{\partial \theta}$는 $K$단계 SGD를 역전파한 2차 미분 항입니다.

**설정**: `outer_lr=0.001` (Adam), `n_tasks_per_epoch=20`, `meta_epochs=200`

### 전체 학습 알고리즘

```
초기화: θ ← random
for epoch = 1 to meta_epochs:
    meta_loss = 0
    for task = 1 to n_tasks_per_epoch:
        u ← sample random driver
        S_u, Q_u ← balanced_split(data[u])

        # Inner loop
        θ'_u ← θ
        for k = 1 to inner_steps:
            θ'_u ← θ'_u - α · ∇_{θ'_u} L(S_u; θ'_u)

        # Query loss
        meta_loss += L(Q_u; θ'_u)

    # Outer update (meta-gradient)
    θ ← θ - β · ∇_θ meta_loss
```

---

## 6. Test-time Evaluation (Sequential Holdout)

학습이 끝난 $\theta$에서 시작하여, 새 운전자 데이터를 **순차적으로** 늘려가며 adaptation 성능을 측정합니다.

### 절차

$$X_{\text{test}} = [x_1, \ldots, x_N], \quad \text{split\_idx} = \lfloor N/2 \rfloor$$

**Holdout** (고정): $X_{\text{holdout}} = X_{\text{test}}[\text{split\_idx}:]$

**Context** (증가): $t = 1, 2, \ldots, \text{split\_idx}$

각 $t$에 대해:

**1. Context로 inner loop 적응**:

$$\theta_t^{(0)} = \theta$$

$$\theta_t^{(k)} = \theta_t^{(k-1)} - \alpha \nabla_{\theta_t^{(k-1)}} \mathcal{L}(X_{\text{test}}[:t];\, \theta_t^{(k-1)})$$

**2. Holdout 평가**:

$$p_i = \sigma(f_{\theta_t^{(K)}}(x_i^{\text{holdout}}))$$

$$\text{AUROC}(t) = \text{AUC}(y_{\text{holdout}},\, p)$$

**3. Sequential AUROC 곡선 저장** + 10/20/30/40/50% 시점 snapshot

이를 통해 **"context 개수가 늘어날수록 adaptation 성능이 얼마나 향상되는가"** 를 시각화합니다.

---

## 7. MAML vs. 일반 Fine-tuning

| | MAML | 일반 Fine-tuning |
|---|---|---|
| 학습 목표 | 빠른 적응이 가능한 초기값 $\theta$ 찾기 | 특정 task에 최적화된 $\theta$ 찾기 |
| Test 적응 | Inner loop $K$번 step | 다수 epoch fine-tuning |
| 필요 데이터 | Support set (소량) | 충분한 학습 데이터 |
| Gradient | 2차 미분 (through inner loop) | 1차 미분 |
| 목적 | Generalization across tasks | Single task performance |

---

## 8. 코드 위치

| 컴포넌트 | 파일 |
|---|---|
| **Main Script** | `scripts/run_maml.py` |
| **모델 (MetaCNNModel)** | `src/model/maml/rm.py` |
| **Trainer** | `src/model/maml/trainer.py` |
| **Experiment** | `src/model/maml/experiment.py` |

---

## 9. 설정

`scripts/run_maml.py`:

```python
@dataclass
class Config:
    features: list = ["IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"]
    train_driver_names: list = ["김진명", "김태근", "조현석", "한규택", "박재일", "이지환"]
    test_driver_name: str = "강신길"
    time_range: tuple = (5, 7)
    downsample: int = 5

    hidden_dim: int = 64       # MetaCNNModel hidden size

    inner_lr: float = 0.005    # α: inner loop learning rate
    outer_lr: float = 0.001    # β: outer loop learning rate (Adam)
    inner_steps: int = 3       # K: inner gradient steps
    n_support: int = 20        # support set size per task
    n_query: int = 20          # query set size per task
    n_tasks_per_epoch: int = 20
    meta_epochs: int = 200
```

---

## 10. 출력

`artifacts/maml/<timestamp>/`

| 파일 | 내용 |
|---|---|
| `best_maml.pt` | 최소 meta query loss 시점의 모델 파라미터 |
| `metrics.txt` | test/train AUROC, AUPRC, Brier score |
| `plots/sequential_auroc_<운전자>.png` | context 크기별 AUROC 곡선 |
| `plots/snapshots/context_{pct}pct/` | 각 context 비율에서의 상세 평가 플롯 |
| `plots/train/<운전자>/` | 학습 운전자 기본 모델 평가 |

---

## 참고문헌

- **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
- **higher (library)**: Grefenstette et al., "Generalized Inner Loop Meta-Learning", arXiv 2019
