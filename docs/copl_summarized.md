# CoPL Collaborative Preference Learning - 핵심 요약
  

> **전체 문서**: [copl.md](./copl.md)

> **설정**: VAE Similarity + GCF (기본 버전) + CNN Reward Model

  

---

  

## 1. 개요

  

CoPL은 협업 필터링 기반 개인화 학습 프레임워크입니다. 본 구현에서는 **운전자별 bump 통과 상황에 대한 선호도 학습**에 적용합니다.



**핵심 아이디어**:

- 운전자(user)가 특정 driving scenario(item)에 대해 good($y=1$)/bad($y=0$) 선호도 피드백을 나타내는 패턴을 그래프로 모델링

- 협업 필터링으로 user/item 임베딩 학습

- 새로운 운전자에게 test-time adaptation으로 적응

  

**논문**: [CoPL: Collaborative Preference Learning for Personalizing LLMs](https://arxiv.org/abs/2503.01658), EMNLP 2025

  

---

  

## 2. 파이프라인

  

```

Input: 운전자별 시계열 데이터 (IMU, Pitch, Bounce 등)

↓

[1] Graph Construction

- User-Item Bipartite Graph (positive/negative edges)

- Item-Item Similarity Graph (VAE 기반)

↓

[2] GCF Training

- Learn E_u (user embeddings), E_i (item embeddings)

- Message passing on pos/neg graphs

↓

[3] Reward Model Training

- Input: (e_u, x) → Output: risk score

- CNN-based architecture

↓

[4] Test-time Adaptation

- Attach test items via k-NN

- Aggregate user embeddings via attention

```

  

---

  

## 3. 데이터 구조

  

### User-Item Bipartite Graph

  

$$A_{\text{pos}} \in \{0,1\}^{n_u \times n_i}, \quad A_{\text{pos}}[u,i] = 1 \text{ if user } u \text{ gave good feedback (}y=1\text{) to item } i$$



$$A_{\text{neg}} \in \{0,1\}^{n_u \times n_i}, \quad A_{\text{neg}}[u,i] = 1 \text{ if user } u \text{ gave bad feedback (}y=0\text{) to item } i$$

  

정규화 (symmetric normalization):

  

$$\bar{A}_{\text{pos}} = D_u^{-1/2} A_{\text{pos}} D_i^{-1/2}, \quad \bar{A}_{\text{neg}} = D_u^{-1/2} A_{\text{neg}} D_i^{-1/2}$$

  

where $D_u = \text{diag}(\sum_i A[u,i])$, $D_i = \text{diag}(\sum_u A[u,i])$

  

### Item-Item Similarity Graph (VAE)

  

**1. VAE로 시계열 인코딩**

  

$$\text{Encoder: } x \in \mathbb{R}^{T \times D} \to (\mu(x), \sigma^2(x)) \in \mathbb{R}^{d_{\text{latent}}}$$

  

$$z \sim \mathcal{N}(\mu(x), \sigma^2(x))$$

  

**2. 코사인 유사도 계산**

  

$$\text{sim}(i, j) = \frac{z_i \cdot z_j}{||z_i|| \, ||z_j||}$$

  

**3. k-NN으로 Sparse Graph 구성**

  

$$A_{ii}[i, j] = \begin{cases}

\text{sim}(i, j) & \text{if } j \in \text{TopK}(i) \\

0 & \text{otherwise}

\end{cases}$$

  

정규화:

  

$$\bar{A}_{ii} = D_{ii}^{-1/2} A_{ii} D_{ii}^{-1/2}$$

  

**설정**: `vae_latent_dim=16`, `vae_epochs=400`, `knn_k=100`

  

---

  

## 4. GCF (Graph Collaborative Filtering)

  

### 초기화

  

$$E_u^{(0)} \in \mathbb{R}^{n_u \times d}, \quad E_i^{(0)} \in \mathbb{R}^{n_i \times d} \quad \text{(Xavier initialization)}$$

  

### Message Passing (Layer $l = 1, \ldots, L$)

  

**User Update:**



정규화 계수: $\alpha_{u,i}^+ = \dfrac{1}{\sqrt{|N_u^+||N_i^+|}},\quad \alpha_{u,i}^- = \dfrac{1}{\sqrt{|N_u^-||N_i^-|}}$



$$m_u^{+,(l)} = \sum_{i \in N_u^+} \alpha_{u,i}^+ \left[ W_u^1 e_i^{(l-1)} + W_u^2 \left( e_i^{(l-1)} \odot e_u^{(l-1)} \right) \right]$$



$$m_u^{-,(l)} = \sum_{i \in N_u^-} \alpha_{u,i}^- \left[ W_u^3 e_i^{(l-1)} + W_u^4 \left( e_i^{(l-1)} \odot e_u^{(l-1)} \right) \right]$$



$$m_u^{0,(l)} = W_u^0 e_u^{(l-1)} \quad \text{(self-connection)}$$



$$e_u^{(l)} = \text{LeakyReLU}\!\left( m_u^{0,(l)} + m_u^{+,(l)} + m_u^{-,(l)} \right)$$

  

**Item Update:**



정규화 계수: $\beta_{i,j}$는 VAE 코사인 유사도 기반 normalized weight



$$m_i^{+,(l)} = \sum_{u \in N_i^+} \alpha_{u,i}^+ \left[ W_i^1 e_u^{(l-1)} + W_i^2 \left( e_u^{(l-1)} \odot e_i^{(l-1)} \right) \right]$$



$$m_i^{-,(l)} = \sum_{u \in N_i^-} \alpha_{u,i}^- \left[ W_i^3 e_u^{(l-1)} + W_i^4 \left( e_u^{(l-1)} \odot e_i^{(l-1)} \right) \right]$$



$$m_i^{ii,(l)} = \sum_{j \in N_i^{ii}} \beta_{i,j} \left[ W_i^5 e_j^{(l-1)} + W_i^6 \left( e_j^{(l-1)} \odot e_i^{(l-1)} \right) \right]$$



$$m_i^{0,(l)} = W_i^0 e_i^{(l-1)} \quad \text{(self-connection)}$$



$$e_i^{(l)} = \text{LeakyReLU}\!\left( m_i^{0,(l)} + m_i^{+,(l)} + m_i^{-,(l)} + \lambda_{ii} m_i^{ii,(l)} \right)$$

  

**핵심**:

- $\odot$: Hadamard product (element-wise)

- Positive/Negative 그래프를 **분리**하여 대조 학습

- Item-item 그래프로 content-based 정보 보완 (가중치 $\lambda_{ii}$)

  

### Final Embeddings

  

$$E_u = \text{normalize}(E_u^{(L)}), \quad E_i = E_i^{(L)}$$

  

### 손실 함수 (Weighted BCE)

  

$$\text{score}(u, i) = e_u^{(L)} \cdot e_i^{(L)}$$

  

$$\mathcal{L} = -\sum_{(u,i,y)} w_y [y \log \sigma(\text{score}) + (1-y) \log(1 - \sigma(\text{score}))] + \lambda_{\text{reg}} (||E_u||^2 + ||E_i||^2)$$

  

where $w_1 = \frac{\#\text{negative}}{\#\text{positive}}$ (class imbalance 처리)

  

**하이퍼파라미터**:

- `gcf_emb_dim`: 128

- `gcf_layers`: 2

- `item_item_weight` ($\lambda_{ii}$): 0.72

- `gcf_lr`: 0.00068

  

---

  

## 5. Reward Model (CNN)

  

### 입력/출력

  

- **Input**: $e_u \in \mathbb{R}^d$ (user embedding), $x \in \mathbb{R}^{T \times D}$ (observation sequence)

- **Output**: $r \in [0, 1]$ (risk preference score)

  

### Forward Pass

  

$$h = \text{LeakyReLU}(\text{Linear}_{\text{obs}}(x) + \text{Linear}_u(e_u)) \in \mathbb{R}^{T \times h}$$

  

$$h = \text{Conv1D}(h) \in \mathbb{R}^{T \times h} \quad \text{(kernel\_size=3, layers=2)}$$

  

$$h = \text{MaxPool1D}(h, T) \in \mathbb{R}^h$$

  

$$r = \sigma(\text{MLP}(h))$$

  

**파라미터**:

- `rm_hidden`: 32

- `rm_kernel_size`: 3

- `rm_layers`: 2

- `rm_batch_size`: 256

  

---

  

## 6. Test-time Adaptation

  

새로운 운전자의 소량 샘플 $\{(x_1, y_1), \ldots, (x_t, y_t)\}$로 user embedding 생성

  

### Step 1: Attach Test Items

  

$$z_i = \text{VAE\_Encoder}(x_i)$$

  

$$N_i = \text{TopK}_j \, \text{sim}(z_i, z_{\text{train}}[j])$$

  

$$w_i = \text{softmax}([\text{sim}(z_i, z_{\text{train}}[j])]_{j \in N_i})$$

  

$$e_i^{\text{test}} = \sum_{j \in N_i} w_i[j] E_i^{\text{train}}[j]$$

  

### Step 2: Adapt User Embedding

  

**Vote 벡터 구성**:

  

$$v[j] = \sum_{i: j \in N_i} \begin{cases}

w_i[j] & \text{if } y_i = 1 \\

-\alpha_{\text{neg}} w_i[j] & \text{if } y_i = 0

\end{cases}$$

  

**User attention**:

  

$$c_u = (A_{\text{pos}} - A_{\text{neg}}) v$$

  

$$w_u = \text{softmax}(c_u / \tau)$$

  

**Aggregation**:

  

$$e_u^{\text{test}} = \sum_u w_u[u] E_u^{\text{train}}[u]$$

  

**파라미터**:

- `adapt_topk`: 20

- `adapt_neg_weight` ($\alpha_{\text{neg}}$): 0.81

- `adapt_user_softmax_temp` ($\tau$): 1.15

  

---

  

## 7. 핵심 알고리즘 요약

  

### 전체 Forward Pass

  

**Training:**

  

```

1. Graph Construction:

- VAE: x → z, build A_ii via k-NN

- Build A_pos, A_neg from labels

  

2. GCF:

For l = 1 to L:

E_u^(l) = f_u(E_u^(l-1), Ā_pos·E_i^(l-1), Ā_neg·E_i^(l-1))

E_i^(l) = f_i(E_i^(l-1), Ā_pos^T·E_u^(l-1), Ā_neg^T·E_u^(l-1), Ā_ii·E_i^(l-1))

  

3. RM:

r = RM(E_u[u], x_i)

Loss = BCE(r, y)

```

  

**Test-time:**

  

```

1. Item Attachment (k-NN):

e_i^test = Σ w_j E_i^train[j] (j ∈ neighbors)

  

2. User Adaptation (attention):

e_u^test = Σ w_u E_u^train[u] (based on vote propagation)

  

3. Prediction:

r = RM(e_u^test, x_test)

```

  

---

  

## 8. 설정 예시

  

`scripts_new/run_copl.py`:

  

```python

@dataclass

class Config:

# Data

train_driver_names: list = ["김진명", "조현석", "한규택", "박재일", "이지환"]

test_driver: str = "강신길"

features: list = ["IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"]

  

# Similarity

similarity_method: str = "vae"

vae_latent_dim: int = 16

vae_epochs: int = 400

knn_k: int = 100

  

# GCF

gcf_model: str = "gcf"

gcf_emb_dim: int = 128

gcf_layers: int = 2

item_item_weight: float = 0.72

gcf_lr: float = 0.00068

gcf_epochs: int = 100

  

# RM

rm_model: str = "cnn"

rm_hidden: int = 32

rm_epochs: int = 200

  

# Adaptation

adapt_topk: int = 20

adapt_neg_weight: float = 0.81

adapt_user_softmax_temp: float = 1.15

```

  

---

  

## 9. 주요 설계 선택

  

### 왜 Positive/Negative 그래프를 분리?

  

일반 CF는 positive만 사용하지만, CoPL은 **explicit negative**도 활용:

- Positive: "이 운전자는 이 상황에 good 피드백($y=1$)을 줬다"

- Negative: "이 운전자는 이 상황에 bad 피드백($y=0$)을 줬다"

  

이를 통해 **대조 학습**(contrastive learning) 효과 획득.

  

### 왜 Item-Item 그래프?

  

**Cold-start 문제 해결**:

- User-item만으로는 새로운 user/item이 고립됨

- VAE로 시계열 유사도를 계산하여 **content-based** 정보 추가

- Test 시 k-NN으로 새로운 아이템을 기존 아이템에 연결 가능

  

### 왜 VAE?

  

- DTW는 계산 비용이 높음 ($O(T^2)$)

- PCA는 시계열의 temporal 패턴을 제대로 캡처하지 못함

- **VAE**는 시계열을 저차원 잠재 공간으로 압축하며, 확률적 표현으로 불확실성 모델링 가능

  

---

  

## 10. 코드 위치

  

| 컴포넌트 | 파일 |

|----------|------|

| **Main Script** | `scripts_new/run_copl.py` |

| **Dataset** | `src_new/model/copl/dataset.py` |

| **VAE Similarity** | `src_new/model/copl/similarity.py` (VAESimBuilder) |

| **GCF Model** | `src_new/model/copl/gcf.py` (CoPLGCF) |

| **Reward Model** | `src_new/model/copl/rm.py` (CNNRewardModel) |

| **Trainer** | `src_new/model/copl/trainer.py` |

| **Experiment** | `src_new/model/copl/experiment.py` |

  

---

  

## 11. 실행

  

```bash

python scripts_new/run_copl.py

```

  

**출력**: `artifacts/copl/YYYYMMDD_HHMMSS/`

- `best_gcf.pt`: GCF 체크포인트

- `best_rm.pt`: RM 체크포인트

- `metrics.txt`: 평가 메트릭

- `plots/`: Sequential AUROC, user attention, embeddings 등

  

---

  

## 참고문헌

  

- **CoPL Paper**: Choi et al., "CoPL: Collaborative Preference Learning for Personalizing LLMs", EMNLP 2025

- **LightGCN**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020