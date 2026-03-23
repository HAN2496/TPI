# CoPL: Collaborative Preference Learning for Personalizing Driver Risk Assessment

  

## 개요

  

CoPL (Collaborative Preference Learning)은 협업 필터링 기반 개인화 학습 프레임워크로, EMNLP 2025에 발표된 논문을 기반으로 구현되었습니다. 본 구현에서는 LLM 개인화가 아닌 **운전자별 위험 상황 선호도 학습**에 적용합니다.

  

**주요 아이디어**: 운전자(user)가 특정 운전 상황(item)에 대해 위험하다고 판단(positive) 또는 안전하다고 판단(negative)하는 선호도를 협업 필터링으로 학습하고, 새로운 운전자에게 적응합니다.

  

**논문**: [arXiv:2503.01658](https://arxiv.org/abs/2503.01658)

  

---

  

## 1. 전체 파이프라인

  

CoPL은 3단계로 구성됩니다:

  

```

┌─────────────────────────────────────────────────────────────────┐

│ Stage 1: Graph Dataset Construction │

│ - Load driver sequences (IMU, Pitch, Bounce, etc.) │

│ - Build user-item bipartite graph │

│ - Compute item-item similarity graph │

└─────────────────────────────────────────────────────────────────┘

↓

┌─────────────────────────────────────────────────────────────────┐

│ Stage 2: GCF (Graph Collaborative Filtering) │

│ - Learn user embeddings E_u ∈ ℝ^(n_u × d) │

│ - Learn item embeddings E_i ∈ ℝ^(n_i × d) │

│ - Message passing on pos/neg interaction graphs │

└─────────────────────────────────────────────────────────────────┘

↓

┌─────────────────────────────────────────────────────────────────┐

│ Stage 3: Reward Model Training │

│ - Input: user embedding e_u, observation sequence x │

│ - Output: preference score ∈ [0,1] │

│ - Architecture: CNN/MLP/MoLE-CNN │

└─────────────────────────────────────────────────────────────────┘

↓

┌─────────────────────────────────────────────────────────────────┐

│ Stage 4: Test-time Adaptation │

│ - Attach test items to train items via k-NN │

│ - Aggregate user embeddings via softmax attention │

└─────────────────────────────────────────────────────────────────┘

```

  

---

  

## 2. 데이터 구조

  

### 2.1 용어 정의

  

| 용어 | 의미 | 예시 |

|------|------|------|

| **User** | 운전자 | 김진명, 조현석, ... |

| **Item** | 특정 운전 상황 (시계열 샘플) | 시간 5~7초 구간의 IMU/Pitch/Bounce 데이터 |

| **Label** | 위험도 (0: 안전, 1: 위험) | DLC 여부로 결정 |

| **Positive edge** | 운전자-위험 상황 연결 | (김진명, item_42) with label=1 |

| **Negative edge** | 운전자-안전 상황 연결 | (김진명, item_58) with label=0 |

  

### 2.2 그래프 구조

  

#### User-Item Bipartite Graph

  

```

Users: u₁ u₂ u₃ ... u_n

│ │ │ │

╱│╲ ╱│╲ ╱│╲ ╱│╲

╱ │ ╲ ╱ │ ╲ ╱ │ ╲ ╱ │ ╲

│ │ │ │ │ │ │ │ │ │ │ │

Items: i₁ i₂ i₃ i₄ i₅ i₆ i₇ i₈ ... i_m

  

Edges:

- Green (positive): user prefers/flags as risky

- Red (negative): user considers safe

```

  

수식적 표현:

  

$$A_{\text{pos}} \in \{0,1\}^{n_u \times n_i}: \quad A_{\text{pos}}[u,i] = 1 \text{ if user } u \text{ labeled item } i \text{ as risky}$$

  

$$A_{\text{neg}} \in \{0,1\}^{n_u \times n_i}: \quad A_{\text{neg}}[u,i] = 1 \text{ if user } u \text{ labeled item } i \text{ as safe}$$

  

정규화된 인접 행렬:

  

$$\bar{A}_{\text{pos}} = D_u^{-1/2} A_{\text{pos}} D_i^{-1/2}$$

  

$$\bar{A}_{\text{neg}} = D_u^{-1/2} A_{\text{neg}} D_i^{-1/2}$$

  

where $D_u = \text{diag}(\sum_i A[u,i])$, $D_i = \text{diag}(\sum_u A[u,i])$

  

**코드 위치**: `src_new/model/copl/dataset.py:88-96`

  

#### Item-Item Similarity Graph

  

아이템 간 유사도는 다음 방법 중 하나로 계산:

  

1. **VAE (Variational Autoencoder)** - default

- 시계열을 잠재 공간으로 인코딩: $z \sim \mathcal{N}(\mu(x), \sigma^2(x))$

- 코사인 유사도 계산: $\text{sim}(i,j) = \cos(z_i, z_j)$

- k-NN으로 sparse graph 구성

  

2. **DTW (Dynamic Time Warping)**

- 시계열 간 정렬된 거리 계산

- RBF kernel: $w_{ij} = \exp(-\text{DTW}(x_i, x_j) / \gamma)$

  

3. **PCA / Kernel PCA**

- 차원 축소 후 유클리드 거리

  

수식:

  

$$A_{ii} \in \mathbb{R}^{n_i \times n_i}: \quad A_{ii}[i,j] = \text{similarity}(\text{item}_i, \text{item}_j)$$

  

$$\bar{A}_{ii} = D_{ii}^{-1/2} A_{ii} D_{ii}^{-1/2} \quad \text{(symmetric normalization)}$$

  

**설정 파라미터** (`run_copl.py:46-63`):

- `similarity_method`: "vae" (default), "dtw", "pca", "kernel_pca"

- `knn_k`: k-NN에서 k 값 (default: 100)

- `mutual`: True이면 상호 k-NN만 유지

- `vae_latent_dim`: VAE 잠재 차원 (default: 16)

  

---

  

## 3. GCF (Graph Collaborative Filtering)

  

### 3.1 모델 아키텍처

  

#### 초기화

  

$$E_u^{(0)} \in \mathbb{R}^{n_u \times d}: \quad \text{User embedding (learnable parameter)}$$

  

$$E_i^{(0)} \in \mathbb{R}^{n_i \times d}: \quad \text{Item embedding (learnable parameter)}$$

  

Xavier uniform initialization으로 초기화됩니다.

  

**코드**: `src_new/model/copl/gcf.py:30-31`

  

#### Message Passing (Layer l)

  

각 레이어에서 다음 연산 수행:

  

##### User Update

  

$$Z_u^{\text{pos}} = \bar{A}_{\text{pos}} \cdot E_i^{(l-1)} \quad \text{(positive neighbors aggregation)}$$

  

$$Z_u^{\text{neg}} = \bar{A}_{\text{neg}} \cdot E_i^{(l-1)} \quad \text{(negative neighbors aggregation)}$$

  

$$\begin{align}

m_u &= W_u^{\text{self}} \cdot E_u^{(l-1)} \\

&+ W_u^{\text{pos1}} \cdot Z_u^{\text{pos}} \\

&+ W_u^{\text{pos2}} \cdot (Z_u^{\text{pos}} \odot E_u^{(l-1)}) \quad \text{(element-wise interaction)} \\

&+ W_u^{\text{neg3}} \cdot Z_u^{\text{neg}} \\

&+ W_u^{\text{neg4}} \cdot (Z_u^{\text{neg}} \odot E_u^{(l-1)})

\end{align}$$

  

$$E_u^{(l)} = \text{LeakyReLU}(m_u)$$

  

여기서:

- $\odot$: element-wise product (Hadamard product)

- $W_u^{\text{self}}, W_u^{\text{pos1}}, W_u^{\text{pos2}}, W_u^{\text{neg3}}, W_u^{\text{neg4}} \in \mathbb{R}^{d \times d}$: learnable weight matrices

  

**핵심**: positive와 negative 이웃을 **따로** 집계하고, self-interaction term ($Z \odot E$)을 통해 비선형성 추가

  

**코드**: `src_new/model/copl/gcf.py:64-68`

  

##### Item Update

  

$$Z_i^{\text{pos}} = \bar{A}_{\text{pos}}^T \cdot E_u^{(l-1)} \quad \text{(users who liked this item)}$$

  

$$Z_i^{\text{neg}} = \bar{A}_{\text{neg}}^T \cdot E_u^{(l-1)} \quad \text{(users who disliked this item)}$$

  

$$Z_i^{ii} = \bar{A}_{ii} \cdot E_i^{(l-1)} \quad \text{(similar items)}$$

  

$$\begin{align}

m_i &= W_i^{\text{self}} \cdot E_i^{(l-1)} \\

&+ W_i^{\text{pos1}} \cdot Z_i^{\text{pos}} \\

&+ W_i^{\text{pos2}} \cdot (Z_i^{\text{pos}} \odot E_i^{(l-1)}) \\

&+ W_i^{\text{neg3}} \cdot Z_i^{\text{neg}} \\

&+ W_i^{\text{neg4}} \cdot (Z_i^{\text{neg}} \odot E_i^{(l-1)}) \\

&+ \lambda_{ii} \cdot (W_i^{ii1} \cdot Z_i^{ii} + W_i^{ii2} \cdot (Z_i^{ii} \odot E_i^{(l-1)}))

\end{align}$$

  

$$E_i^{(l)} = \text{LeakyReLU}(m_i)$$

  

여기서:

- $\lambda_{ii}$ (item_item_weight): 아이템 유사도 그래프의 가중치 (default: 0.72)

- $W_i^{ii1}, W_i^{ii2}$: item-item graph용 가중치

  

**코드**: `src_new/model/copl/gcf.py:70-80`

  

#### Final Embeddings

  

L개 레이어 적용 후:

  

$$E_u^{\text{final}} = \text{normalize}(E_u^{(L)}, \text{dim}=-1) \quad \text{(L2 normalization for users)}$$

  

$$E_i^{\text{final}} = E_i^{(L)} \quad \text{(no normalization for items)}$$

  

**코드**: `src_new/model/copl/gcf.py:85-86`

  

### 3.2 손실 함수

  

#### 3.2.1 기본: Weighted Binary Cross-Entropy

  

$$\text{score}(u, i) = E_u[u] \cdot E_i[i] \quad \text{(inner product)}$$

  

$$\mathcal{L}_{\text{BCE}} = -\sum_{(u,i,y)} w_y \cdot [y \log \sigma(\text{score}) + (1-y) \log(1 - \sigma(\text{score}))] + \lambda_{\text{reg}} \cdot (||E_u||^2 + ||E_i||^2)$$

  

where:

- $w_1 = \text{pos\_weight}$ (imbalanced class 처리)

- $\sigma(x) = \text{sigmoid function}$

  

**설정**: `use_pos_weight=True`면 pos_weight = (#negative / #positive)

  

**코드**: `src_new/model/copl/gcf.py:100-108`

  

#### 3.2.2 변형 1: Cosine Embedding Loss

  

$$E_u^{\text{final}} = \text{normalize}(E_u^{(L)})$$

  

$$E_i^{\text{final}} = \text{normalize}(E_i^{(L)})$$

  

$$\text{score}(u, i) = E_u[u] \cdot E_i[i] \quad \text{(cosine similarity)}$$

  

$$\mathcal{L}_{\text{cosine}} = \sum \max(0, \text{margin} - y_{\text{signed}} \cdot \text{score}(u, i))$$

  

where $y_{\text{signed}} = 2y - 1 \in \{-1, +1\}$

  

**모델**: `CoPLGCFCosine` (gcf_model="gcf_cosine")

  

**코드**: `src_new/model/copl/gcf.py:111-166`

  

#### 3.2.3 변형 2: Pointwise BPR

  

$$\mathcal{L}_{\text{BPR}} = -\sum [y \log \sigma(\text{score}) \cdot w_{\text{pos}} + (1-y) \log \sigma(-\text{score})] + \lambda_{\text{reg}} \cdot (||E_u||^2 + ||E_i||^2)$$

  

**모델**: `CoPLGCFPointwiseBPR` (gcf_model="gcf_pointwise_bpr")

  

**코드**: `src_new/model/copl/gcf.py:169-189`

  

#### 3.2.4 변형 3: InfoNCE (Softmax)

  

positive 샘플만 사용:

  

$$S = \frac{E_u^{\text{pos}} \cdot (E_i^{\text{pos}})^T}{\tau} \quad \text{(similarity matrix, batch\_size × batch\_size)}$$

  

$$\mathcal{L}_{\text{InfoNCE}} = -\sum_i \log \left[ \frac{\exp(S[i,i])}{\sum_j \exp(S[i,j])} \right]$$

  

대각선 원소가 정답 (i번째 user-item 쌍)

  

**모델**: `CoPLGCFSoftmax` (gcf_model="gcf_softmax")

  

**코드**: `src_new/model/copl/gcf.py:192-218`

  

#### 3.2.5 변형 4: Margin Ranking Loss

  

$$\mathcal{L}_{\text{margin}} = -\sum_{i \in \text{pos}, j \in \text{neg}} \log \sigma(\text{score}_i - \text{score}_j) \cdot \text{bal\_weight} + \lambda_{\text{reg}} \cdot (||E_u||^2 + ||E_i||^2)$$

  

where $\text{bal\_weight} = \frac{\#\text{neg}}{\#\text{pos}}$

  

**모델**: `CoPLGCFMargin` (gcf_model="gcf_margin")

  

**코드**: `src_new/model/copl/gcf.py:221-241`

  

### 3.3 하이퍼파라미터

  

| 파라미터 | 의미 | Default |

|----------|------|---------|

| `gcf_emb_dim` | 임베딩 차원 d | 128 |

| `gcf_layers` | 레이어 수 L | 2 |

| `gcf_dropout` | Sparse dropout 확률 | 0.0 |

| `item_item_weight` | λ_ii (아이템 그래프 가중치) | 0.72 |

| `gcf_lr` | Learning rate | 0.00068 |

| `gcf_weight_decay` | Adam weight decay | 0.001 |

| `gcf_lambda_reg` | L2 regularization | 0.0 |

| `gcf_epochs` | Training epochs | 100 |

  

**코드**: `scripts_new/run_copl.py:33-44`

  

---

  

## 4. Reward Model

  

GCF로 학습된 user/item 임베딩을 사용해 선호도 점수를 예측합니다.

  

### 4.1 입력/출력

  

**Input:**

- $e_u \in \mathbb{R}^d$: user embedding (from GCF)

- $x \in \mathbb{R}^{T \times D}$: item observation sequence (T=time steps, D=feature dim)

  

**Output:**

- $r \in [0, 1]$: preference score

  

### 4.2 아키텍처

  

#### 4.2.1 기본: MLP

  

$$h_{\text{obs}} = \text{Linear}(D \to \text{hidden})(x) \in \mathbb{R}^{T \times \text{hidden}}$$

  

$$h_u = \text{Linear}(d \to \text{hidden})(e_u) \in \mathbb{R}^{\text{hidden}}$$

  

$$h = \text{LeakyReLU}(h_{\text{obs}} + h_u).\text{mean}(\text{dim}=1) \in \mathbb{R}^{\text{hidden}}$$

  

$$\text{out} = \text{MLP}(\text{hidden} \to \text{hidden} \to 1)(h)$$

  

$$r = \sigma(\text{out})$$

  

**코드**: `src_new/model/copl/rm.py:7-22`

  

#### 4.2.2 CNN (default)

  

$$h = \text{LeakyReLU}(\text{Linear}_{\text{obs}}(x) + \text{Linear}_u(e_u)) \in \mathbb{R}^{T \times \text{hidden}}$$

  

$$h = \text{Conv1D}(h, \text{kernel\_size}=k, \text{layers}=L) \in \mathbb{R}^{T \times \text{hidden}}$$

  

$$h = \text{MaxPool1D}(h, T) \in \mathbb{R}^{\text{hidden}}$$

  

$$r = \sigma(\text{MLP}(h))$$

  

**파라미터**:

- `rm_hidden`: CNN hidden channels (default: 32)

- `rm_kernel_size`: 3

- `rm_layers`: 2

  

**코드**: `src_new/model/copl/rm.py:25-47`

  

#### 4.2.3 MoLE-CNN (Mixture of LoRA Experts)

  

User 임베딩을 기반으로 expert를 선택하는 MoE 구조:

  

**Gating Network:**

  

$$\text{soft}_w = \text{softmax}(\text{MLP}(e_u) / \tau) \in \mathbb{R}^{\text{num\_experts}}$$

  

$$\text{hard}_w = \text{one\_hot}(\arg\max(\text{soft}_w))$$

  

$$\text{routing}_w = \text{hard}_w - \text{soft}_w.\text{detach}() + \text{soft}_w \quad \text{(straight-through estimator)}$$

  

**MoLE Linear Layer:**

  

$$\text{out} = W_{\text{base}} \cdot x + W_{\text{shared}} \cdot x + \sum_k \text{routing}_w[k] \cdot (A_k \cdot x) \cdot B_k$$

  

where $A_k \in \mathbb{R}^{\text{in} \times \text{rank}}$, $B_k \in \mathbb{R}^{\text{rank} \times \text{out}}$

  

(shared LoRA와 expert-specific LoRA 결합)

  

**파라미터**:

- `rm_num_experts`: 4

- `rm_mole_rank`: 6

- `rm_mole_tau`: 2.0

  

**코드**: `src_new/model/copl/rm.py:78-129`

  

### 4.3 학습

  

$$\mathcal{L}_{\text{RM}} = \text{BCE}(\sigma(\text{RM}(e_u, x)), y) + \lambda_{\text{reg}} \cdot ||\theta_{\text{RM}}||^2$$

  

**설정**:

- `rm_batch_size`: 256

- `rm_lr`: 0.00026

- `rm_epochs`: 200

- `rm_lambda_reg`: 1e-6

  

**코드**: `src_new/model/copl/trainer.py` (CoPLRMTrainer)

  

---

  

## 5. Test-time Adaptation

  

새로운 운전자(unseen user)의 소량 샘플로 user embedding을 생성합니다.

  

### 5.1 알고리즘

  

#### Step 1: Attach Test Items to Train Items

  

새로운 아이템 $x_{\text{test}}$를 학습된 아이템들과 연결:

  

**1. Transform test item to latent space:**

  

$$z_{\text{test}} = \text{Encoder}(x_{\text{test}}) \quad \text{(using same VAE/DTW/PCA as training)}$$

  

**2. Find k-NN in train items:**

  

$$\text{neigh\_idx}[i] = \text{TopK}_j(\text{similarity}(z_{\text{test}}[i], z_{\text{train}}[j]))$$

  

$$\text{neigh}_w[i] = \text{softmax}([\text{similarity}(z_{\text{test}}[i], z_{\text{train}}[j]) \text{ for } j \in \text{neigh\_idx}[i]])$$

  

**3. Aggregate train item embeddings:**

  

$$e_i^{\text{test}}[i] = \sum_j \text{neigh}_w[i,j] \cdot E_i^{\text{train}}[\text{neigh\_idx}[i,j]]$$

  

**파라미터**: `adapt_topk=20` (k-NN에서 k)

  

**코드**: `src_new/model/copl/dataset.py:109-117`

  

#### Step 2: Adapt User Embedding

  

테스트 라벨을 기반으로 train user들의 가중 평균:

  

**1. Build vote vector** $v \in \mathbb{R}^{n_{\text{train\_items}}}$:

  

for each test sample $(x, y)$:

- if $y = 1$ (risky): $v[\text{neigh\_idx}] \mathrel{+}= \text{neigh}_w$ (support positive neighbors)

- if $y = 0$ (safe) and `adapt_use_neg=True`: $v[\text{neigh\_idx}] \mathrel{-}= \alpha_{\text{neg}} \cdot \text{neigh}_w$ (penalize negative neighbors)

  

**2. Propagate votes to users:**

  

$$c_u = A_{\text{pos}} \cdot v - A_{\text{neg}} \cdot v \in \mathbb{R}^{n_{\text{users}}}$$

  

(user $u$ gets high score if they have many positive interactions with items that the test user marked as risky)

  

**3. Softmax attention:**

  

$$w_u = \text{softmax}(c_u / \tau_{\text{user}})$$

  

**4. Aggregate user embeddings:**

  

$$e_u^{\text{test}} = \sum_u w_u[u] \cdot E_u^{\text{train}}[u]$$

  

**파라미터**:

- `adapt_use_neg`: True (부정 샘플도 사용)

- `adapt_neg_weight`: 0.81 (α_neg)

- `adapt_user_softmax_temp`: 1.15 (τ_user)

  

**코드**: `src_new/model/copl/dataset.py:119-143`

  

### 5.2 수식 정리

  

전체 과정:

  

**Given:**

- Test samples: $\{(x_1, y_1), \ldots, (x_t, y_t)\}$

- Train embeddings: $E_u^{\text{train}}, E_i^{\text{train}}$

- Similarity encoder: $\Phi(\cdot)$

  

**Step 1 (Item Attachment):**

  

$$z_i = \Phi(x_i)$$

  

$$N_i = \text{TopK}_j \text{sim}(z_i, \Phi(x_j^{\text{train}}))$$

  

$$w_i = \text{softmax}([\text{sim}(z_i, \Phi(x_j^{\text{train}}))]_{j \in N_i})$$

  

$$e_i^{\text{test}} = \sum_{j \in N_i} w_i[j] E_j^{\text{train}}$$

  

**Step 2 (User Adaptation):**

  

$$v = \sum_i [y_i \cdot w_i \cdot I_{N_i} - \alpha_{\text{neg}} \cdot (1-y_i) \cdot w_i \cdot I_{N_i}]$$

  

$$c_u = (A_{\text{pos}} - A_{\text{neg}}) \cdot v$$

  

$$w_u = \text{softmax}(c_u / \tau)$$

  

$$e_u^{\text{test}} = \sum_u w_u[u] E_u^{\text{train}}$$

  

**Prediction:**

  

$$r = \sigma(\text{RM}(e_u^{\text{test}}, x))$$

  

**코드**: `src_new/model/copl/experiment.py:109-179`

  

---

  

## 6. 평가 (Evaluation)

  

### 6.1 Sequential AUROC

  

테스트 운전자의 context 크기를 점진적으로 증가시키며 성능 측정:

  

```

Test set: {(x_1, y_1), ..., (x_n, y_n)}

  

Split:

- Context: x_1, ..., x_{n/2}

- Holdout: x_{n/2+1}, ..., x_n

  

For t = 1, 2, ..., n/2:

1. Adapt user embedding using context[:t]

2. Predict on entire holdout set

3. Compute AUROC

```

  

**시각화**: Context size vs AUROC 그래프

  

**코드**: `src_new/model/copl/experiment.py:125-175`

  

### 6.2 메트릭

  

| 메트릭 | 수식 | 의미 |

|--------|------|------|

| **AUROC** | $\int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$ | ROC 곡선 아래 면적 |

| **AUPRC** | $\int_0^1 \text{Precision}(\text{Recall}) \, d(\text{Recall})$ | PR 곡선 아래 면적 |

| **Brier Score** | $\frac{1}{n} \sum (\hat{y} - y)^2$ | 예측 확률과 실제 라벨 차이 |

  

**코드**: `src_new/evaluation.py:evaluate_predictions`

  

### 6.3 시각화

  

생성되는 주요 plot:

  

1. **Sequential AUROC**: Context size별 성능 변화

2. **User Attention (w_u)**: 각 train user에 대한 attention weight

3. **Item Bridge**: 테스트 아이템이 어느 train user의 아이템과 연결되는지

4. **Embeddings**: t-SNE로 user/item 임베딩 시각화

5. **RM Distributions**: 각 user에 대한 RM 점수 분포

  

**코드**: `src_new/model/copl/visualization.py`

  

---

  

## 7. 실행 방법

  

### 7.1 기본 실행

  

```bash

python scripts_new/run_copl.py

```

  

**출력**:

- Artifacts: `artifacts/copl/YYYYMMDD_HHMMSS/`

- `best_gcf.pt`: GCF 모델 체크포인트

- `best_rm.pt`: RM 모델 체크포인트

- `metrics.txt`: 모든 메트릭 요약

- `plots/`: 시각화 결과

  

### 7.2 설정 변경

  

`scripts_new/run_copl.py`의 `Config` 클래스 수정:

  

```python

@dataclass

class Config:

# Data

train_driver_names: list = field(default_factory=lambda: [

"김진명", "조현석", "한규택", "박재일", "이지환"

])

test_driver: str = "강신길"

features: list = field(default_factory=lambda: [

"IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"

])

  

# GCF model

gcf_model: str = "gcf" # "gcf" | "gcf_cosine" | "gcf_softmax" | ...

gcf_emb_dim: int = 128

gcf_layers: int = 2

  

# Similarity

similarity_method: str = "vae" # "vae" | "dtw" | "pca"

  

# RM model

rm_model: str = "cnn" # "mlp" | "cnn" | "mole_cnn"

```

  

### 7.3 평가만 실행

  

이미 학습된 모델로 평가만:

  

```python

cfg = Config(timestamp="20250101_120000") # 기존 실험 폴더

```

  

---

  

## 8. 주요 설계 선택

  

### 8.1 왜 Positive/Negative 그래프를 분리?

  

일반적인 collaborative filtering은 positive feedback만 사용하지만, CoPL은 explicit negative feedback도 활용합니다:

  

- **Positive graph**: "이 운전자는 이 상황을 위험하다고 봤다"

- **Negative graph**: "이 운전자는 이 상황을 안전하다고 봤다"

  

분리 이유:

1. **대조 학습**: Positive와 negative 이웃의 영향을 독립적으로 모델링

2. **선호 패턴**: 한 운전자가 안전하다고 본 상황을 다른 운전자는 위험하다고 볼 수 있음

3. **Adaptation**: 테스트 시 positive/negative 샘플을 모두 활용

  

### 8.2 왜 Item-Item 그래프?

  

User-item 그래프만으로는 cold-start 문제가 심각합니다:

- 새로운 user는 이웃이 전혀 없음

- Item-item 유사도를 통해 **내용 기반(content-based)** 정보 보충

  

수식적으로:

- **Without** $A_{ii}$: item embedding은 오직 user interaction으로만 학습

- **With** $A_{ii}$: similar items끼리 embedding이 가까워짐

  

이를 통해 테스트 시 k-NN으로 새로운 아이템을 기존 아이템에 연결 가능.

  

### 8.3 왜 Straight-through Estimator (MoLE)?

  

MoLE에서 discrete expert selection의 gradient 문제:

  

$$\text{hard}_w = \text{one\_hot}(\arg\max(\text{soft}_w)) \quad \to \text{ not differentiable}$$

  

**해결: Straight-through estimator**

- forward: use $\text{hard}_w$ (discrete)

- backward: use $\text{soft}_w$ (continuous gradient)

  

**구현:**

```python

routing_w = hard_w - soft_w.detach() + soft_w

```

  

---

  

## 9. 관련 파일

  

### 9.1 모델 구현

  

| 파일 | 내용 |

|------|------|

| `src_new/model/copl/gcf.py` | CoPLGCF 및 변형들 |

| `src_new/model/copl/gcf_gcn.py` | GCN 버전 (미사용) |

| `src_new/model/copl/rm.py` | Reward Model (MLP/CNN/MoLE) |

| `src_new/model/copl/dataset.py` | Graph dataset 구성 |

| `src_new/model/copl/similarity.py` | VAE/DTW/PCA similarity |

| `src_new/model/copl/trainer.py` | GCF/RM Trainer |

| `src_new/model/copl/experiment.py` | 전체 파이프라인 |

  

### 9.2 실행 스크립트

  

| 파일 | 내용 |

|------|------|

| `scripts_new/run_copl.py` | 메인 실행 스크립트 |

  

### 9.3 참고 문서

  

| 파일 | 내용 |

|------|------|

| `docs/CoPL/README.md` | 원본 CoPL 논문 구현 설명 |

| `docs/CoPL/CoPL_Paper.pdf` | EMNLP 2025 논문 원문 |

  

---

  

## 10. 수식 요약

  

### 10.1 GCF Forward Pass

  

**Initialize:**

  

$$E_u^{(0)}, E_i^{(0)} \sim \text{Xavier}$$

  

**For** $l = 1, \ldots, L$:

  

*User message passing:*

  

$$Z_u^+ = \bar{A}_{\text{pos}} E_i^{(l-1)}$$

  

$$Z_u^- = \bar{A}_{\text{neg}} E_i^{(l-1)}$$

  

$$m_u = W_u^0 E_u^{(l-1)} + W_u^1 Z_u^+ + W_u^2 (Z_u^+ \odot E_u^{(l-1)}) + W_u^3 Z_u^- + W_u^4 (Z_u^- \odot E_u^{(l-1)})$$

  

$$E_u^{(l)} = \text{LeakyReLU}(m_u)$$

  

*Item message passing:*

  

$$Z_i^+ = \bar{A}_{\text{pos}}^T E_u^{(l-1)}$$

  

$$Z_i^- = \bar{A}_{\text{neg}}^T E_u^{(l-1)}$$

  

$$Z_i^{ii} = \bar{A}_{ii} E_i^{(l-1)}$$

  

$$m_i = W_i^0 E_i^{(l-1)} + W_i^1 Z_i^+ + W_i^2 (Z_i^+ \odot E_i^{(l-1)}) + W_i^3 Z_i^- + W_i^4 (Z_i^- \odot E_i^{(l-1)}) + \lambda_{ii} (W_i^5 Z_i^{ii} + W_i^6 (Z_i^{ii} \odot E_i^{(l-1)}))$$

  

$$E_i^{(l)} = \text{LeakyReLU}(m_i)$$

  

**Output:**

  

$$E_u = \text{normalize}(E_u^{(L)})$$

  

$$E_i = E_i^{(L)}$$

  

### 10.2 Reward Model Forward Pass

  

$$h = \text{LeakyReLU}(W_{\text{obs}} x + W_u e_u) \quad \text{(fusion)}$$

  

$$h = \text{CNN}(h) \quad \text{(temporal modeling)}$$

  

$$h = \text{MaxPool}(h) \quad \text{(aggregation)}$$

  

$$r = \sigma(\text{MLP}(h)) \quad \text{(prediction)}$$

  

### 10.3 Test-time Adaptation

  

**Item attachment:**

  

$$z_t = \Phi(x_{\text{test}})$$

  

$$\{j_1, \ldots, j_k\} = \text{TopK}_j \, \text{sim}(z_t, z_{\text{train}}[j])$$

  

$$w = \text{softmax}([\text{sim}(z_t, z_{\text{train}}[j_i])]_{i=1 \ldots k})$$

  

$$e_i^{\text{test}} = \sum_{i=1}^k w_i E_i^{\text{train}}[j_i]$$

  

**User adaptation:**

  

$$v = \sum_{t=1}^T [y_t w_t I_{\text{neighbors}(t)} - \alpha_{\text{neg}} (1-y_t) w_t I_{\text{neighbors}(t)}]$$

  

$$c_u = (A_{\text{pos}} - A_{\text{neg}}) v$$

  

$$w_u = \text{softmax}(c_u / \tau)$$

  

$$e_u^{\text{test}} = \sum_u w_u[u] E_u^{\text{train}}[u]$$

  

---

  

## 참고문헌

  

- **CoPL Paper**: Choi et al., "CoPL: Collaborative Preference Learning for Personalizing LLMs", EMNLP 2025

- **Graph Collaborative Filtering**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020

- **MoLE**: Mixture of Cluster-conditional LoRA Experts for Vision-language Instruction Tuning