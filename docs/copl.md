# CoPL Model Methodology

This document details the methodology and mathematical formulation for the `CoPLGCF` and `CoPLGCF_GCN` models implemented in `scripts/train_copl.py`.

## 1. Problem Definition & Graph Construction

The problem is framed as a link prediction task on a bipartite user-item graph, augmented with an item-item similarity graph.

### Graphs
Let $U$ be the set of users and $I$ be the set of items.
We construct three adjacency matrices:
1.  **Positive User-Item Graph ($A_{pos}$)**:
    $A_{pos} \in \{0, 1\}^{|U| \times |I|}$. $A_{pos}[u, i] = 1$ if user $u$ has a positive interaction with item $i$, else 0.
    Normalized as $D_{u}^{-1/2} A D_{i}^{-1/2}$.

2.  **Negative User-Item Graph ($A_{neg}$)**:
    $A_{neg} \in \{0, 1\}^{|U| \times |I|}$. $A_{neg}[u, i] = 1$ if user $u$ has a negative interaction with item $i$, else 0.
    Normalized similarly.

3.  **Item-Item Similarity Graph ($A_{ii}$)**:
    Constructed from item features $X \in \mathbb{R}^{|I| \times T \times D}$.
    -   Features are flattened and standardized.
    -   PCA is applied to reduce dimensionality to $Z \in \mathbb{R}^{|I| \times d'}$.
    -   A k-Nearest Neighbor (kNN) graph is built based on Euclidean distance in $Z$.
    -   Edge weights are computed using a Gaussian kernel: $w_{ij} = \exp(-\gamma ||z_i - z_j||^2)$.
    -   $A_{ii}$ is row-normalized.

---

## 2. CoPLGCF (Collaborative Pattern Learning - Graph Collaborative Filtering)

`CoPLGCF` uses a multi-channel message passing scheme where different edge types (self, positive, negative, item-item) have separate learnable transformation matrices.

### Initialization
-   User embeddings: $E_u^{(0)} \in \mathbb{R}^{|U| \times d}$
-   Item embeddings: $E_i^{(0)} \in \mathbb{R}^{|I| \times d}$

### Message Passing (Layer $l$)
For each layer $l \in \{1, \dots, L\}$, we update user and item embeddings.

#### User Update
For a user node $u$, the message $m_u^{(l)}$ aggregates information from:
1.  **Self-loop**: $W_{u,self}^{(l)} E_u^{(l-1)}$
2.  **Positive neighbors ($Z_{u,pos} = A_{pos} E_i^{(l-1)}$)**:
    -   Linear: $W_{u,pos1}^{(l)} Z_{u,pos}$
    -   Interaction: $W_{u,pos2}^{(l)} (Z_{u,pos} \odot E_u^{(l-1)})$
3.  **Negative neighbors ($Z_{u,neg} = A_{neg} E_i^{(l-1)}$)**:
    -   Linear: $W_{u,neg3}^{(l)} Z_{u,neg}$
    -   Interaction: $W_{u,neg4}^{(l)} (Z_{u,neg} \odot E_u^{(l-1)})$

$$
m_u^{(l)} = W_{u,self}^{(l)} E_u^{(l-1)} + W_{u,pos1}^{(l)} Z_{u,pos} + W_{u,pos2}^{(l)} (Z_{u,pos} \odot E_u^{(l-1)}) \\
+ W_{u,neg3}^{(l)} Z_{u,neg} + W_{u,neg4}^{(l)} (Z_{u,neg} \odot E_u^{(l-1)})
$$

#### Item Update
For an item node $i$, the message $m_i^{(l)}$ aggregates information from:
1.  **Self-loop**: $W_{i,self}^{(l)} E_i^{(l-1)}$
2.  **Positive neighbors ($Z_{i,pos} = A_{pos}^T E_u^{(l-1)}$)**:
    -   Linear: $W_{i,pos1}^{(l)} Z_{i,pos}$
    -   Interaction: $W_{i,pos2}^{(l)} (Z_{i,pos} \odot E_i^{(l-1)})$
3.  **Negative neighbors ($Z_{i,neg} = A_{neg}^T E_u^{(l-1)}$)**:
    -   Linear: $W_{i,neg3}^{(l)} Z_{i,neg}$
    -   Interaction: $W_{i,neg4}^{(l)} (Z_{i,neg} \odot E_i^{(l-1)})$
4.  **Item-Item Similarity Neighbors ($Z_{i,ii} = A_{ii} E_i^{(l-1)}$)**:
    -   Weighted by scalar $\lambda_{ii}$.
    -   Terms: $\lambda_{ii} [ W_{i,ii1}^{(l)} Z_{i,ii} + W_{i,ii2}^{(l)} (Z_{i,ii} \odot E_i^{(l-1)}) ]$

$$
m_i^{(l)} = W_{i,self}^{(l)} E_i^{(l-1)} + \dots (\text{pos/neg from users}) \dots + \lambda_{ii} (\dots \text{item-item terms} \dots)
$$

The final embedding for layer $l$ is obtained via LeakyReLU activation:
$$E_u^{(l)} = \text{LeakyReLU}(m_u^{(l)})$$
$$E_i^{(l)} = \text{LeakyReLU}(m_i^{(l)})$$

Final user embeddings are L2-normalized: $E_u = \text{normalize}(E_u^{(L)})$.

---

## 3. CoPLGCF_GCN (Simplified GCN Variant)

`CoPLGCF_GCN` simplifies the architecture by using a standard GCN formulation. Instead of separate weights for each edge type, it constructs a **unified signed graph** and applies a single shared weight matrix per layer.

### Message Aggregation
The aggregation step simulates a signed adjacency matrix operation:
$$ A_{total} \approx A_{pos} - A_{neg} $$

#### User Aggregation
$$ msg_u^{(l)} = A_{pos} E_i^{(l-1)} - A_{neg} E_i^{(l-1)} $$
(Aggregates positive item signal and subtracts negative item signal).

#### Item Aggregation
$$ msg_i^{(l)} = (A_{pos}^T E_u^{(l-1)} - A_{neg}^T E_u^{(l-1)}) + \lambda_{ii} A_{ii} E_i^{(l-1)} $$
(Aggregates user signals and adds item-item similarity signal).

### Update Rule (Residual GCN)
A single linear layer $W^{(l)}$ (with bias) is used. A residual connection from the previous embedding is added (acting effectively as a self-loop).

$$ E_u^{(l)} = \text{LeakyReLU}\left( (msg_u^{(l)} + E_u^{(l-1)}) W^{(l)} + b^{(l)} \right) $$
$$ E_i^{(l)} = \text{LeakyReLU}\left( (msg_i^{(l)} + E_i^{(l-1)}) W^{(l)} + b^{(l)} \right) $$

Final user embeddings are L2-normalized.

---

## 4. Optimization

Both models share the same training objective.

### Pointwise Prediction
The score for a user-item pair $(u, i)$ is the dot product of their final embeddings:
$$ \hat{y}_{ui} = \sigma(e_u \cdot e_i) $$
where $\sigma$ is the sigmoid function.

### Loss Function
The model is trained using **Weighted Binary Cross Entropy (BCE)** to handle class imbalance, plus L2 regularization on embeddings.

$$ \mathcal{L} = -\frac{1}{N} \sum_{(u,i) \in \mathcal{D}} \left[ w_{pos} y_{ui} \log(\hat{y}_{ui}) + (1-y_{ui}) \log(1-\hat{y}_{ui}) \right] + \lambda_{reg} (||e_u||^2 + ||e_i||^2) $$

where $w_{pos}$ is a weight assigned to positive samples (typically $N_{neg} / N_{pos}$).

---

## 5. Time-Series Reward Model (TSRewardModel)

After training the CoPL model to learn user embeddings ($E_u$) and item embeddings ($E_i$), a separate Reward Model is trained to predict user preferences based on *only* the observation sequence $X_{obs}$.

The Reward Model $f_\theta$ takes as input:
1.  **Fixed User Embedding**: $e_u \in \mathbb{R}^{d_{emb}}$ (from the pre-trained CoPL model).
2.  **Observation Sequence**: $X_{obs} \in \mathbb{R}^{T \times d_{obs}}$.

### Architecture
The model projects both inputs into a common hidden space, combines them, temporal-pools, and then passes through an MLP.

1.  **Feature Projection**:
    $$ h_{obs} = X_{obs} W_{obs} + b_{obs} \quad \in \mathbb{R}^{T \times H} $$
    $$ h_u = e_u W_u + b_u \quad \in \mathbb{R}^{H} $$

2.  **Combination & Non-Linearity**:
    The user embedding is broadcasted across time steps:
    $$ h_{combined}^{(t)} = \tanh( h_{obs}^{(t)} + h_u ) \quad \text{for } t=1 \dots T $$

3.  **Temporal Pooling**:
    Mean pooling over the time dimension:
    $$ h_{pool} = \frac{1}{T} \sum_{t=1}^T h_{combined}^{(t)} \quad \in \mathbb{R}^{H} $$

4.  **MLP & Output**:
    $$ h_{mlp} = \text{ReLU}( h_{pool} W_1 + b_1 ) $$
    $$ h_{final} = \text{ReLU}( h_{mlp} W_2 + b_2 ) $$
    $$ \hat{y} = \sigma( h_{final} W_{head} + b_{head} ) $$

### Training Goal
The Reward Model is trained to classify whether a given observation sequence $X_{obs}$ corresponds to a positive interaction for user $u$.
It uses the same **Weighted Binary Cross Entropy** loss as the CoPL model, but the user embeddings $e_u$ are fixed (frozen) during this phase.

$$ \mathcal{L}_{RM} = \text{WeightedBCE}(\hat{y}, y_{target}) + \lambda_{reg} ||e_u||^2 $$
*(Note: Small regularization on the input user embedding is optional but included in implementation).*

---

## 6. 데이터 전처리 파이프라인

CoPL 학습에 사용되는 데이터 전처리는 크게 **(1) 유틸리티 함수**, **(2) Item-Item 유사도 그래프 구축**, **(3) 테스트 아이템/유저 Adaptation** 세 단계로 구성됩니다.

### 6.1 유틸리티 함수

#### 표준화 (Standardization)

시계열 특징 벡터의 스케일을 통일하기 위해 Z-score 표준화를 적용합니다.

$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i, \quad \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2} + \epsilon$$

$$\tilde{x}_i = \frac{x_i - \mu}{\sigma}$$

여기서 $\epsilon = 10^{-6}$은 0-분산 방지를 위한 안정 상수입니다. `standardize_fit`은 학습 데이터에서 $\mu, \sigma$를 계산하고, `standardize_apply`는 이를 적용합니다. 테스트 데이터에도 **학습 데이터의 $\mu, \sigma$를 그대로 사용**하여 데이터 누수(data leakage)를 방지합니다.

#### 이분 그래프 인접행렬 정규화 (Bipartite Adjacency Normalization)

유저-아이템 이분 그래프 $A \in \mathbb{R}^{|U| \times |I|}$를 대칭 정규화합니다:

$$\hat{A} = D_u^{-1/2} \, A \, D_i^{-1/2}$$

여기서 $D_u = \text{diag}(\sum_j A_{uj})$는 유저 차수 행렬, $D_i = \text{diag}(\sum_u A_{ui})$는 아이템 차수 행렬입니다. 각 엣지 $(u, i)$에 대해:

$$\hat{A}_{ui} = \frac{A_{ui}}{\sqrt{d_u \cdot d_i + \epsilon}}$$

이 정규화는 차수(degree)가 높은 노드의 영향력이 과도하게 커지는 것을 방지합니다.

#### 정방 인접행렬 정규화 (Square Adjacency Normalization)

Item-Item 그래프 $A_{ii} \in \mathbb{R}^{|I| \times |I|}$에 대해 동일한 형태의 대칭 정규화를 적용합니다:

$$\hat{A}_{ii} = D^{-1/2} \, A_{ii} \, D^{-1/2}$$

$$(\hat{A}_{ii})_{jk} = \frac{(A_{ii})_{jk}}{\sqrt{d_j \cdot d_k + \epsilon}}$$

#### Median Heuristic Gamma

RBF(Gaussian) 커널의 대역폭 파라미터 $\gamma$를 데이터 기반으로 자동 결정합니다. **Median Heuristic**은 쌍별 유클리드 거리(pairwise Euclidean distance)의 중앙값 $\tilde{d}$를 사용합니다:

$$\gamma_{med} = \frac{1}{2 \tilde{d}^2 + \epsilon}$$

$N \leq 2000$이면 모든 쌍을 계산하고, $N > 2000$이면 랜덤 샘플링으로 최대 200,000개 쌍의 거리를 추정합니다. 최종 $\gamma$는 사용자 설정 배율 `gamma_mul`을 곱하여 결정됩니다:

$$\gamma = \gamma_{med} \times \texttt{gamma\_mul}$$

---

### 6.2 Item-Item 유사도 그래프 구축 (Similarity Strategy)

학습 아이템들 간의 시계열 유사도 그래프 $A_{ii}$ 구축은 **Strategy 패턴**을 통해 유연하게 선택할 수 있습니다 (`cfg.similarity_method`).

**공통 목표**: 아이템 시계열 $X \in \mathbb{R}^{N \times T \times D}$를 저차원 잠재 공간 $Z \in \mathbb{R}^{N \times d'}$로 변환하고, 그 공간에서 kNN 그래프를 구축합니다.

#### 6.2.1 Linear PCA (`pca`) - 기본값
기존의 선형 차원 축소 방식입니다.
1.  **Flatten + 표준화**: $X_{flat} \in \mathbb{R}^{N \times (T \cdot D)}$로 펼치고 Z-score 정규화.
2.  **PCA**: 선형 주성분 분석으로 $Z$ 추출.
3.  **kNN + RBF**: $Z$ 공간에서 유클리드 거리 기반 kNN 그래프 구축. $\gamma$는 Median Heuristic으로 자동 결정.

#### 6.2.2 RBF Kernel PCA (`kernel_pca`)
시계열 데이터의 비선형 매니폴드 구조를 포착하기 위해 커널 방식을 사용합니다.
1.  **커널 대역폭 결정**: 원본 공간 $X_{flat}$에서 Median Heuristic으로 RBF 커널의 $\gamma_{ker}$를 계산.
2.  **Kernel PCA**: RBF 커널을 사용하여 고차원 특징 공간으로 매핑 후 PCA 수행.
3.  **kNN + RBF**: 투영된 $Z$ 공간에서 다시 kNN 그래프 구축.

#### 6.2.3 1D-CNN VAE (`vae`)
딥러닝 기반의 생성 모델을 사용하여 비선형 패턴을 학습합니다.
1.  **Model**: 1D-CNN Encoder + Decoder 구조의 Variational Autoencoder.
2.  **Training**: Reconstruction Loss (MSE) + KL Divergence로 학습 ($\mathcal{L} = \mathcal{L}_{recon} + \beta \mathcal{L}_{KL}$).
3.  **Latent Representation**: 학습된 Encoder가 출력하는 평균 벡터 $\mu$를 아이템의 잠재 표현 $Z$로 사용.
4.  **kNN + RBF**: $Z$ 공간에서 kNN 그래프 구축.

---

### 6.3 테스트 아이템 임베딩 생성 (`attach_test_items`)

학습 과정에서 관찰하지 못한 **테스트 아이템**의 임베딩을 생성합니다. Inductive 추론을 위해, 선택된 **Similarity Strategy**에 따라 테스트 아이템을 잠재 공간으로 투영합니다.

#### 절차

1.  **전처리**: 테스트 아이템 시계열을 **학습 데이터의 $\mu, \sigma$로** 표준화.
2.  **잠재 공간 투영 ($Z_{test}$)**:
    -   `pca`: 학습된 PCA 행렬로 선형 변환.
    -   `kernel_pca`: 학습 데이터와의 커널 내적을 통해 투영.
    -   `vae`: 학습된 Encoder를 통과하여 $\mu$ 추출.
3.  **이웃 탐색**: $Z$ 공간에서 **top-$k$ 학습 아이템 이웃** 탐색.
4.  **가중치 계산**: RBF 커널로 유사도 가중치 계산 (Softmax 정규화).
    $$w_{tj} = \frac{\exp(-\gamma \|z_t - z_j\|^2)}{\sum_{j' \in \text{top-}k} \exp(-\gamma \|z_t - z_{j'}\|^2)}$$
5.  **임베딩 생성**: 학습 아이템 임베딩의 가중 평균.
    $$e_{i}^{test} = \sum_{j \in \text{top-}k} w_{tj} \cdot e_{j}^{train}$$

---

### 6.4 테스트 유저 임베딩 생성 (`adapt_test_user_embedding`)

학습 과정에서 관찰하지 못한 **테스트 유저**의 임베딩을 **Item-Item Bridge** 방식으로 생성합니다.

#### 핵심 아이디어

테스트 유저의 선호(positive/negative 아이템)가 학습 아이템들과 얼마나 유사한지를 측정하고, 이를 기반으로 **학습 유저들의 임베딩을 가중 결합**합니다.

#### Step 1: 아이템별 기여도(Vote) 계산

테스트 유저의 각 아이템이 학습 아이템 $j$에 대해 기여하는 점수를 누적합니다:

$$v_j = \sum_{t \in \text{pos}(test)} w_{tj} - \eta \sum_{t \in \text{neg}(test)} w_{tj}$$

여기서 $w_{tj}$는 `attach_test_items_to_train`에서 계산된 이웃 가중치이고, $\eta$ (`adapt_neg_weight`)는 부정 기여의 강도를 조절합니다.

- **Positive 아이템**: 학습 아이템 $j$와 유사하면 $v_j$가 증가 → 해당 아이템을 좋아하는 학습 유저에게 유리
- **Negative 아이템**: 학습 아이템 $j$와 유사하면 $v_j$가 감소 → 해당 아이템을 좋아하는 학습 유저에게 불리

#### Step 2: 학습 유저별 점수 계산

학습 유저 $u$의 점수는, 해당 유저가 positive하게 평가한 학습 아이템들의 $v_j$ 합산입니다:

$$c_u = \sum_{j \in \mathcal{N}_u^+} v_j - \sum_{j \in \mathcal{N}_u^-} v_j$$

여기서 $\mathcal{N}_u^+, \mathcal{N}_u^-$는 학습 유저 $u$의 positive/negative 아이템 집합입니다. 이 연산은 행렬 곱으로 효율적으로 수행됩니다: $c = A_{pos} \cdot v - A_{neg} \cdot v$.

#### Step 3: Softmax 가중 결합

학습 유저 점수를 softmax로 정규화하여 테스트 유저 임베딩을 생성합니다:

$$w_u = \text{softmax}\left(\frac{c_u}{\tau}\right), \quad e_u^{test} = \sum_{u} w_u \cdot e_u^{train}$$

여기서 $\tau$ (`adapt_user_softmax_temp`)는 온도 파라미터입니다:
- $\tau \to 0$: 가장 유사한 **한 명의 유저**에 집중 (hard assignment)
- $\tau \to \infty$: 모든 유저에게 **균등 가중치** (uniform average)

