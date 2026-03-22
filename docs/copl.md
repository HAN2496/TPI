# CoPL GCF 방법론 정리 (gcf.py 기준)

이 문서는 CoPL의 그래프 협업 필터링 모듈을 [src_new/model/copl/gcf.py](src_new/model/copl/gcf.py) 구현 기준으로 수식화해 설명한다. 특히 `CoPLGCF` 본체와 변형(`CoPLGCFCosine`, `CoPLGCFPointwiseBPR`, `CoPLGCFSoftmax`, `CoPLGCFMargin`)의 차이를 학습 목적함수 관점에서 정리한다.

## 1. 문제 설정

학습 대상은 사용자-아이템 상호작용 그래프이며, 상호작용 라벨은 이진값이다.

- 사용자 집합: $\mathcal{U}$, 크기 $n_u$
- 아이템 집합: $\mathcal{I}$, 크기 $n_i$
- 임베딩 차원: $d$
- 학습 라벨: $y_{ui} \in \{0,1\}$

코드에서는 양/음성 상호작용을 분리된 이분 그래프로 다룬다.

- 양성 인접행렬(정규화): $\tilde{A}^{+} \in \mathbb{R}^{n_u \times n_i}$
- 음성 인접행렬(정규화): $\tilde{A}^{-} \in \mathbb{R}^{n_u \times n_i}$
- 선택적 아이템-아이템 인접행렬(정규화): $\tilde{A}^{ii} \in \mathbb{R}^{n_i \times n_i}$

초기 임베딩은 학습 파라미터다.

$$
E_u^{(0)} \in \mathbb{R}^{n_u \times d}, \quad E_i^{(0)} \in \mathbb{R}^{n_i \times d}
$$

## 2. Sparse Dropout

각 레이어에서 그래프 메시지 전달 전에 sparse edge dropout을 적용한다.

- edge 값 벡터 $v$에 dropout 확률 $p$ 적용
- 테스트 시(`test=True`) dropout 비활성화

수식적으로는 다음과 같이 볼 수 있다.

$$
\hat{A} = \text{DropEdge}(\tilde{A}, p)
$$

구현 위치: `sparse_dropout`.

## 3. CoPLGCF 레이어 업데이트

레이어 $\ell \in \{1,\dots,L\}$에서 이전 임베딩을 다음처럼 둔다.

$$
E_u^{\ell-1}, \; E_i^{\ell-1}
$$

### 3.1 사용자/아이템 메시지 집계

양/음성 그래프를 통해 집계 메시지를 만든다.

$$
Z_u^{+} = \hat{A}^{+} E_i^{\ell-1}, \qquad
Z_u^{-} = \hat{A}^{-} E_i^{\ell-1}
$$

$$
Z_i^{+} = (\hat{A}^{+})^\top E_u^{\ell-1}, \qquad
Z_i^{-} = (\hat{A}^{-})^\top E_u^{\ell-1}
$$

### 3.2 고차 상호작용(원소곱) 포함 선형 결합

`CoPLGCF`의 핵심은 self/positive/negative + 원소곱 항을 분리 파라미터로 합성하는 것이다.

사용자 업데이트 전 메시지:

$$
M_u^{\ell} =
W_{u,self}^{\ell} E_u^{\ell-1}
+ W_{u,p1}^{\ell} Z_u^{+}
+ W_{u,p2}^{\ell} (Z_u^{+} \odot E_u^{\ell-1})
+ W_{u,n3}^{\ell} Z_u^{-}
+ W_{u,n4}^{\ell} (Z_u^{-} \odot E_u^{\ell-1})
$$

아이템 업데이트 전 메시지:

$$
M_i^{\ell} =
W_{i,self}^{\ell} E_i^{\ell-1}
+ W_{i,p1}^{\ell} Z_i^{+}
+ W_{i,p2}^{\ell} (Z_i^{+} \odot E_i^{\ell-1})
+ W_{i,n3}^{\ell} Z_i^{-}
+ W_{i,n4}^{\ell} (Z_i^{-} \odot E_i^{\ell-1})
$$

여기서 $\odot$는 원소별 곱이다.

### 3.3 아이템-아이템 그래프 보강(선택)

$\tilde{A}^{ii}$가 있으면 추가 메시지를 더한다.

$$
Z_{ii} = \hat{A}^{ii} E_i^{\ell-1}
$$

$$
M_i^{\ell} \leftarrow M_i^{\ell} + \alpha_{ii}
\left(
W_{ii,1}^{\ell} Z_{ii} + W_{ii,2}^{\ell} (Z_{ii} \odot E_i^{\ell-1})
\right)
$$

여기서 $\alpha_{ii}$는 코드의 `item_item_weight`다.

### 3.4 활성화 및 다음 레이어

LeakyReLU($0.2$)를 적용한다.

$$
E_u^{\ell} = \phi(M_u^{\ell}), \qquad E_i^{\ell} = \phi(M_i^{\ell})
$$

최종 레이어 이후 출력은 다음과 같다.

- 사용자 임베딩은 L2 정규화:

$$
E_u = \text{norm}(E_u^{L})
$$

- 아이템 임베딩은 정규화 없이 그대로 사용:

$$
E_i = E_i^{L}
$$

## 4. 점수 함수

기본 pointwise 점수는 내적이다.

$$
s_{ui} = \langle e_u, e_i \rangle
$$

코드에서는 logits로 사용한다.

## 5. 기본 손실: Weighted BCE + L2

라벨 $y\in\{0,1\}$, logit $s$일 때 BCE는

$$
\ell_{bce}(s,y) = -y\log\sigma(s) - (1-y)\log(1-\sigma(s))
$$

`pos_weight`가 있으면 양성 항 가중치를 높인다. 샘플 가중치(`sample_weight`)도 곱할 수 있다.

정규화 항:

$$
\ell_{reg} = \frac{1}{B} \sum_{b=1}^{B} \left(\|e_u^{(b)}\|_2^2 + \|e_i^{(b)}\|_2^2\right)
$$

최종:

$$
\mathcal{L} = \frac{1}{B} \sum_{b=1}^{B} \ell_{bce}(s_b, y_b) + \lambda \ell_{reg}
$$

## 6. 변형 모델들

### 6.1 CoPLGCFCosine

- 출력 임베딩을 사용자/아이템 모두 L2 정규화
- 손실은 CosineEmbeddingLoss
- 타깃 변환: $y=1 \to +1$, $y=0 \to -1$

수식:

$$
\mathcal{L}_{cos} = \frac{1}{B} \sum_{b=1}^{B} \ell_{cos}(e_u^{(b)}, e_i^{(b)}, t_b)
$$

여기서 $t_b \in \{-1,+1\}$.

### 6.2 CoPLGCFPointwiseBPR

pointwise 형태의 로그시그모이드 목적을 쓴다.

$$
\ell_b = -\left(y_b \log \sigma(s_b) + (1-y_b)\log\sigma(-s_b)\right)
$$

`pos_weight`와 `sample_weight`를 반영할 수 있고, 마지막에 $\lambda \ell_{reg}$를 더한다.

### 6.3 CoPLGCFSoftmax

양성 샘플만 모아 in-batch 대조 형태 softmax CE를 계산한다.

- 양성 인덱스 집합: $\mathcal{P}$
- 유사도 행렬:

$$
S_{ab} = \frac{\langle e_u^{(a)}, e_i^{(b)}\rangle}{\tau}, \quad a,b\in\mathcal{P}
$$

- 정답은 대각 원소(같은 샘플 쌍)

$$
\mathcal{L}_{softmax} = \frac{1}{|\mathcal{P}|}\sum_{a\in\mathcal{P}} \text{CE}(S_{a,:}, a)
$$

양성 샘플이 없으면 기본 BCE 경로로 폴백한다.

### 6.4 CoPLGCFMargin

양성 점수와 음성 점수의 쌍대 차이를 직접 밀어낸다.

- $s^+ \in \mathcal{S}^+$, $s^- \in \mathcal{S}^-$
- 차이: $\Delta = s^+ - s^-$

$$
\mathcal{L}_{margin} = \text{mean}\left(-\log\sigma(\Delta) \cdot w_{bal}\right) + \lambda \ell_{reg}
$$

여기서 $w_{bal}=|\mathcal{S}^-|/(|\mathcal{S}^+|+\epsilon)$로 클래스 불균형을 보정한다.

## 7. gcf.py의 설계 포인트

1. 양/음성 상호작용을 분리 메시지로 다뤄 단순 sign 반전보다 표현력이 높다.
2. $Z \odot E$ 항을 넣어 2차 상호작용을 선형층으로 학습한다.
3. item-item 그래프를 별도 경로로 주입해 시계열/유사도 기반 이웃 정보를 반영한다.
4. 사용자 임베딩만 기본 정규화하고 아이템 임베딩은 자유도를 남긴다.
5. 손실 함수 변형을 통해 데이터 특성(불균형, 순위학습, 대조학습)에 맞게 실험할 수 있다.

## 8. gcf_gcn.py와의 방법론적 차이 요약

- `gcf.py`: self/pos/neg/interaction별 다중 선형층을 쓰는 고표현력 모델
- `gcf_gcn.py`: 레이어당 단일 선형층과 단순 메시지($A^+E - A^-E$)를 쓰는 경량 모델

즉, `gcf.py`는 파라미터 수와 결합 항이 많아 더 복잡한 사용자-아이템 관계를 표현하도록 설계되어 있다.

## 9. 구현 매핑

- 그래프 인코딩: `encode_graph`
- 기본 분류 손실: `forward_pointwise` + `weighted_bce_with_logits`
- 변형 손실: `CoPLGCFCosine`, `CoPLGCFPointwiseBPR`, `CoPLGCFSoftmax`, `CoPLGCFMargin`
- sparse dropout: `sparse_dropout`

참조 코드: [src_new/model/copl/gcf.py](src_new/model/copl/gcf.py)
