# Prototype Mixture Bayesian Additive Reward Model

## 목표

주행 trajectory $\xi_{u,i}$와 사용자 피드백 $y_{u,i} \in \{0,1\}$가 있을 때,
사용자 $u$의 내면 보상함수 $R_u$를 추정한다.

$$
P(y_{u,i}=1 \mid \xi_{u,i}, u) = \sigma(R_u(\xi_{u,i}))
$$

여기서 $y=1$은 `Good/Positive`, $y=0$은 `Bad/Negative` 피드백을 뜻한다.

---

## Feature Map

원본 주행 시계열은 다음과 같다.

$$
\xi_i \in \mathbb{R}^{T \times D}
$$

전체 구간과 $Q$개의 sub-window를 함께 사용한다.

$$
\xi_i
\rightarrow
\{\xi_i^{(0)}, \xi_i^{(1)}, \ldots, \xi_i^{(Q)}\}
$$

$\xi_i^{(0)}$는 전체 구간, $\xi_i^{(q)}$는 $q$번째 sub-window이다.
각 구간을 물리적으로 해석 가능한 요약 feature로 변환한다.

$$
z_i
=
[
g(\xi_i^{(0)}),
g(\xi_i^{(1)}),
\ldots,
g(\xi_i^{(Q)})
]
\in \mathbb{R}^{K}
$$

예시는 다음과 같다.

$$
z_i =
[
\operatorname{rms}(a_{\text{long}}),
\max |a_{\text{long}}|,
\operatorname{rms}(\dot{a}_{\text{long}}),
\max |\omega_{\text{pitch}}|,
\operatorname{energy}(\omega_{\text{bounce}}),
\operatorname{rms}(a_{\text{lat}}),
\operatorname{rms}(\omega_{\text{yaw}}),
\operatorname{rms}(\delta_{\text{steer}}),
\operatorname{mean}(p_{\text{accel}}),
\operatorname{mean}(F_{\text{brake}}),
\ldots
]
$$

그 다음 hard rule 대신 soft additive basis를 만든다.

$$
\phi(z_i) =
[
1,\,
z_{i,k},\,
(z_{i,k}-c_{k,m})_+,\,
(c_{k,m}-z_{i,k})_+
]_{k,m}
$$

$$
(a)_+ = \max(a,0)
$$

따라서 특정 threshold를 고정하지 않고, 부드러운 구간별 민감도를 학습할 수 있다.
현재 기본 실행값은 controller activation 이후 첫 구간인 $[5,6]$초와 $Q=1$을 사용한다.
sub-window와 hinge basis는 옵션으로 남겨둔다.

---

## 사용자 보상함수

사용자별 reward는 additive basis 위의 선형 함수로 둔다.

$$
R_u(\xi_i) = \theta_u^\top \phi(g(\xi_i))
$$

$$
P(y_{u,i}=1 \mid \xi_i,u)
=
\sigma(\theta_u^\top \phi_i)
$$

여기서

$$
\phi_i = \phi(g(\xi_i))
$$

이다.

---

## 사용자별 Prototype Prior

기존 사용자들의 피드백을 하나의 평균 prior로만 압축하지 않는다.
먼저 전체 사용자 데이터로 약한 population anchor를 만든 뒤, 각 기존 사용자 $v$의 선호도 profile을 따로 추정한다.

$$
\theta_v
=
\arg\min_\theta
\sum_i
\ell(y_{v,i},\sigma(\theta^\top \phi_{v,i}))
+
\frac{1}{2}
(\theta-\mu)^\top
\Sigma^{-1}
(\theta-\mu)
$$

여기서 $\mu,\Sigma$는 전체 사용자 데이터를 이용해 만든 population anchor이다.
사용자별 $\theta_v$는 하나의 preference prototype으로 저장된다.

$$
\mathcal{P}
=
\{\theta_1,\theta_2,\ldots,\theta_K\}
$$

따라서 새 사용자는 하나의 평균 prior에서만 시작하지 않고, 여러 기존 사용자 prototype에 대한 soft assignment로 표현된다.

$$
q_0(k)
=
P(c=k)
$$

현재 구현에서는 기존 사용자별 prototype에 더해 작은 weight의 population prototype도 함께 둔다.

$$
P(\theta_{\text{new}})
=
\sum_{k=1}^{K}
q_0(k)
\mathcal{N}(\theta_k,\Sigma_k)
$$

---

## Online Prototype Posterior Update

target user의 $t$개 context feedback을 다음처럼 둔다.

$$
\mathcal{D}_t
=
\{(\phi_s,y_s)\}_{s=1}^{t}
$$

각 prototype $k$에 대해 target user의 posterior reward coefficient를 따로 추정한다.

$$
m_{t,k}
=
\arg\min_\theta
\frac{1}{\tau}
\sum_{s=1}^{t}
\ell(y_s,\sigma(\theta^\top \phi_s))
+
\frac{1}{2}
(\theta-\theta_k)^\top
\Sigma_k^{-1}
(\theta-\theta_k)
$$

동시에 target user가 어떤 prototype에 가까운지도 업데이트한다.

$$
q_t(k)
\propto
q_0(k)
\exp
\left(
-
\frac{J_{t,k}}{\gamma}
\right)
$$

$J_{t,k}$는 $k$번째 prototype에서의 MAP objective이고, $\gamma$는 prototype assignment를 얼마나 날카롭게 할지 조절한다.

예를 들어 brake-heavy trajectory에 `Bad`를 준 사용자는 brake-sensitive prototype 쪽 weight가 커진다.
반대로 bounce-heavy trajectory에 `Good`을 주면 bounce-sensitive prototype의 weight는 줄어든다.

최종 예측은 prototype별 posterior prediction의 mixture로 계산한다.

$$
P(y=1 \mid \xi,\mathcal{D}_t)
=
\sum_{k=1}^{K}
q_t(k)
\sigma(m_{t,k}^{\top}\phi(g(\xi)))
$$

해석용 사용자 reward coefficient는 mixture mean으로 요약한다.

$$
\bar{m}_t
=
\sum_{k=1}^{K}
q_t(k)m_{t,k}
$$

$$
\hat{R}_t(\xi)
=
\bar{m}_t^\top \phi(g(\xi))
$$

즉 이 방법은 “평균 사용자로부터 얼마나 벗어나는가”가 아니라,
“어떤 기존 사용자 선호 profile에 가까워지는가”를 함께 추정한다.

---

## 해석

각 additive group $G_k$는 하나의 물리 개념에 대응한다.

예:

- longitudinal acceleration
- longitudinal jerk
- pitch rate
- bounce energy
- vertical acceleration

특정 물리 개념 $G_k$의 posterior contribution은 다음과 같다.

$$
C_k(\xi)
=
\sum_{j \in G_k}
\bar{m}_{t,j}\phi_j(\xi)
$$

평가 sample 전체에서의 평균 contribution:

$$
\bar{C}_k
=
\frac{1}{N}
\sum_{i=1}^{N}
C_k(\xi_i)
$$

불확실성:

$$
U_k
=
\sqrt{
\sum_{j \in G_k}
S_{t,jj}
}
$$

$S_t$는 prototype별 posterior covariance와 prototype 간 분산을 합친 mixture covariance approximation이다.

$|\bar{C}_k|$가 크고 $U_k$가 작으면, 해당 물리량에 대한 사용자 선호가 비교적 확실하다고 볼 수 있다.

예:

$$
\bar{C}_{\text{long jerk}} < 0
\quad\Rightarrow\quad
\text{사용자는 longitudinal jerk가 큰 주행을 Bad로 보는 경향}
$$

$$
\bar{C}_{\text{pitch rate}} > 0
\quad\Rightarrow\quad
\text{사용자는 pitch rate 변화에 비교적 관대한 경향}
$$

---

## Plot 해석

`plots/{target_user}.png`는 target user의 online posterior 예측 결과이다.
각 context feedback이 들어온 뒤 남은 query trajectory에 대해 `Good` 확률을 예측하고, 이 값이 실제 label을 얼마나 잘 구분하는지 본다.
곡선이나 점들이 positive와 negative를 잘 분리하면, 새 사용자에 대한 posterior reward가 잘 적응한 것이다.

`plots/{target_user}_prior.png`는 target feedback을 반영하기 전의 prior 예측이다.
이 plot은 기존 사용자 prototype만으로 새 사용자를 얼마나 설명할 수 있는지 보여준다.
posterior plot이 prior plot보다 좋아지면, 새 사용자의 소수 feedback이 실제로 개인화에 기여했다는 뜻이다.

`plots/seq_auroc_{target_user}.png`는 context feedback 개수에 따른 성능 변화를 보여준다.
x축이 커질수록 더 많은 target feedback을 사용한 것이고, y축은 그 시점의 query AUROC이다.
초반에 빠르게 오르면 적은 feedback만으로도 사용자 prototype assignment가 잘 잡힌다는 뜻이고, 뒤로 갈수록 흔들리면 feedback 순서나 trajectory 다양성에 민감하다는 뜻이다.

`plots/posterior_top_features.png`는 posterior reward에서 영향이 큰 feature group을 보여준다.
양수 contribution은 `Good` 방향, 음수 contribution은 `Bad` 방향으로 작용한다.
절댓값이 큰 항목은 해당 사용자의 판단에 더 강하게 연결된 물리량으로 해석한다.

`plots/train/{train_user}.png`는 각 train user의 prototype이 자기 자신의 feedback을 얼마나 잘 설명하는지 확인하는 plot이다.
이 성능이 너무 낮으면 해당 사용자의 prototype 자체가 불안정하다는 뜻이고, posterior에서 그 prototype weight가 커지더라도 해석 신뢰도는 낮게 봐야 한다.

`plots/train/user_profile_heatmap.png`는 train 사용자별 preference profile을 feature group 단위로 비교한다.
행은 기존 사용자 prototype, 열은 물리 feature group이다.
색이 비슷한 사용자들은 비슷한 선호 구조를 가진 것으로 볼 수 있고, 특정 열에서 사용자별 색이 크게 갈리면 그 물리량이 개인차를 만드는 축일 가능성이 크다.

`plots/prototypes/prototype_similarity.png`는 사용자 prototype coefficient 간 cosine similarity를 보여준다.
값이 높으면 두 사용자의 reward coefficient 방향이 비슷하다는 뜻이다.
이 그림은 명시적 clustering은 아니지만, 어떤 사용자들이 같은 선호 유형에 가까운지 확인하는 용도로 사용한다.

`plots/prototypes/prototype_weights.png`는 target user에 대한 prototype prior weight와 posterior weight를 비교한다.
prior는 feedback을 보기 전의 가정이고, posterior는 target feedback을 본 뒤의 soft assignment이다.
posterior에서 특정 사용자의 weight가 커지면, target user가 그 기존 사용자의 선호 profile과 가까워졌다고 해석한다.

`plots/prototypes/prototype_weight_evolution.png`는 target feedback이 하나씩 들어올 때 prototype posterior weight가 어떻게 변하는지 보여준다.
특정 prototype으로 빠르게 수렴하면 새 사용자의 선호 유형이 일찍 식별된 것이다.
여러 prototype 사이에서 계속 흔들리면, 현재 feedback만으로는 사용자의 선호 유형이 아직 모호하다는 뜻이다.

`plots/prototypes/prior_posterior_contributions.png`는 prior와 posterior의 feature-group contribution을 비교한다.
posterior에서 새로 커진 항목은 target feedback을 반영하면서 중요해진 물리량이다.
부호가 바뀐 항목은 기존 사용자 평균적 해석과 target user의 개인 선호가 다를 수 있는 후보로 본다.

---

## 요약

$$
\text{다른 사용자 피드백}
\rightarrow
(\theta_1,\ldots,\theta_K)
\rightarrow
\text{prototype mixture prior}
\rightarrow
\text{online feedback}
\rightarrow
\{q_t(k),m_{t,k}\}_{k=1}^{K}
\rightarrow
R_u(\xi)
$$

이 방법은 hard rule model이 아니다.  
물리적으로 해석 가능한 feature 위에서 동작하는 prototype mixture Bayesian additive reward model이다.
기존 사용자별 preference profile과 새 사용자의 prototype posterior를 함께 저장한다.
