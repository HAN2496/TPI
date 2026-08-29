# Preference Loop

시뮬레이터에서 "랜덤 탐색 데이터로 학습한 유저별 보상 모델이 그 유저의 최적 controller를 복원할 수 있는가"를 검증하는 루프.

## 표기

- 시나리오 $s$: seed 하나가 (초기속도, 범프 위치/폭/높이)를 결정. rollout의 랜덤성은 전부 seed로 재현된다.
- Controller $c \in \mathcal{C} = [30, 300]^P$. 현재 $P=1$ (kp)이며, 1차원은 예시일 뿐 구조는 다차원 기준.
- Trajectory $\tau(c, s) = \mathrm{rollout}(c, s)$: controller $c$로 시나리오 $s$를 달린 궤적.
- Feature $z = \phi(\tau) \in \mathbb{R}^D$: 궤적의 에피소드 평균 feature로 만든 design row $[1, \bar f_1, \dots]$. 현재 $D=3$ (bias, pitch rate, long accel). **유저와 무관한 순수 물리량.** 이하 $z(c,s) := \phi(\tau(c,s))$로 줄여 쓴다.
- 유저 $i$의 보상 파라미터 $\theta_i \in \mathbb{R}^D$: population MVN에서 샘플.
- 피드백 확률 $p(y{=}1 \mid c,s,\theta_i) = \sigma\!\left(\theta_i^\top z(c,s)\right)$.
- 유저의 최적 controller: $c_i^* = \arg\max_c J_i(c)$, $\quad J_i(c) = \theta_i^\top \mathbb{E}_s[z(c,s)]$.

## 1. 데이터 생성

유저 $i$마다 $N$개 에피소드 (기본 100):

$$c_{ij} \sim U(\mathcal{C}), \quad s_{ij} \sim p(s), \quad \tau_{ij} = \mathrm{rollout}(c_{ij}, s_{ij})$$

$$z_{ij} = \phi(\tau_{ij}), \quad y_{ij} \sim \text{Bernoulli}\!\left(\sigma(\theta_i^\top z_{ij})\right)$$

- controller와 시나리오를 **에피소드마다 독립적으로 새로 샘플 (1:1 쌍)**. 같은 $c$를 여러 시나리오에서 반복 평가하지 않는다.
- 모든 seed는 (기준 seed, 용도, 유저, 에피소드)에서 결정적으로 파생 → 유저·에피소드 간 독립, 완전 재현 가능.

## 2. 보상 추정

$\{(z_{ij}, y_{ij})\}$로 유저별 posterior $\hat\theta_i$를 추정한다 (계층 베이지안 — 본 문서 범위 밖).

- Train 유저: 전원 공동 fit.
- Test 유저: population fit에서 제외(held-out). 학습된 population prior에서 시작해 자기 라벨만으로 개인화 fit.

## 3. Controller 최적화

### 3-0. 최적화 문제의 구조

유저 $i$에게 좋은 controller란 특정 도로 하나에서가 아니라 **앞으로 만날 시나리오 분포 전체에서 평균적으로** 좋은 거동을 내는 controller다:

$$c_i^* = \arg\max_c J_i(c), \qquad J_i(c) = \mathbb{E}_s\!\left[\theta_i^\top z(c,s)\right] = \theta_i^\top \mathbb{E}_s\!\left[z(c,s)\right]$$

마지막 등호는 보상이 feature에 선형이라서 성립한다 ($\theta_i$가 상수라 기대값 밖으로 나옴). 여기서 시나리오 평균 feature를

$$g(c) := \mathbb{E}_s[z(c,s)]$$

라고 정의하면 $J_i(c) = \theta_i^\top g(c)$로 깔끔하게 쪼개진다. 각 조각의 의미:

- $g(c)$: "**controller $c$로 달리면 평균적으로 어떤 거동이 나오는가**" — 예컨대 kp를 200으로 두면 평균 pitch rate가 얼마인가 같은, 유저와 무관한 순수 동역학 정보.
- $\theta_i$: "**그 거동을 유저 $i$가 얼마나 좋아하는가**" — 유저별 선호.

즉 최적 gain을 찾는 데 필요한 재료는 딱 둘이고, 우리는 둘 다 추정치로 대체한다: $\theta_i \to \bar\theta_i$ (2단계 posterior mean), $g \to \hat g$ (아래 3-1). $\hat g$는 유저와 무관하므로 **한 번 만들어 전 유저가 공유**하고, 개인화는 $\bar\theta_i$를 통해서만 들어간다. 실제 목적함수는

$$\hat J_i(c) = \bar\theta_i^\top \hat g(c)$$

이고, 3-1이 $\hat g$를 어떻게 만드는가(online/offline), 3-2가 $\hat J_i$의 argmax를 어떻게 찾는가(Grid/CMA-ES)다.

### 3-1. 기대 feature $\hat g$ 만들기: Online / Offline

**Online — 시뮬레이터를 굴릴 수 있을 때.** $g$의 정의를 Monte Carlo로 그대로 근사한다:

$$\hat g(c) = \frac{1}{M}\sum_{m=1}^{M} z(c, s_m), \qquad S_{\mathrm{opt}} = \{s_1, \dots, s_M\}\ \text{고정 (기본 } M{=}40)$$

직관적으로는 **고정된 시험 코스 $M$개를 만들어 두고, 어떤 후보 $c$가 오든 전부 같은 코스에서 시험 주행시켜 평균 성적을 매기는 것**이다. 코스를 고정하는 이유는 두 가지: (1) 후보끼리 "쉬운 코스를 받은 운" 없이 공정하게 비교된다, (2) $\hat J_i$가 부를 때마다 값이 흔들리는 확률적 함수가 아니라 고정 표본에 대한 결정적 함수가 되어 탐색이 안정된다. 노이즈 분산은 $1/M$로 줄고, 비용은 후보 하나당 rollout $M$번. Grid는 후보 20개가 고정이라 $20 \times M$ rollout을 한 번만 계산해 전 유저가 재사용하고, CMA-ES는 후보를 즉석에서 제안하므로 제안할 때마다 굴린다.

**Offline — 시뮬레이터 없이 로그만 있을 때 (실차 대응).** 새 rollout이 불가능하므로 남는 재료는 1단계에서 쌓아둔 feedback 로그뿐이다. 로그의 에피소드 하나는 "**어떤 gain으로 달렸고**($c_{ij}$), **어떤 시나리오를 만났고**($s_{ij}$: 초기속도, 범프 위치/폭/높이 — 메타데이터로 기록됨), 그 결과 **어떤 거동이 나왔는지**($z_{ij}$)"의 기록이다. feature는 유저와 무관한 물리량이므로 라벨 $y$는 버리고 train 유저 전원의 기록을 하나로 합친다 (기본 10명 × 100개 = 1000쌍).

핵심은 $z$의 변동 대부분이 gain이 아니라 **시나리오에서** 온다는 사실이다 (큰 범프면 gain이 뭐든 discomfort가 크다). 그래서 $z$를 gain만의 함수로 보고 국소 평균하는 대신, **(gain, 시나리오 공변량) 둘 다의 함수로 회귀**한다:

$$\hat f(c, s) \approx \mathbb{E}[z \mid c, s] \qquad \text{(정규화 입력에 3차 다항 + ridge)}$$

이러면 시나리오가 만들던 큰 변동이 "노이즈"가 아니라 공변량으로 설명되는 변동이 되어 잔차가 급감하고, gain 효과는 국소 이웃이 아니라 1000개 전체로 추정된다. 기대 feature는 이 모델을 **로그에 있던 시나리오 목록 전체에 평균**해서 만든다:

$$\hat g(c) = \frac{1}{n}\sum_{j} \hat f(c, s_j)$$

어떤 후보 $c$든 같은 시나리오 목록 $\{s_j\}$로 평가되므로, online에서 $S_{\mathrm{opt}}$가 하던 "고정 시험 코스" 역할을 로그 시나리오들이 대신한다 — 후보 간 시나리오 운이 구조적으로 소거된다. 통계학의 공변량 보정(ANCOVA), off-policy evaluation의 direct method에 해당하는 표준 기법이다.

성립 조건은 두 가지다: (1) 1단계의 무작위 배정 — gain이 시나리오와 독립으로 뽑혔으므로 회귀가 교란 없이 gain 효과를 식별한다, (2) 시나리오 공변량이 관측된다 — 시뮬레이터에서는 자명하고, 실차에서도 차속·노면 추정치는 로그된다. 남는 오차원은 $\hat f$의 모형 오차(다항 차수)뿐이며, rollout 0회라 최적화 비용은 사실상 0이다.

두 모드 공통 성질: $\hat g$는 하나를 전 유저가 공유하므로 그 오차는 **전 유저 공통 편향**이다. 유저 간 비교에는 공정하지만, 오차가 크면 전 유저의 선택이 같은 방향으로 함께 밀린다.

### 3-2. 탐색: Grid / CMA-ES + Oracle 런

$\hat J_i(c) = \bar\theta_i^\top \hat g(c)$가 준비되면 argmax를 찾는다.

- **Grid**: gain 범위를 linspace 후보 20개로 자르고, 전부 $\hat J_i$를 계산해 최고를 고른다. 전역을 빠짐없이 보므로 봉우리를 놓치지 않지만, 후보 수가 $20^P$로 폭발해 사실상 $P=1$ 전용. 그리고 전수 argmax라 $\hat g$에 노이즈 스파이크가 있으면 그것까지 정확히 집어낸다 — 신호에 가장 충실한 만큼 노이즈에도 가장 취약하다.
- **CMA-ES**: 정규화된 $[0,1]^P$에서 "현재 분포로부터 후보 8개 제안 → $\hat J_i$로 채점 → 순위에 따라 분포(평균·공분산) 갱신"을 15세대 반복하고, 최종 분포의 평균을 답으로 삼는다. 차원이 올라가도 동작하는 본 수단이고, 순위 기반 국소 탐색이라 좁은 스파이크에는 Grid보다 덜 낚이는 대신 미수렴·국소해 위험이 있다.
- **Oracle 런**: 위 탐색을 유저마다 **두 번** 돌린다 — 한 번은 추정 선호 $\bar\theta_i$로 (→ $\hat c_i$, 우리 답), 한 번은 true 선호 $\theta_i$로 (→ $c_i^{\mathrm{oracle}}$). 탐색기·모드·예산이 완전히 같고 목적함수의 $\theta$만 다르므로, 4단계에서 두 답의 성능 차(regret)는 탐색이나 $\hat g$의 한계와 무관하게 **보상 추정($\hat\theta$) 오차만** 반영한다. $c_i^{\mathrm{oracle}}$은 "보상 추정이 완벽했다면 같은 파이프라인이 골랐을 답"이다.

## 4. 평가

$S_{\mathrm{opt}}$와 겹치지 않는 held-out 시나리오 집합 $S_{\mathrm{eval}}$ (기본 21개, 전 유저 공유)에서 실제 rollout으로 채점:

$$J^{\mathrm{eval}}_i(c) = \frac{1}{\lvert S_{\mathrm{eval}}\rvert} \sum_{s \in S_{\mathrm{eval}}} \theta_i^\top z(c,s)$$

- $\mathrm{regret}_i = J^{\mathrm{eval}}_i(c_i^{\mathrm{oracle}}) - J^{\mathrm{eval}}_i(\hat c_i)$
- $\mathrm{controller\_error}_i = \lVert \hat c_i - c_i^{\mathrm{oracle}} \rVert$
- true best: $S_{\mathrm{eval}}$ 위 grid landscape의 argmax. $P=1$에서만 가능한 진단용.

오차 분해: $\hat c_i \leftrightarrow c_i^{\mathrm{oracle}}$ 간격은 보상 추정($\hat\theta$) 오차, $c_i^{\mathrm{oracle}} \leftrightarrow$ true best 간격은 탐색·surrogate 오차.

## Seed 구조

| 용도 | seed | 공유 범위 |
|---|---|---|
| feedback controller | (기준 seed, 0, 유저, 에피소드) | 유저별 독립 |
| feedback 시나리오 | (기준 seed, 1, 유저, 에피소드) | 유저별 독립 |
| $S_{\mathrm{opt}}$ | (기준 seed, 2, 에피소드) | 전 유저·후보·세대 |
| $S_{\mathrm{eval}}$ | (기준 seed, 3, 에피소드) | 전 유저·후보 |

탐색 내부는 후보 간 같은 시나리오(공정 비교), 탐색과 평가는 분리된 시나리오(선택 편향 제거).

## Pseudo code

$\tau(c,s) = \mathrm{rollout}(c,s)$는 궤적, $\phi(\cdot)$는 궤적 → feature 추출이다.

**Algorithm 1 — 데이터 생성 + 보상 추정 (공통)**

$$
\begin{aligned}
&\textbf{for } i = 1, \dots, n \textbf{ do} \\
&\quad \textbf{for } j = 1, \dots, N \textbf{ do} \\
&\quad\quad c_{ij} \sim U(\mathcal{C}), \qquad s_{ij} \sim p(s) \\
&\quad\quad \tau_{ij} \leftarrow \mathrm{rollout}(c_{ij},\, s_{ij}), \qquad z_{ij} \leftarrow \phi(\tau_{ij}) \\
&\quad\quad y_{ij} \sim \mathrm{Bernoulli}\big(\sigma(\theta_i^\top z_{ij})\big) \\
&\{\hat\theta_i\}_{i=1}^{n} \leftarrow \mathrm{HierBayes}\big(\{(z_{ij}, y_{ij})\}\big)
\qquad \text{(test 유저는 prior + 자기 라벨)} \\
&S_{\mathrm{eval}} \leftarrow \{s'_1, \dots, s'_K\}
\qquad \text{(고정, } S_{\mathrm{opt}}\text{와 분리)}
\end{aligned}
$$

**Algorithm 2 — 탐색 서브루틴** (목적함수 $J$를 받아 argmax 후보를 반환)

$$
\begin{aligned}
&\textbf{function } \mathrm{Grid}(J) \\
&\quad \mathcal{G} \leftarrow \mathrm{linspace}(\mathcal{C},\, 20^P \text{개}) \\
&\quad \textbf{return } \arg\max_{c \,\in\, \mathcal{G}} J(c)
\end{aligned}
$$

$$
\begin{aligned}
&\textbf{function } \mathrm{CMAES}(J)
\qquad \text{(정규화 } [0,1]^P \text{ 공간)} \\
&\quad m \leftarrow 0.5, \quad \sigma \leftarrow 0.25, \quad C \leftarrow I \\
&\quad \textbf{for } g = 1, \dots, 15 \textbf{ do} \\
&\quad\quad x_1, \dots, x_8 \sim \mathcal{N}(m,\, \sigma^2 C)
\qquad \text{(현재 분포에서 후보 제안)} \\
&\quad\quad \text{각 } x_k \text{를 } \mathcal{C} \text{로 복원해 } J(x_k) \text{ 채점} \\
&\quad\quad (m, \sigma, C) \leftarrow \text{점수 순위 상위 후보 방향으로 갱신} \\
&\quad \textbf{return } \mathrm{decode}(m)
\end{aligned}
$$

**Algorithm 3 — Online 최적화 + 평가**

$$
\begin{aligned}
&S_{\mathrm{opt}} \leftarrow \{s_1, \dots, s_M\} \qquad \text{(고정)} \\
&\hat g(c) := \tfrac{1}{M} \textstyle\sum_{s \in S_{\mathrm{opt}}} \phi\big(\tau(c, s)\big)
\qquad \text{(호출마다 rollout } M\text{번)} \\
&\textbf{for } i = 1, \dots, n \textbf{ do} \\
&\quad \hat J_i(c) := \hat\theta_i^\top \hat g(c),
\qquad J^*_i(c) := \theta_i^\top \hat g(c) \\
&\quad \hat c_i \leftarrow \mathrm{Grid}(\hat J_i) \text{ 또는 } \mathrm{CMAES}(\hat J_i) \\
&\quad c_i^{\mathrm{oracle}} \leftarrow \mathrm{Grid}(J^*_i) \text{ 또는 } \mathrm{CMAES}(J^*_i)
\qquad \text{(같은 탐색, } \theta\text{만 true)} \\
&\quad J^{\mathrm{eval}}_i(c) := \tfrac{1}{K} \textstyle\sum_{s \in S_{\mathrm{eval}}} \theta_i^\top \phi\big(\tau(c, s)\big) \\
&\quad \mathrm{regret}_i \leftarrow J^{\mathrm{eval}}_i(c_i^{\mathrm{oracle}}) - J^{\mathrm{eval}}_i(\hat c_i)
\end{aligned}
$$

**Algorithm 4 — Offline 최적화 + 평가** ($\hat g$의 출처만 다르고 유저 루프는 Algorithm 3과 동일)

$$
\begin{aligned}
&\mathcal{D} \leftarrow \textstyle\bigcup_{i \,\in\, \mathrm{train}} \{(c_{ij}, s_{ij}, z_{ij})\}_{j=1}^{N}
\qquad \text{(추가 rollout 없음)} \\
&\hat f \leftarrow \mathrm{fit}\big((c, s) \mapsto z \,;\ \mathcal{D}\big)
\qquad \text{(시나리오 효과를 공변량으로 설명)} \\
&\hat g(c) := \tfrac{1}{n} \textstyle\sum_{j} \hat f(c,\, s_j)
\qquad \text{(로그 시나리오 목록 = 고정 시험 코스, rollout 0회)} \\
&\text{이후 유저 루프는 Algorithm 3과 동일} \quad (\text{rollout은 } J^{\mathrm{eval}}\text{에만 사용})
\end{aligned}
$$

두 모드의 차이는 $\hat g$ 한 줄뿐이다: online은 시뮬레이터를 다시 굴려 만들고, offline은 이미 쌓인 로그의 회귀로 대체한다. 탐색·oracle·평가는 동일하다.

## Offline surrogate의 설계 배경

초기 구현은 시나리오를 무시하고 $z$를 gain만의 함수로 본 k-NN 회귀였다 ("비슷한 gain의 로그 25개 평균"). 이는 실패했다 (run 20260813_235607): 시나리오 노이즈(±1.5 logit)가 gain 신호(전 구간 ~1–2)와 같은 자릿수라 25개 평균으로도 잔차 ~0.3이 남아 $\hat g$가 노이즈 지그재그가 됐고, $\hat g$는 전 유저가 공유하며 $\theta$의 부호 방향도 유저 간 동일하므로 **우연히 좋은 시나리오가 몰린 gain에서 전 유저의 $\hat J$가 함께 솟았다** — Grid argmax가 전 유저에서 72.6/257.4 두 값으로 붕괴 (winner's curse). $k$를 키우면 노이즈는 줄지만 개인화 신호까지 평활되어 해결이 안 된다.

공변량 회귀로 교체한 뒤 같은 로그·같은 $\hat\theta$로 재검증한 결과: oracle pick(true $\theta$ + surrogate)이 15명 중 13명에서 true best와 일치, 나머지 2명도 grid 한 칸 차이. 즉 surrogate 오차는 사실상 제거됐고, 우리 pick의 남은 편차는 보상 추정($\hat\theta$) 오차로 귀속된다 — regret 분해가 의도대로 동작한다.

## 관련 문헌 (offline surrogate 최적화, 근접도 순)

"offline 로그만으로 surrogate를 세우고 그 위에서 최적화한다"는 공통점을 기준으로 정리. 우리 방법은 아래 1–4의 조합이다.

| 순위 | 논문 | Surrogate / 방법 | 우리와 같은 점 | 다른 점 |
|---|---|---|---|---|
| 1 | Kallus & Zhou, "Policy Evaluation and Optimization with Continuous Treatments" (AISTATS 2018) | 커널/회귀 기반 OPE + 정책 최적화 | 로그된 (context, **연속 action**, outcome)에서 새 action 규칙을 평가·최적화 — (시나리오, gain, feature) 구조와 수학이 거의 1:1 | IPS/DR 추정기가 주인공, 회귀(DM)는 베이스라인; 개인화·선호 없음 |
| 2 | Dudík, Langford & Li, "Doubly Robust Policy Evaluation and Learning" (ICML 2011) | 보상 회귀 = **Direct Method**의 정식 정의 | "로그로 결과 모델을 적합하고, 로그된 context들에 평균해 정책을 채점" — 우리 offline의 교과서 원형 | action이 이산; DM의 모형 편향을 보완하는 DR이 본론 |
| 3 | Robins (1986), g-computation (+ Hirano & Imbens 2004, 연속 처치) | outcome 회귀 후 공변량 분포에 평균 | $\hat g(c) = \frac{1}{n}\sum_j \hat f(c, s_j)$ 공식 그 자체; 무작위 배정 시 무편향 논리 동일 | 인과추론 언어; argmax가 아니라 효과 추정이 목적 |
| 4 | Box & Wilson (1951), Response Surface Methodology (현대판: Myers, Montgomery & Anderson-Cook 교과서) | **저차 다항 회귀 표면** + 최적점 탐색 | 현재 쓰는 3차 다항 surrogate → argmax가 문자 그대로 RSM | 데이터를 로그가 아니라 실험 설계로 뽑음; 시나리오 공변량 개념 없음 |
| 5 | Trabucco et al., "Conservative Objective Models" (ICML 2021) + Design-Bench (2022) | **NN surrogate**로 offline 설계 최적화 | "로그된 (x, y)만으로 surrogate 적합 → x 최적화" = offline model-based optimization; surrogate 외삽 오차를 optimizer가 파고드는 문제가 주제 | context 축 없음; 고차원 설계 대상이라 보수적 규제가 본론 |
| 6 | Ankenman, Nelson & Staum, "Stochastic Kriging for Simulation Metamodeling" (2010) | **GP** surrogate로 확률적 시뮬레이션 응답면 근사 | 시나리오 노이즈 있는 응답을 GP로 요약 — surrogate를 GP로 바꿀 때의 참조점 | 데이터를 시뮬레이터에서 능동 수집 (로그 아님) |
| 7 | Swaminathan & Joachims, "Counterfactual Risk Minimization" (ICML 2015) | 회귀 대신 **importance weighting(IPS)** | 같은 "로그만으로 정책 개선" 문제의 반대편 해법 — DM과 IPS 두 가족의 대비 | 모델 대신 로깅 정책의 propensity 필요 (우리는 kp 균일분포라 적용 가능 — DR ablation 카드) |
| 8 | Yu et al., MOPO / Kidambi et al., MOReL (NeurIPS 2020) | 로그로 **dynamics 모델** 학습 → 불확실성 페널티 하 정책 최적화 | "학습된 가상 환경에서 최적화"를 RL 스케일로 | 정책 공간 RL; gain-scheduling(RL) 단계에서 참조 |

읽는 순서: 2 → 1 → 5. 2가 프레임, 1이 연속 action 확장, 5가 "surrogate 위 최적화의 고장 모드" 감각. 3·4는 표준 근거 인용용, 7은 "왜 IPS를 안 썼나" 질문 대비용.
