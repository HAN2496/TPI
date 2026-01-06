# TimeSHAP XAI for TPI

TimeSHAP 기반 설명 가능한 AI 모듈입니다.

## 개요

이 모듈은 TPI 모델의 예측을 설명하기 위해 TimeSHAP을 사용합니다:

- **어떤 feature가 중요한가?** (Feature-level importance)
- **어떤 timestep이 중요한가?** (Temporal importance)
- **어떤 (feature, timestep) 조합이 중요한가?** (Cell-level importance)

## 문제 정의

TPI는 차량 센서 데이터로 도로 이벤트를 감지합니다:

- 입력: `(F, L)` - Feature x Timestep 형태의 시계열 데이터
- 라벨: `good`/`bad` - 이벤트 발생 여부
- 모델: `p(y=1|τ) = sigmoid(reduce(r(s_t)))`
  - **Online 모델**: 각 타임스텝의 보상 `r(s_t)` 추정 후 집계
  - **Offline 모델**: 전체 궤적의 보상 `R(τ)` 직접 추정
  - `reduce`: `sum` 또는 `mean`

## 구조

```
src/xai/
├── __init__.py              # 모듈 exports
├── model_adapters.py        # 모델을 TimeSHAP 형식으로 변환
├── timeshap_explainer.py    # TimeSHAP 인터페이스
├── visualization.py         # TimeSHAP 시각화 (라이브러리 제공 함수 활용)
└── README.md               # 이 파일

scripts/
└── explain_model.py         # CLI 스크립트
```

## 사용 방법

### 1. 학습된 모델 설명하기

```bash
# OnlineLSTM 모델 설명
uv run python scripts/explain_model.py \
    -d 강신길 \
    -mt online_lstm \
    -mn base \
    -t "[5,7]" \
    --n_samples 5 \
    --n_background 50 \
    --device cuda

# OfflineLSTM 모델 설명
uv run python scripts/explain_model.py \
    -d 강신길 \
    -mt offline_lstm \
    -mn base \
    -t "[5,7]" \
    --n_samples 5 \
    --n_background 50
```

### 2. Python 코드에서 사용하기

```python
import numpy as np
from src.xai import TimeSHAPExplainer, create_background_dataset, plot_all_timeshap

# 배경 데이터 생성 (학습 데이터에서 샘플링)
background_data = create_background_dataset(
    train_X,  # (N, T, F)
    n_samples=50,
    strategy='random'
)

# Explainer 생성
explainer = TimeSHAPExplainer(
    model=trained_model,
    background_data=background_data,
    model_type='online',  # 'online' or 'offline'
    device='cpu',
    feature_names=['IMU_VerAccelVal', 'Bounce_rate_6D', ...]
)

# 단일 샘플 설명
instance = val_X[0]  # (T, F)
explanation = explainer.explain_all(instance, pruning_dict={'tol': 0.01})

# 시각화
figures = plot_all_timeshap(
    explanation,
    instance,
    feature_names=['IMU_VerAccelVal', 'Bounce_rate_6D', ...],
    save_dir='./explanations/sample_0'
)

# 결과 확인
print(f"Prediction: {explanation['prediction']:.4f}")
print(f"Top features: {np.argsort(explanation['feature_scores'])[::-1][:3]}")
print(f"Top timesteps: {np.argsort(explanation['event_scores'])[::-1][:3]}")
```

### 3. 개별 레벨 설명

```python
# Event-level: 시간 범위별 중요도
event_scores, event_data = explainer.explain_event(instance)

# Feature-level: 피처별 중요도
feature_scores, feature_data = explainer.explain_feature(instance)

# Cell-level: (feature, timestep) 중요도
cell_scores, cell_data = explainer.explain_cell(instance)
```

## 출력

`experiments/<driver>/<model_type>/<model_name>/explanations/` 디렉토리에 저장됩니다:

```
explanations/
├── sample_0_label_1/
│   ├── event_importance.png         # 시간별 중요도
│   ├── feature_importance.png       # 피처별 중요도
│   ├── cell_importance.png          # (feature, time) 히트맵
│   ├── temporal_pruning.png         # 시간 coalition pruning
│   ├── feature_pruning.png          # 피처 coalition pruning
│   ├── summary.png                  # 종합 요약
│   ├── top_cells.png                # 가장 중요한 cells
│   └── explanation_data.npz         # 설명 데이터 (numpy)
├── sample_1_label_0/
│   └── ...
...
```

## 주요 클래스

### `TimeSHAPExplainer`

TimeSHAP 기반 설명 생성기.

**주요 메서드:**
- `explain_event()`: 시간 범위별 중요도
- `explain_feature()`: 피처별 중요도
- `explain_cell()`: (feature, timestep) 중요도
- `explain_all()`: 모든 레벨 설명 (효율적)

### `OnlineModelAdapter` / `OfflineModelAdapter`

TPI 모델을 TimeSHAP 형식으로 변환.

- **Online**: `step_rewards()` 메서드 사용
- **Offline**: `forward()` 또는 `decision_function()` 사용

## 의존성

- `timeshap>=1.0.0`
- `shap>=0.37.0,<0.43`
- `plotly>=5.0.0`
- `altair>=4.2.0`

## 참고

- TimeSHAP 논문: [TimeSHAP: Explaining Recurrent Models through Sequence Perturbations (KDD 2021)](https://arxiv.org/abs/2012.00073)
- TimeSHAP GitHub: https://github.com/feedzai/timeshap
