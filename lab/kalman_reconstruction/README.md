# Kalman reconstruction lab

허용 센서만으로 Bounce와 vehicle state를 복원하는 실험이다.

## 구성

| 파일 | 역할 |
|---|---|
| `state_space.py` | `StateSpace` (u 입력 지원 KF/RTS/road posterior), discretize, calibrate, metric |
| `models.py` | body(1-DOF/qc2/hc8/pitch) × disturbance(rw/ou/matern/osc2) 조합과 `SPECS` 레지스트리 |
| `iri.py` | spatial road, Golden Car IRI |
| `viz.py` | waveform/state/spatial/hybrid/model-free plot 전부 |
| `hybrid.py` | 실제 데이터 기반 1-DOF KF residual LSTM |
| `model_free.py` | Online/offline model-free sequence model 6종 |
| `run.py` | 데이터 분리, fitting, 평가, 저장을 담당하는 단일 CLI |
| `bound_sensitivity.py` | Matern 3/2의 $f$, $\lambda$ 상한 확장 sweep (validation 선택, test 보고) |
| `methods.md` | 수식, 논문 근거, 결과와 한계. §5.6 pitch 비식별성, §10 lag 지표와 인과 필터 지연 |
| `outputs/` | parameter, metric, figure, reconstructed state |

body는 (f, qc, P)를 만들고 `augment()`가 지정한 가속도 행에 latent 외란 블록(`disturbance()`)을 꽂는다. 이름 → (body, 외란) 매핑은 `LATENT_1DOF`/`PITCH`. 외란 추가는 `disturbance()`에 분기 하나, 모델 추가는 매핑 + `model_spec` start/bounds 등록이면 `run.py`의 fit/평가/plot이 그대로 적용된다. `pitch_hc*`는 wheel speed 2ch + `a_z` 관측, `a_x` 입력의 pitch-plane half-car이며 target이 `Pitch_rate_6D`다.

## 실행

```powershell
python -m lab.kalman_reconstruction.run classical
python -m lab.kalman_reconstruction.run pitch
python -m lab.kalman_reconstruction.run hybrid
python -m lab.kalman_reconstruction.run model-free
python -m lab.kalman_reconstruction.run all
```

`hybrid`는 Matern 3/2 KF와 causal LSTM을 실제 train data로 학습한다. `model-free`는 LSTM, GRU, causal Transformer, Bi-LSTM, 1-D U-Net, full-attention Transformer를 같은 split으로 학습한다. `all`은 세 실험을 순서대로 실행한다.

## 데이터 경계

- Hybrid 입력: vertical acceleration에서 계산한 KF state와 innovation
- Model-free 입력: `Config.x_channels`의 허용 센서 13개 전부
- Model-free target: `Bounce_rate_6D` 한 개
- Train: 2,429 episode
- Validation: `조현석` 300 episode
- Test driver: 모든 fitting, normalization, early stopping에서 제외
- 합성 data는 사용하지 않음

Online 세 모델은 미래 입력을 바꾸어도 현재까지의 출력이 변하지 않는 causality test를 통과한다. Offline 세 모델은 의도적으로 전체 episode를 사용한다.
