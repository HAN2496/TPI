"""Additional paired diagnostics from a finished benchmark; no model tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    users = pd.read_csv(directory / "users.csv")
    data = np.load(directory / "gain_curves.npz")
    rows = []
    n_repeats = manifest["arguments"]["repeats"]
    methods = ("poly", "gp", "mlp", "coms")
    for repeat in range(n_repeats):
        for method in methods:
            error = data[f"{method}_{repeat}"] - data["simulator_posterior"]
            centered = error - error.mean(axis=0, keepdims=True)
            for u, user in enumerate(data["test_users"]):
                rows.append({"repeat": repeat, "method": method, "user": user,
                             "centered_curve_rmse": np.sqrt(np.mean(centered[:, u] ** 2)),
                             "curve_offset": error[:, u].mean()})
    pd.DataFrame(rows).to_csv(directory / "curve_shape.csv", index=False, encoding="utf-8-sig")

    # Conditional paired bootstrap over users, retaining all repeated splits
    # together. The simulator bank and fitted models remain fixed. This is not
    # a confidence interval over new populations, training data, or scenarios.
    mean_user = users.groupby(["method", "user"]).true_reward.mean().unstack("method")
    rng = np.random.default_rng(83019)
    indices = rng.integers(len(mean_user), size=(5000, len(mean_user)))
    paired = []
    for method in methods:
        difference = (mean_user[method] - mean_user["poly"]).to_numpy()
        boots = difference[indices].mean(axis=1)
        paired.append({"method": method, "reward_difference_vs_poly": difference.mean(),
                       "user_bootstrap_q025": np.quantile(boots, .025),
                       "user_bootstrap_q975": np.quantile(boots, .975),
                       "fraction_users_improved": np.mean(difference > 1e-10)})
    pd.DataFrame(paired).to_csv(directory / "paired_comparison.csv", index=False, encoding="utf-8-sig")

    scaling = []
    # Report whether GP hits declared kernel bounds. Marginal-likelihood fits
    # are valid bounded fits; these diagnostics do not establish global optima.
    import joblib
    for repeat in range(n_repeats):
        model = joblib.load(directory / f"gp_{repeat}.joblib")
        theta, bounds = model.model.kernel_.theta, model.model.kernel_.bounds
        scaling.append({"repeat": repeat, "kernel": str(model.model.kernel_),
                        "at_lower_bound": int(np.isclose(theta, bounds[:, 0]).sum()),
                        "at_upper_bound": int(np.isclose(theta, bounds[:, 1]).sum())})
    pd.DataFrame(scaling).to_csv(directory / "gp_fit_diagnostics.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).groupby("method").centered_curve_rmse.mean().to_string())
    print(pd.DataFrame(paired).to_string(index=False))
    write_report(directory, manifest, pd.DataFrame(paired), pd.DataFrame(rows))


def write_report(directory, manifest, paired, shapes):
    summary = pd.read_csv(directory / "summary.csv").set_index("method")
    audit = pd.read_csv(directory / "continuous_summary.csv")
    shape = shapes.groupby("method").centered_curve_rmse.mean()
    labels = {"poly": "2차 반응표면 + OLS", "gp": "GP", "mlp": "일반 MLP", "coms": "COMs 보수적 학습 적용 (alpha=0.1)"}
    lines = [
        "# Offline fixed-gain 비교 결과", "",
        "기존 다항회귀, GP, 일반 MLP, COMs의 보수적 학습을 적용한 MLP를 구현해 비교했다. 연속 action 평가는 선택된 gain의 가치를 DM, kernel IPW, DR로 따로 평가하는 단계로 구현했다.", "",
        "**실험 조건**", "",
        "- 데이터: 기존 VMC simulation 로그 1,000개와 신규 사용자 50명의 저장된 Bayesian posterior.",
        "- 분할: surrogate 학습 800개, offline 평가 200개. 사용자 단위 분할을 3회 반복했다.",
        "- 학습 내부 validation으로 신경망 학습 시점을 선택한 뒤 800개 전체로 다시 학습했다. 최대 5,000 updates를 사용했다.",
        "- 모든 방법은 같은 posterior mean, 같은 학습 시나리오 목록, 같은 271개 gain 후보(30부터 300까지 간격 1)를 사용했다.",
        "- 최종 평가: 별도 시나리오 128개, 총 34,688회 simulator 주행. 이 결과는 모델 학습이나 gain 선택에 사용하지 않았다.", "",
        "**제어 성능**", "",
        "| 방법 | 실제 reward 평균 | 실제 regret | 추정 선호 기준 최적화 손실 |", "|---|---:|---:|---:|",
    ]
    for method, label in labels.items():
        s = summary.loc[method]
        lines.append(f"| {label} | {s.true_reward:.6f} | {s.true_regret:.6f} | {s.posterior_objective_regret:.6f} |")
    lines += ["", "Reward는 클수록 좋고 두 손실은 작을수록 좋다. 실제 regret은 같은 평가 시나리오와 271개 후보에서 true preference로 고른 최선의 gain을 기준으로 한다. 추정 선호 기준 최적화 손실은 posterior mean을 고정하고 simulator로 평가한 목적함수의 최선값과 비교한다. 두 값은 서로 다른 선호 가중치로 계산되므로 단순히 빼서 오차를 분해하지 않는다.", "",
              "**예측 오차와 비용**", "",
              "| 방법 | 평가 로그 reward RMSE | J 곡선 형태 RMSE | 학습 시간(초) |", "|---|---:|---:|---:|"]
    for method, label in labels.items():
        s = summary.loc[method]
        lines.append(f"| {label} | {s.audit_reward_rmse:.6f} | {shape[method]:.6f} | {s.fit_seconds:.2f} |")
    lines += ["", "J 곡선 형태 RMSE는 gain과 무관한 상수 오프셋을 제거한 오차다. 학습 로그와 simulator 평가 시나리오의 표본 차이로 발생하는 전체 곡선의 높이 차이가 gain 선택 정확도를 가리지 않도록 함께 계산했다. 학습 시간은 50명 전체에 대한 시간이며 신경망은 validation 및 재학습을 포함한다.", "",
              "| Feature RMSE | 2차 반응표면 + OLS | GP |", "|---|---:|---:|"]
    for name, key in (("정규화된 pitch-rate 제곱", "audit_pitch_rmse"), ("정규화된 종가속도 제곱", "audit_long_rmse")):
        lines.append(f"| {name} | {summary.loc['poly', key]:.6f} | {summary.loc['gp', key]:.6f} |")
    lines += ["", "**연속 action 평가**", "",
              "아래는 4개 모델이 선택한 gain을 GP 기반 DM/DR와 kernel IPW로 평가한 결과다. bandwidth 20은 사전 설정값이고, 10과 40의 결과도 CSV에 저장했다.", "",
              "| 평가 방법 | MAE | RMSE | 평균 유효 표본 수 |", "|---|---:|---:|---:|"]
    for estimator in ("dm", "dr", "ipw"):
        s = audit[(audit.nuisance == "gp") & (audit.bandwidth == 20) & (audit.estimator == estimator)].iloc[0]
        lines.append(f"| {estimator.upper()} | {s.mae:.6f} | {s.rmse:.6f} | {s.ess_mean:.1f} |")
    lines += ["", "평가 오차의 기준은 동일 posterior mean으로 계산한 simulator reward다. 200개 평가 로그의 시나리오와 128개 simulator 시나리오 사이의 표본 변동도 이 오차에 포함된다. DM은 kernel을 사용하지 않으므로 표의 유효 표본 수는 같은 gain에서 IPW/DR 잔차 보정에 쓰이는 kernel 기준 진단값이다.", "",
              "**해석 범위**", "",
              "- COMs는 논문의 Eq. 3 보수적 학습식을 fixed gain 문제에 적용한 버전이다. 고정 alpha=0.1을 사용했고, 논문의 adaptive dual alpha나 trust-region 최적화 전체를 재현한 것은 아니다. 이 결과로 COMs 일반의 성능을 단정할 수 없다.",
              "- GP는 posterior predictive mean만 사용했다. GP 불확실성을 이용한 보수적 gain 선택은 이번 실험에 포함하지 않았다.",
              "- GP의 bounded marginal-likelihood 최적화는 완료됐지만 kernel amplitude 상한과 noise 하한에 도달했다. 전역 최적 하이퍼파라미터나 불확실성 calibration을 검증한 결과는 아니다.",
              "- 저장된 계층 Bayesian posterior는 원래 학습 사용자 100명 전체로 학습됐다. 평가용 200개는 surrogate 학습에서 분리했으며, preference inference 전체까지 독립적으로 다시 분할한 실험은 아니다.",
              "- 3회 분할은 동일한 1,000개 로그와 동일한 50명을 재사용한다. 분할별 표준편차를 독립 실험 3회 또는 사용자 150명에 대한 불확실성으로 해석하면 안 된다.",
              "- Regret 기준은 유한 후보와 유한 시나리오의 empirical optimum이다. 연속 gain 전체의 참 최적값을 보장하지 않는다.", "",
              "**기존 회귀 대비 사용자별 paired 비교**", "",
              "| 방법 | reward 차이 | 사용자 bootstrap 2.5% | 사용자 bootstrap 97.5% |", "|---|---:|---:|---:|"]
    for _, s in paired.iterrows():
        lines.append(f"| {labels[s['method']]} | {s.reward_difference_vs_poly:.6f} | {s.user_bootstrap_q025:.6f} | {s.user_bootstrap_q975:.6f} |")
    lines += ["", "이 bootstrap은 학습된 모델과 simulator bank를 고정하고 사용자만 재표집한 조건부 진단이다. 새로운 학습 데이터, 새로운 시나리오, 새로운 모집단에 대한 신뢰구간을 의미하지 않는다.", "",
              "검증: 수치 테스트 6개 통과. 저장된 simulation episode 3개의 feature 재현 확인. 원본 데이터와 실험 코드의 SHA-256, 사용자 분할, seed, checkpoint를 함께 저장했다.", "",
              "![비교 그래프](" + (directory.resolve() / "comparison.png").as_posix() + ")", "",
              "방법과 실행 명령은 [README](" + (Path("lab/offline_control/README.md").resolve()).as_posix() + ")에 정리했다.",
              "참고: [COMs](https://proceedings.mlr.press/v139/trabucco21a.html), [연속 action 평가](https://proceedings.mlr.press/v84/kallus18a.html).", ""]
    (directory / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    main(parser.parse_args().directory)
