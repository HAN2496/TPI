"""EM 식별성 검증 (합성 데이터): "EM 이 틀린 것인가, 이 문제에서 Q, R 이 식별되지 않는 것인가".

sup 적합의 플랜트·Q·R 을 참값으로 두고 그 모델에서 센서 데이터를 생성한 뒤 (실제 토크 입력 사용),
흐트러뜨린 Q, R 에서 EM 을 돌려 (1) 우도가 참값 모델 수준에 닿는지, (2) Q, R 대각이 회복되는지,
(3) EM 의 Q, R 로 만든 필터가 참 pitch rate 를 참 Q, R 필터만큼 복원하는지 본다.
  회복됨   → EM·모델 구조는 맞고, 실제 데이터에서의 실패는 모델 불일치 (유색 잡음 등) 탓
  회복 안됨 → 이 관측 구성에서는 Q, R (특히 pitch 관련) 이 원리적으로 식별되지 않음
"""
import argparse
import json

import numpy as np

from .pitch_staged_reconstruction import DEG, VARIANTS, observations
from .run import OUTPUT, data, training_split
from .run_em_noise_covariance import em
from .state_space import StateSpace, calibrate, innovation_metrics, metrics


def simulate(ss, u, x0, rng):
    N, T = u.shape[:2]
    x, y = np.zeros((N, T, len(ss.A))), np.zeros((N, T, len(ss.H)))
    Lq, Lr = np.linalg.cholesky(ss.Q + 1e-15 * np.eye(len(ss.A))), np.linalg.cholesky(ss.R)
    x[:, 0] = x0
    for t in range(T):
        if t:
            x[:, t] = x[:, t - 1] @ ss.A.T + u[:, t - 1] @ ss.B.T + rng.standard_normal((N, len(ss.A))) @ Lq.T
        y[:, t] = x[:, t] @ ss.H.T + u[:, t] @ ss.D.T + rng.standard_normal((N, len(ss.H))) @ Lr.T
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="g_physical")
    parser.add_argument("--source", default="sup")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--perturb", default="3,0.33", help="EM 시작점: Q×a, R×b")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg, x, y, ids, test = data()
    train, _, _ = training_split(ids, test)
    index = train[np.linspace(0, len(train) - 1, args.episodes).astype(int)]
    obs, fs = observations(x), cfg.fs
    spec = VARIANTS[args.model]
    with (OUTPUT / "pitch_staged_metrics.csv").open(encoding="utf-8-sig") as stream:
        import csv
        fit = next(row["parameters"] for row in csv.DictReader(stream) if row["model"] == args.model and row["objective"] == args.source)
    d = dict(spec["fixed"]) | {item.split("=")[0]: float(item.split("=")[1]) for item in fit.split()}
    d = {(key[4:] if key.startswith("log_") else key): value for key, value in d.items()}
    true = spec["build"](d, fs)
    u = obs["torque"][index]
    x0 = np.zeros((len(index), len(true.A)))
    x0[:, 0] = obs["vbar"][index][:, 0]
    rng = np.random.default_rng(args.seed)
    xs, ys = simulate(true, u, x0, rng)
    q_true = xs[..., 3]
    a, b = (float(v) for v in args.perturb.split(","))
    start = StateSpace(true.A, true.H, true.Q * a, true.R * b, true.P, B=true.B, D=true.D)
    fitted, history = em(start, ys, u, x0, args.iters, 1e-6, False)

    def evaluate(name, ss):
        state, nu, cov = ss.filter(ys, u, x0, innovations=True)
        q_hat = state[..., 3]
        corr = np.median(metrics(q_true, q_hat, fs)[0])
        gain = calibrate(q_hat, q_true)[0]
        energy = innovation_metrics(nu, cov)["energy"]
        print(f"{name:>28}: energy={energy:.4f}  corr(q_hat, q_true)={corr:.3f}  gain(q_true~q_hat)={gain:.2f}  "
              f"Qdiag[1,3,4,7]={np.round(np.diag(ss.Q)[[1, 3, 4, 7]], 6)}  Rdiag={np.round(np.diag(ss.R), 6)}", flush=True)
        return dict(energy=float(energy), corr=float(corr), gain=float(gain), Q_diag=np.diag(ss.Q).tolist(), R_diag=np.diag(ss.R).tolist())

    out = dict(model=args.model, source=args.source, episodes=args.episodes, iters=len(history), perturb=[a, b],
               true=evaluate("true Q,R (reference)", true), start=evaluate("perturbed start", start),
               em=evaluate(f"EM after {len(history)} iters", fitted), history=history)
    ratio_q = np.diag(fitted.Q)[[1, 3, 4, 7]] / np.diag(true.Q)[[1, 3, 4, 7]]
    ratio_r = np.diag(fitted.R) / np.diag(true.R)
    print(f"recovery ratio  Q[a_x, q, grade, w_s] = {np.round(ratio_q, 3)}   R = {np.round(ratio_r, 3)}   (1 = recovered)", flush=True)
    out["ratio_q"], out["ratio_r"] = ratio_q.tolist(), ratio_r.tolist()
    (OUTPUT / f"em_identifiability_synthetic_{args.model}_{args.source}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
