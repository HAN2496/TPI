"""단계 모델(pitch_staged_reconstruction)의 플랜트를 고정하고 잡음 공분산 Q, R 만 EM 으로 추정한다.

EM-KF (Shumway & Stoffer 1982; Ghahramani & Hinton 1996; Särkkä & Svensson §16.3.2; arXiv 2105.00250 §2.2):
  E-step  RTS smoother 로 E[x_t], P_{t|T}, Cov(x_t, x_{t-1} | y_{1:T}) = P_{t|T} J_{t-1}ᵀ
  M-step  Q ← 평균 E[(x_t − A x_{t-1} − B u_t)(·)ᵀ],  R ← 평균 E[(y_t − H x_t − D u_t)(·)ᵀ]   (논문 식 30, 31)
플랜트 A, H, B, D, P0 는 pitch_staged_metrics.csv 의 적합 결과(--sources)에서 가져와 고정. 라벨은 평가에만 쓴다.
"""
import argparse
import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from .pitch_staged_reconstruction import DEG, SEALED, VARIANTS, observations, signed_lag_ms
from .run import OUTPUT, data, training_split, write_csv
from .state_space import StateSpace, calibrate, innovation_metrics, metrics


def smoother_moments(ss, T):  # 데이터와 무관한 공분산 재귀: P_{t|T} 와 smoother gain J_t (Särkkä 12.5, 12.12)
    A, H, Q, R, eye = ss.A, ss.H, ss.Q, ss.R, np.eye(len(ss.A))
    p, pred, post = ss.P.copy(), [], []
    for _ in range(T):
        p = A @ p @ A.T + Q
        pred.append(p)
        k = np.linalg.solve(H @ p @ H.T + R, H @ p).T
        c = eye - k @ H
        p = c @ p @ c.T + k @ R @ k.T
        post.append(p)
    Ps, J = [None] * T, [None] * T
    Ps[-1] = post[-1]
    for t in range(T - 2, -1, -1):
        J[t] = np.linalg.solve(pred[t + 1], A @ post[t]).T
        Ps[t] = post[t] + J[t] @ (Ps[t + 1] - pred[t + 1]) @ J[t].T
    return np.stack(Ps), np.stack(J[:-1])


def em(ss, y, u, x0, iters, tol, diag, rebuild=None, system=False):
    """diag=False: full Q, R.  diag=True: 대각만.  rebuild 가 주어지면 structured: R 과 q_p, q_g 는 고정하고
    연속시간 세기 q_a (a_x 구동, 상태 1), q_z (bounce 구동, 상태 7) 만 M-step 대각 비율로 갱신해 Van Loan 으로 Q 를 다시 만든다
    (generalized EM 근사; 식별되는 항목만 EM 에 맡기는 방법 1, methods.md §5.8-12).
    system=True: 논문 (arXiv 2105.00250) 식 (27)–(28) 의 완전 EM — [A B], [H D] 도 스무더 모멘트의 최소제곱 해로 갱신
    (모든 원소 자유, 물리 구조 없음). m0 (에피소드별 x0) 와 P0 는 고정. 상태는 닮음변환까지만 정해지므로 평가는 라벨 선형 판독으로."""
    A, H, B, D = ss.A, ss.H, ss.B, ss.D
    N, T, _ = y.shape
    Q, R, history = ss.Q.copy(), ss.R.copy(), []
    bu = u @ B.T if u is not None else np.zeros((N, T, len(A)))
    du = u @ D.T if u is not None and D is not None else 0
    outer = lambda a, b: np.einsum("nti,ntj->ij", a, b) / N  # 에피소드 평균한 시간합 모멘트
    for it in range(iters):
        model = StateSpace(A, H, Q, R, ss.P, B=B, D=D)
        state, nu, cov = model.filter(y, u, x0, smooth=True, innovations=True)
        history.append(innovation_metrics(nu, cov)["energy"])
        Ps, J = smoother_moments(model, T)
        cross = Ps[1:] @ J.transpose(0, 2, 1)
        if system:
            uk = u if u is not None else np.zeros((N, T, 0))
            xp, xc, up = state[:, :-1], state[:, 1:], uk[:, 1:]
            Z = np.block([[outer(xp, xp) + Ps[:-1].sum(0), outer(xp, up)], [outer(up, xp), outer(up, up)]])
            AB = np.hstack([outer(xc, xp) + cross.sum(0), outer(xc, up)]) @ np.linalg.inv(Z)
            Zy = np.block([[outer(state, state) + Ps.sum(0), outer(state, uk)], [outer(uk, state), outer(uk, uk)]])
            HD = np.hstack([outer(y, state), outer(y, uk)]) @ np.linalg.inv(Zy)
            A, H = AB[:, :len(A)], HD[:, :len(A)]
            if u is not None:
                B, D = AB[:, len(A):], HD[:, len(A):]
                bu, du = u @ B.T, u @ D.T
        r = state[:, 1:] - state[:, :-1] @ A.T - bu[:, 1:]
        e = y - state @ H.T - du
        Q_new = (np.einsum("nti,ntj->ij", r, r) / N
                 + (Ps[1:] - cross @ A.T - A @ cross.transpose(0, 2, 1) + A @ Ps[:-1] @ A.T).sum(0)) / (T - 1)
        R_new = (np.einsum("nti,ntj->ij", e, e) / N + (H @ Ps @ H.T).sum(0)) / T
        Q_new, R_new = (Q_new + Q_new.T) / 2, (R_new + R_new.T) / 2
        if rebuild is not None:
            Q_new, R_new = rebuild((Q_new[1, 1] / Q[1, 1], Q_new[7, 7] / Q[7, 7])).Q, R
        elif diag:
            Q_new, R_new = np.diag(np.diag(Q_new)), np.diag(np.diag(R_new))
        done = it > 0 and abs(history[-1] - history[-2]) < tol * abs(history[-2])
        Q, R = Q_new, R_new
        if done:
            break
    return StateSpace(A, H, Q, R, ss.P, B=B, D=D), history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="f_fixlever")
    parser.add_argument("--sources", default="sup,ml,joint:3", help="플랜트를 가져올 pitch_staged_metrics.csv 의 objective")
    parser.add_argument("--variants", default="full,diag",
                        help="full | diag | structured (q_a, q_z 만, R·q_p·q_g 고정) | system (A, B, H, D 까지 자유: 논문 식 27–28)")
    parser.add_argument("--iters", type=int, default=3000)  # 60회는 미수렴 (methods.md §5.8-10)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--fit-episodes", type=int, default=200)
    parser.add_argument("--suffix", default="", help="출력 파일 이름 접미사 (em_noise_covariance<suffix>_*)")
    args = parser.parse_args()
    cfg, x, y, ids, test = data()
    drivers = np.array([value.split()[0] for value in ids])
    dev = test & ~np.isin(drivers, SEALED)
    train, _, _ = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(args.fit_episodes, len(train))).astype(int)]
    label, obs, fs = y[:, :, 2], observations(x), cfg.fs
    spec = VARIANTS[args.model]
    with (OUTPUT / "pitch_staged_metrics.csv").open(encoding="utf-8-sig") as stream:
        fits = {row["objective"]: row["parameters"] for row in csv.DictReader(stream) if row["model"] == args.model}

    def arrays(index, n):
        yv = np.stack([obs[channel][index] for channel in spec["channels"]], -1)
        x0 = np.zeros((len(index), n))
        x0[:, 0] = yv[:, 0, 0]
        return yv, obs["torque"][index], x0

    rows, histories = [], {}

    def report(source, method, ss, iterations, seconds, ss0=None):
        yv, uv, x0 = arrays(np.arange(len(label)), len(ss.A))
        state, nu, cov = ss.filter(yv, uv if ss.B is not None else None, x0, innovations=True)
        q_hat = state[..., 3]
        pred = DEG * q_hat + label[train].mean() - DEG * q_hat[train].mean()
        corr, rmse, _ = metrics(label[dev], pred[dev], fs)
        # 라벨 선형 판독 (train 으로 회귀): system EM 처럼 상태의 의미가 닮음변환으로 흐트러졌을 때의 pitch 상한
        w = np.linalg.lstsq(np.c_[state[train].reshape(-1, state.shape[-1]), np.ones(train.size * state.shape[1])],
                            label[train].ravel(), rcond=None)[0]
        readout = state @ w[:-1] + w[-1]
        corr_readout = np.median(metrics(label[dev], readout[dev], fs)[0])
        white_dev, white_fit = innovation_metrics(nu[dev], cov), innovation_metrics(nu[fit_index], cov)
        cp = np.percentile(corr, [10, 50, 90])
        plant_change = np.linalg.norm(ss.A - ss0.A) / np.linalg.norm(ss0.A) if ss0 is not None else 0.0
        rows.append(dict(model=args.model, source=source, method=method, iterations=iterations,
                         energy_fit=white_fit["energy"], energy_dev=white_dev["energy"], nis=white_dev["nis"],
                         acf1=white_dev["acf1"], corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2],
                         rmse_median=np.median(rmse), signed_lag_ms=signed_lag_ms(label[dev], pred[dev], fs),
                         free_gain=calibrate(q_hat[train], label[train])[0], corr_readout=corr_readout,
                         plant_change=plant_change, seconds=seconds,
                         Q_diag=" ".join(f"{v:.4g}" for v in np.diag(ss.Q)), R_diag=" ".join(f"{v:.4g}" for v in np.diag(ss.R))))
        print(f"{source:9s} {method:9s} it={iterations:3d} energy_fit={white_fit['energy']:.4f} corr={cp[1]:.3f} "
              f"readout={corr_readout:.3f} free_gain={rows[-1]['free_gain']:+.1f} nis={white_dev['nis']:.2f} "
              f"acf1={white_dev['acf1']:+.2f} |dA|/|A|={plant_change:.3f} Rdiag=[{rows[-1]['R_diag']}] ({seconds:.0f}s)", flush=True)

    for source in args.sources.split(","):
        d = dict(spec["fixed"]) | {item.split("=")[0]: float(item.split("=")[1]) for item in fits[source].split()}
        d = {(key[4:] if key.startswith("log_") else key): value for key, value in d.items()}
        ss0 = spec["build"](d, fs)
        report(source, "source", ss0, 0, 0.0)
        yv, uv, x0 = arrays(fit_index, len(ss0.A))
        for variant in args.variants.split(","):
            started, current = time.perf_counter(), {k: d[k] for k in ("qa", "qz") if k in d}  # structured 용 (bounce 없는 모델은 qa 만)

            def rebuild(scale):  # structured: q_a, q_z 를 비율로 갱신해 같은 플랜트로 Q 를 다시 이산화
                current["qa"], current["qz"] = current["qa"] * scale[0], current["qz"] * scale[1]
                return spec["build"](d | current, fs)

            ss, history = em(ss0, yv, uv if ss0.B is not None else None, x0, args.iters, args.tol, variant == "diag",
                             rebuild if variant == "structured" else None, system=(variant == "system"))
            histories[f"{source}_{variant}"] = history
            report(source, f"em_{variant}", ss, len(history), time.perf_counter() - started, ss0)
            if variant == "structured":
                print(f"          q_a {d['qa']:.4g} -> {current['qa']:.4g}, q_z {d['qz']:.4g} -> {current['qz']:.4g}", flush=True)
        write_csv(OUTPUT / f"em_noise_covariance{args.suffix}_metrics.csv", rows)
    (OUTPUT / f"em_noise_covariance{args.suffix}_history.json").write_text(json.dumps(histories, indent=1), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, history in histories.items():
        ax.plot(history, "o-", ms=3, label=key)
    ax.set(xlabel="EM iteration", ylabel="innovation energy on fit episodes (lower = higher likelihood)")
    ax.grid(alpha=.3), ax.legend(), fig.tight_layout()
    fig.savefig(OUTPUT / f"em_noise_covariance{args.suffix}_convergence.png", dpi=150), plt.close(fig)
    print(f"outputs={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
