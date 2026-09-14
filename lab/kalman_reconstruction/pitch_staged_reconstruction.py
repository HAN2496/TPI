"""단순 종방향-피치 모델부터 단계적으로 확장하는 pitch-rate 복원 실험 (methods_gpt.md 후속).

Stage a — Model A' (reduced longitudinal-pitch), 변형 3개로 요소를 분리:
  a_naive  IMU 지연·레버암 없음 (methods_gpt.md Model 1 원형)
  a_lag    + IMU 1차 지연 상태 a_I (실측: 휠 파생 a_x가 IMU보다 60 ms 선행)
  a_full   + IMU 높이 레버암 h_I (a_x_IMU에 -h_I*qdot)
Stage b — Model B' (+ heave z_s, a_z 레버암 x_I*qdot, IMU 지연 공유)

공통 원칙: 중력 g = 9.81 고정, 출력 gain = 180/pi 고정(offset만 train), ML은 innovation 우도(Sarkka Thm 16.9),
Powell은 best-visited 재시작으로 조기 종료를 우회, 평가는 dev-test 3명(SEALED 2명은 최종 검증까지 봉인).
"""
import argparse
import csv
from dataclasses import replace
import time

import numpy as np
from scipy.optimize import minimize

from .run import OUTPUT, data, training_split, write_csv
from .state_space import GRAVITY, StateSpace, calibrate, discretize, discretize_input, innovation_metrics, metrics
from .viz import plot_waveforms

SEALED = ("신민철", "이강근")
DEG = 180 / np.pi
PARAMS = {  # name: (start, bounds)
    "fp": (1.2, (0.3, 3.0)), "zp": (0.3, (0.05, 1.5)), "ba": (-0.25, (-2.0, 2.0)),
    "hi": (0.3, (-1.0, 1.0)), "log_tau": (np.log(0.06), (np.log(0.005), np.log(0.3))),
    "fz": (1.3, (0.8, 3.0)), "zz": (0.3, (0.05, 1.5)), "czp": (0.0, (-300.0, 300.0)),
    "xi": (0.3, (-1.5, 1.5)),
    "log_qa": (0.0, (-6.0, 4.0)), "log_qp": (np.log(0.1), (-8.0, 4.0)), "log_qg": (-9.0, (-16.0, 0.0)),
    "log_qz": (0.0, (-6.0, 6.0)), "log_rw": (-9.0, (-14.0, 0.0)), "log_rx": (np.log(0.01), (-10.0, 2.0)),
    "log_rz": (np.log(0.1), (-8.0, 4.0)),
    "ell": (-0.29, (-1.0, 1.0)), "kappa": (-0.075, (-1.0, 1.0)), "log_rd": (np.log(1e-3), (-14.0, 0.0)),
    "log_lam_a": (np.log(5.0), (np.log(0.5), np.log(50.0))), "bt": (1.4e-3, (-0.01, 0.01)),
    "sf": (8e-5, (-5e-3, 5e-3)), "sr": (5e-4, (-5e-3, 5e-3)),
    "log_rl": (0.0, (-6.0, 6.0)),  # 방법 B: 학습용 기준 센서(6D pitch rate) 채널의 잡음 분산 [(deg/s)^2]
    "log_ga": (np.log(0.013), (np.log(0.009), np.log(0.018))),  # 토크 정상 이득 a_x/ΣT = i/(r m) [m/s² per Nm], b_T = ga·λ_a
}
# GV60 제원(축거 2.90 m, 약 2.3 t, 감속비 10.65, 타이어 반경 0.36 m)과 실측(자유감쇠, IMU 지연, Δv_w 회귀)으로 좁힌 물리 범위
PHYSICAL = {
    "fp": (1.6, (1.2, 2.0)), "zp": (0.22, (0.15, 0.4)), "ba": (0.25, (0.1, 0.4)), "hi": (-0.15, (-0.5, 0.5)),
    "log_tau": (np.log(0.03), (np.log(0.015), np.log(0.1))), "fz": (1.5, (1.2, 1.8)), "zz": (0.25, (0.15, 0.4)),
    "czp": (0.0, (-15.0, 15.0)), "kappa": (-0.075, (-0.12, -0.04)), "log_lam_a": (np.log(5.0), (0.0, np.log(20.0))),
    "log_rx": (np.log(1e-3), (np.log(4e-4), 2.0)), "log_rz": (np.log(2e-3), (np.log(9e-4), 4.0)),
}
A_NOISE = ["log_qa", "log_qp", "log_qg", "log_rw", "log_rx"]


def unpack(names, p, fixed):
    d = dict(fixed)
    for name, value in zip(names, p):
        d[name[4:] if name.startswith("log_") else name] = np.exp(value) if name.startswith("log_") else value
    return d


def build_a(d, fs):
    wp, lag = 2 * np.pi * d["fp"], "tau" in d
    n = 5 + lag  # [v_x, a_x, theta, q, grade, (a_I)]
    f, qc = np.zeros((n, n)), np.zeros((n, n))
    f[0, 1] = f[2, 3] = 1.0
    f[3, 1], f[3, 2], f[3, 3] = d["ba"], -wp * wp, -2 * d["zp"] * wp
    qc[1, 1], qc[3, 3], qc[4, 4] = d["qa"], d["qp"], d["qg"]
    imu = np.zeros(n)
    imu[1], imu[2], imu[4] = 1.0, GRAVITY, GRAVITY
    imu[1:4] -= d["hi"] * f[3, 1:4]  # -h_I * qdot (w_p feedthrough 무시)
    h = np.zeros((2, n))
    h[0, 0] = 1.0
    if lag:
        f[5] = imu / d["tau"]
        f[5, 5] = -1 / d["tau"]
        h[1, 5] = 1.0
    else:
        h[1] = imu
    P = np.eye(n)
    P[4, 4] = 0.01
    A, Q = discretize(f, qc, 1 / fs)
    return StateSpace(A, h, Q, np.diag([d["rw"], d["rx"]]), P)


def build_b(d, fs):
    wp, wz, tau = 2 * np.pi * d["fp"], 2 * np.pi * d["fz"], d["tau"]
    n = 9  # [v_x, a_x, theta, q, grade, a_I, z_s, w_s, a_Iz]
    f, qc = np.zeros((n, n)), np.zeros((n, n))
    f[0, 1] = f[2, 3] = f[6, 7] = 1.0
    f[3, 1], f[3, 2], f[3, 3] = d["ba"], -wp * wp, -2 * d["zp"] * wp
    f[7, 2], f[7, 6], f[7, 7] = d["czp"], -wz * wz, -2 * d["zz"] * wz
    qc[1, 1], qc[3, 3], qc[4, 4], qc[7, 7] = d["qa"], d["qp"], d["qg"], d["qz"]
    imu_x = np.zeros(n)
    imu_x[1], imu_x[2], imu_x[4] = 1.0, GRAVITY, GRAVITY
    imu_x[1:4] -= d["hi"] * f[3, 1:4]
    f[5] = imu_x / tau
    f[5, 5] = -1 / tau
    imu_z = f[7].copy()  # zdd 표현식
    imu_z[1:4] += d["xi"] * f[3, 1:4]  # + x_I * qdot
    f[8] = imu_z / tau
    f[8, 8] = -1 / tau
    h = np.zeros((3, n))
    h[0, 0] = h[1, 5] = h[2, 8] = 1.0
    noise = [d["rw"], d["rx"], d["rz"]]
    if "ell" in d:  # 휠속 차이 채널: dvw = ell*q + kappa*a_x (지연 없음 — 휠속이 빠른 채널)
        row = np.zeros(n)
        row[3], row[1] = d["ell"], d["kappa"]
        h = np.vstack([h, row])
        noise.append(d["rd"])
    if "rl" in d:  # 방법 B: 학습 시에만 붙는 기준 센서 채널  label = (180/pi)*q + e  (배포 필터에는 없음)
        row = np.zeros(n)
        row[3] = DEG
        h = np.vstack([h, row])
        noise.append(d["rl"])
    B = D = None
    if "bt" in d or "ga" in d:  # 토크 입력 u = [T_f, T_r]: a_x 동역학 + 휠속 차이 slip feedthrough
        f[1, 1] = -d["lam_a"]
        bc = np.zeros((n, 2))
        bc[1] = d["ga"] * d["lam_a"] if "ga" in d else d["bt"]  # ga = 정상 이득(제원), bt = 자유 이득
        _, B = discretize_input(f, bc, 1 / fs)
        D = np.zeros((len(h), 2))
        D[3, 0], D[3, 1] = d["sf"], d["sr"]
    P = np.eye(n)
    P[4, 4] = 0.01
    A, Q = discretize(f, qc, 1 / fs)
    return StateSpace(A, h, Q, np.diag(noise), P, B=B, D=D)


B_NAMES = ["fp", "zp", "ba", "hi", "log_tau", "fz", "zz", "czp", "xi"] + A_NOISE + ["log_qz", "log_rz"]
VARIANTS = {
    "a_naive": dict(build=build_a, channels=("vbar", "ax"), names=["fp", "zp", "ba"] + A_NOISE,
                    fixed={"hi": 0.0}),
    "a_lag": dict(build=build_a, channels=("vbar", "ax"), names=["fp", "zp", "ba", "log_tau"] + A_NOISE,
                  fixed={"hi": 0.0}),
    "a_full": dict(build=build_a, channels=("vbar", "ax"),
                   names=["fp", "zp", "ba", "hi", "log_tau"] + A_NOISE, fixed={}),
    "b_full": dict(build=build_b, channels=("vbar", "ax", "az"), names=B_NAMES, fixed={}),
    "c_wheel": dict(build=build_b, channels=("vbar", "ax", "az", "dvw"),
                    names=B_NAMES + ["ell", "kappa", "log_rd"], fixed={}),
    "d_torque": dict(build=build_b, channels=("vbar", "ax", "az", "dvw"),
                     names=B_NAMES + ["ell", "kappa", "log_rd", "log_lam_a", "bt", "sf", "sr"], fixed={}),
    "e_fixgeo": dict(build=build_b, channels=("vbar", "ax", "az", "dvw"),  # 회사 제원: IMU는 CG 전방 <0.5 m
                     names=[n for n in B_NAMES if n != "xi"] + ["ell", "kappa", "log_rd", "log_lam_a", "bt", "sf", "sr"],
                     fixed={"xi": 0.42}),
    "f_fixlever": dict(build=build_b, channels=("vbar", "ax", "az", "dvw"),  # + 실측 레버 ell = -0.29 m 고정
                       names=[n for n in B_NAMES if n != "xi"] + ["kappa", "log_rd", "log_lam_a", "bt", "sf", "sr"],
                       fixed={"xi": 0.42, "ell": -0.29}),
    "g_physical": dict(build=build_b, channels=("vbar", "ax", "az", "dvw"),  # f + 물리 범위(PHYSICAL) + 토크 이득을 제원 정상 이득으로
                       names=[n for n in B_NAMES if n != "xi"] + ["kappa", "log_rd", "log_lam_a", "log_ga", "sf", "sr"],
                       fixed={"xi": 0.42, "ell": -0.29}, params=PHYSICAL),
}
STAGES = {"a": ("a_naive", "a_lag", "a_full"), "b": ("b_full",), "c": ("c_wheel",), "d": ("d_torque",),
          "e": ("e_fixgeo",), "f": ("f_fixlever",), "g": ("g_physical",)}


def observations(x):
    return dict(vbar=x[:, :, 5:9].mean(2) / 3.6, ax=x[:, :, 4] * GRAVITY, az=(x[:, :, 1] - 1) * GRAVITY,
                dvw=(x[:, :, 5:7].mean(2) - x[:, :, 7:9].mean(2)) / 3.6,
                torque=np.stack((x[:, :, 10], x[:, :, 9]), -1))  # [T_f, T_r] = [Mg2, Mg1] (매핑 확정)


def run_variant(spec, obs, index, p, fs):
    y = np.stack([obs[channel][index] for channel in spec["channels"]], -1)
    ss = spec["build"](unpack(spec["names"], p, spec["fixed"]), fs)
    if "noise" in spec:  # alt: 파라미터 잡음 대신 EM 이 준 Q, R 행렬을 그대로 사용
        ss = replace(ss, Q=spec["noise"][0], R=spec["noise"][1])
    x0 = np.zeros((len(y), len(ss.A)))
    x0[:, 0] = y[:, 0, 0]
    u = obs["torque"][index] if ss.B is not None else None
    return ss.filter(y, u, x0=x0, innovations=True)


def posterior_variance(ss, T, i=3):
    """사후 상태 분산 P_{t|t}[i, i] (선형 시불변이라 데이터와 무관, Riccati 한 번). Abbeel Pred 기준의 Ω_t 에 사용."""
    p, eye, out = ss.P.copy(), np.eye(len(ss.A)), np.empty(T)
    for t in range(T):
        p = ss.A @ p @ ss.A.T + ss.Q
        k = np.linalg.solve(ss.H @ p @ ss.H.T + ss.R, ss.H @ p).T
        c = eye - k @ ss.H
        p = c @ p @ c.T + k @ ss.R @ k.T
        out[t] = p[i, i]
    return out


def coordinate_search(objective, start, bounds, maxiter=60, restarts=4, alpha=0.2, floor=0.01):
    """Abbeel 외 2005 §III-F 의 좌표 상승 근사: 원소를 하나씩 (1±α) 배 (로그 파라미터는 ±log(1+α)) 로 바꿔 보고
    좋아지면 채택, 한 바퀴에 개선이 없으면 α 를 반으로. α < 1 % 또는 maxiter 바퀴에서 종료."""
    p = np.asarray(start, float)
    best = objective(p)
    for _ in range(maxiter):
        improved = False
        for i in range(len(p)):
            for sign in (1, -1):
                q = p.copy()
                q[i] = np.clip(q[i] + sign * np.log1p(alpha), *bounds[i])
                c = objective(q)
                if c < best:
                    p, best, improved = q, c, True
        if not improved:
            alpha /= 2
            if alpha < floor:
                break
    return best, p


def fit(objective, start, bounds, maxiter=60, restarts=4):
    best = [np.inf, np.asarray(start, float)]

    def wrapped(p):
        cost = objective(p)
        cost = cost if np.isfinite(cost) else 1e6
        if cost < best[0]:
            best[:] = cost, p.copy()
        return cost

    for _ in range(restarts):
        before = best[0]
        minimize(wrapped, best[1], method="Powell", bounds=bounds, options={"maxiter": maxiter})
        if best[0] > before - 1e-4 * max(abs(before), 1e-3):
            break
    return best


def signed_lag_ms(true, pred, fs, max_shift=60):
    shifts = np.arange(-max_shift, max_shift + 1)
    out = []
    for a, b in zip(true, pred):
        cs = []
        for s in shifts:
            aa, bb = (a[:len(a) - s], b[s:]) if s >= 0 else (a[-s:], b[:len(b) + s])
            aa, bb = aa - aa.mean(), bb - bb.mean()
            cs.append(np.sum(aa * bb) / np.sqrt(np.sum(aa * aa) * np.sum(bb * bb) + 1e-12))
        out.append(shifts[int(np.argmax(cs))])
    return 1000 * np.median(out) / fs


def merge_csv(path, rows):
    keys = {(row["model"], row["objective"]) for row in rows}
    old = []
    if path.exists():
        with path.open(encoding="utf-8-sig") as stream:
            old = [row for row in csv.DictReader(stream) if (row["model"], row["objective"]) not in keys]
    write_csv(path, rows + old)  # 새 행을 앞에: 열이 추가돼도 (bounce_corr) DictWriter 의 fieldnames 가 상위집합이 되도록


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("a", "b", "c", "d", "e", "f", "g", "all"))
    parser.add_argument("--objectives", default="ml,sup",
                        help="쉼표 구분: ml, sup, joint:<mu>, aug, aug:<r_label>, alt:<n> (플랜트 sup ↔ Q,R full EM 교대 n회), "
                             "alts:<n> (플랜트 sup ↔ q_a,q_z 만 structured EM 교대 n회), joint2:<mu_p>:<mu_b> (우도 + pitch + bounce 라벨), "
                             "res:<source> / reso:<source> / pred:<P>:<source> (Abbeel 2005: 플랜트 고정, Q·R 만 라벨 기준)")
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--fit-episodes", type=int, default=300)
    parser.add_argument("--em-iters", type=int, default=2000, help="alt 의 EM 반복 수 (60회로는 미수렴: 우도가 계속 오르며 corr 이 내려감)")
    parser.add_argument("--optimizer", default="powell", choices=("powell", "coord"),
                        help="coord = Abbeel 2005 식 좌표 상승 (objective 이름에 @coord 접미)")
    args = parser.parse_args()
    cfg, x, y, ids, test = data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    drivers = np.array([value.split()[0] for value in ids])
    dev = test & ~np.isin(drivers, SEALED)
    train, validation, validation_driver = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(args.fit_episodes, len(train))).astype(int)]
    label, bounce = y[:, :, 2], y[:, :, 0]  # Pitch_rate_6D (deg/s), Bounce_rate_6D (단위 미확정)
    obs = observations(x)
    obs["label"] = label - label[train].mean()
    names = sum(STAGES.values(), ()) if args.stage == "all" else STAGES[args.stage]
    objectives = tuple(value.strip() for value in args.objectives.split(","))
    print(f"train={len(train)} fit={len(fit_index)} dev-test={dev.sum()} "
          f"(sealed {', '.join(SEALED)}: {(test & np.isin(drivers, SEALED)).sum()} ep) "
          f"validation={validation_driver}", flush=True)
    results, rows, previous = {}, [], {}
    if (OUTPUT / "pitch_staged_metrics.csv").exists():
        with (OUTPUT / "pitch_staged_metrics.csv").open(encoding="utf-8-sig") as stream:
            previous = {(row["model"], row["objective"]): row["parameters"] for row in csv.DictReader(stream)}
    for name in names:
        spec = VARIANTS[name]
        target, target_b = label[fit_index], bounce[fit_index]
        for objective in objectives:
            # objective: ml | sup | joint:<mu> (방법 A, 우도 + mu*NRMSE) | aug (방법 B, 라벨 채널 + r_label fit) | aug:<r_label>
            kind, _, value = objective.partition(":")
            train_spec = dict(spec, fixed=dict(spec["fixed"]))
            if kind == "aug":
                train_spec["channels"] = spec["channels"] + ("label",)
                if value:
                    train_spec["fixed"]["rl"] = float(value)
                else:
                    train_spec["names"] = spec["names"] + ["log_rl"]
            P_label = 0.0
            if kind in ("res", "reso", "pred"):
                # Abbeel 외 2005 재현: 플랜트(A,H,B,D)는 <source> 적합값에 고정, 잡음 Q·R (연속시간 세기 8개) 만 라벨 기준으로 적합
                #   res:<source>            Res  = mean (y − 57.3 q̂)²            (offset 없음)
                #   reso:<source>           Res + offset 정렬 (sup 과의 차이 확인용)
                #   pred:<P>:<source>       Pred = mean [log Ω + (y − 57.3 q̂)²/Ω],  Ω = 57.3² P_{t|t}[q,q] + P  (P = 기준 센서 잡음 분산, 고정)
                P_label, _, source = value.partition(":") if kind == "pred" else ("0", "", value)
                P_label = float(P_label)
                d0 = {i.split("=")[0]: float(i.split("=")[1]) for i in previous[name, source].split()}
                plant = [n for n in spec["names"] if not n.startswith(("log_q", "log_r"))]
                noise = [n for n in spec["names"] if n not in plant]
                train_spec = dict(spec, names=noise, fixed=spec["fixed"] | {(n[4:] if n.startswith("log_") else n): d0[n] for n in plant})
            table = PARAMS | spec["params"] if "params" in spec else PARAMS  # 변형별 물리 범위 override
            start = [table[key][0] for key in train_spec["names"]]
            bounds = [table[key][1] for key in train_spec["names"]]
            if kind in ("res", "reso", "pred"):
                start = [np.log(d0[n]) for n in train_spec["names"]]  # 출처 적합의 잡음값에서 출발 (논문의 '초기 추정치')
            tag = "@coord" if args.optimizer == "coord" else ""
            started, key = time.perf_counter(), f"{name}_{objective}{tag}"

            def cost(p):
                state, nu, cov = run_variant(train_spec, obs, fit_index, p, cfg.fs)
                energy = innovation_metrics(nu, cov)["energy"]
                pred = DEG * state[..., 3]
                nrmse = np.sqrt(np.mean((pred - pred.mean() + target.mean() - target) ** 2)) / target.std()
                if kind in ("res", "reso"):  # Abbeel Res (reso 는 offset 정렬 판)
                    return np.mean((pred - target) ** 2) if kind == "res" else np.mean((pred - pred.mean() + target.mean() - target) ** 2)
                if kind == "pred":  # Abbeel Pred: 필터 자신의 pitch 분산으로 라벨의 로그우도
                    ss = spec["build"](unpack(train_spec["names"], p, train_spec["fixed"]), cfg.fs)
                    omega = DEG ** 2 * posterior_variance(ss, target.shape[1]) + P_label
                    return np.mean(np.log(omega) + (pred - target) ** 2 / omega)
                if kind == "joint2":  # 우도 + mu_p·pitch NRMSE + mu_b·bounce NRMSE (bounce 는 heave 속도 w_s, 라벨 단위 미확정이라 자유 이득)
                    mu_p, mu_b = (float(v) for v in value.split(":"))
                    gain_b, offset_b = calibrate(state[..., 7], target_b)
                    nrmse_b = np.sqrt(np.mean((gain_b * state[..., 7] + offset_b - target_b) ** 2)) / target_b.std()
                    return energy + mu_p * nrmse + mu_b * nrmse_b
                return {"ml": energy, "aug": energy, "sup": nrmse, "alt": nrmse, "alts": nrmse}[kind] if kind != "joint" else energy + float(value) * nrmse

            eval_spec = spec
            if kind in ("alt", "alts"):  # 플랜트 파라미터만 sup, Q·R 은 플랜트 고정 후 EM (라벨 미사용) — n회 교대, 마지막은 EM
                # alt: full EM (Q, R 전체).  alts: structured EM — R, q_p, q_g 는 sup 값에 고정, q_a·q_z 만 갱신 (methods.md §5.8-12)
                from .run_em_noise_covariance import em
                plant = [n for n in spec["names"] if not n.startswith(("log_q", "log_r"))]
                noise = [n for n in spec["names"] if n not in plant]
                train_spec = dict(spec, names=plant, fixed=unpack(noise, [table[n][0] for n in noise], spec["fixed"]))
                start, bounds = [table[n][0] for n in plant], [table[n][1] for n in plant]
                yv = np.stack([obs[c][fit_index] for c in spec["channels"]], -1)

                def em_noise(q):  # alt: 잡음 시작값에서 cold start (플롯의 em:alt 재실행과 동일).  alts: 현재 q_a, q_z 에서 이어서
                    dq = unpack(plant, q, train_spec["fixed"])
                    ss = spec["build"](dq, cfg.fs)
                    x0 = np.zeros((len(yv), len(ss.A)))
                    x0[:, 0] = yv[:, 0, 0]
                    current = {"qa": dq["qa"], "qz": dq["qz"]}

                    def rebuild(scale):
                        current["qa"], current["qz"] = current["qa"] * scale[0], current["qz"] * scale[1]
                        return spec["build"](dq | current, cfg.fs)

                    ss, history = em(ss, yv, obs["torque"][fit_index] if ss.B is not None else None, x0, args.em_iters, 1e-6, False,
                                     rebuild if kind == "alts" else None)
                    if kind == "alts":
                        train_spec["fixed"] |= current  # 다음 라운드의 빌드와 최종 parameters 열에 수렴한 q_a, q_z 반영
                    return (ss.Q, ss.R), history[-1]

                p = np.asarray(start, float)
                if (name, "sup") in previous:  # warm start: 시작값에서는 EM 잡음과 맞는 골짜기를 못 찾음 (NRMSE 1.06) → sup 플랜트에서 출발
                    d0 = {i.split("=")[0]: float(i.split("=")[1]) for i in previous[name, "sup"].split()}
                    p = np.array([np.log(d0[n]) if n.startswith("log_") else d0[n] for n in plant])
                    if kind == "alts":
                        train_spec["fixed"] |= {n[4:]: d0[n] for n in noise}  # R, q_p, q_g (와 q_a, q_z 시작값) 를 sup 적합값으로
                    print(f"  {kind} warm start from {name}_sup", flush=True)
                for round_ in range(int(value)):
                    train_spec["noise"], energy = em_noise(p)
                    loss, p = fit(cost, p, bounds, args.maxiter, args.restarts)
                    print(f"  {kind} round {round_ + 1}: EM energy={energy:.4g} -> sup NRMSE={loss:.4f}"
                          + (f"  (q_a={train_spec['fixed']['qa']:.3g}, q_z={train_spec['fixed']['qz']:.3g})" if kind == "alts" else ""), flush=True)
                train_spec["noise"], _ = em_noise(p)
                eval_spec = dict(spec, names=plant, fixed=train_spec["fixed"], noise=train_spec["noise"])
            else:
                optimizer = coordinate_search if args.optimizer == "coord" else fit
                loss, p = optimizer(cost, start, bounds, args.maxiter, args.restarts)
                eval_spec = train_spec if kind in ("res", "reso", "pred") else spec  # res/pred: 잡음만 학습, 플랜트는 fixed 에
                p = p[:len(eval_spec["names"])]  # 평가는 항상 라벨 없는 배포 필터로 (aug의 log_rl은 마지막 원소라 제거)
            state, nu, cov = run_variant(eval_spec, obs, np.arange(len(x)), p, cfg.fs)
            q_hat = state[..., 3]
            offset = label[train].mean() - DEG * q_hat[train].mean()
            pred = DEG * q_hat + offset
            corr, rmse, _ = metrics(label[dev], pred[dev], cfg.fs)
            lag = signed_lag_ms(label[dev], pred[dev], cfg.fs)
            free_gain = calibrate(q_hat[train], label[train])[0]
            gain_b, offset_b = calibrate(state[train][..., 7], bounce[train])  # bounce 는 모든 목적함수에서 보고 (자유 이득)
            bounce_corr = np.median(metrics(bounce[dev], gain_b * state[dev][..., 7] + offset_b, cfg.fs)[0])
            white = innovation_metrics(nu[dev], cov)
            d = unpack(eval_spec["names"], p, eval_spec["fixed"])  # alt: 잡음 열은 빌드용 시작값 (실제 Q,R 은 EM, em:alt 로 재현)
            ss_eval = spec["build"](d, cfg.fs)
            if "noise" in eval_spec:
                ss_eval = replace(ss_eval, Q=eval_spec["noise"][0], R=eval_spec["noise"][1])
            omega = DEG ** 2 * posterior_variance(ss_eval, label.shape[1])  # 라벨 로그손실 (Abbeel Pred 의 test 지표, P = 0): 공분산 정직성
            label_logloss = np.mean(np.log(omega) + (pred[dev] - label[dev]) ** 2 / omega)
            results[key] = dict(pred=pred[dev], corr=corr)
            cp, rp = np.percentile(corr, [10, 50, 90]), np.median(rmse)
            values = " ".join(f"{key2}={d[key2[4:] if key2.startswith('log_') else key2]:.6g}"
                              for key2 in spec["names"])
            rows.append(dict(model=name, objective=objective + tag, loss=loss, energy_dev=white["energy"],
                             nis=white["nis"], acf1=white["acf1"], free_gain=free_gain,
                             corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2], rmse_median=rp,
                             signed_lag_ms=lag, bounce_corr=bounce_corr, label_logloss=label_logloss,
                             fit_seconds=time.perf_counter() - started, parameters=values))
            print(f"{key:16s} corr={cp[1]:.3f} rmse={rp:.2f} lag={lag:+.0f}ms free_gain={free_gain:+.1f} "
                  f"nis={white['nis']:.2f} acf1={white['acf1']:+.2f} bounce_corr={bounce_corr:.3f} "
                  f"label_logloss={label_logloss:.3f} loss={loss:.4g} ({rows[-1]['fit_seconds']:.0f}s)", flush=True)
            merge_csv(OUTPUT / "pitch_staged_metrics.csv", rows)
    plot_waveforms(label[dev], results, ids[dev], cfg.fs,
                   OUTPUT / f"pitch_staged_models_{args.stage}.png", tuple(results)[-1])
    print(f"outputs={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
