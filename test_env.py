"""envs/vmc 스택 빠른 점검.

사용: python test_env.py [stage ...]   (기본: rollout sweep feasible plot)
  rollout  — 세 제어기(p/lqr/mpc)가 돌아가는지 + Trajectory 스키마 확인
  sweep    — 제어기 파라미터 → 응답 방향성 (kp ↑ ⇒ pitch_rate ↓)
  feasible — 가상 사용자 weight의 feasible 영역 점검 (spread·base보상 상관·라벨 정보량)
  plot     — 제어기별 pitch rate·종가속 응답 시계열 저장 (outputs/test_env/)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from envs.vmc import VMCEnv
from envs.vmc.features import rms
from envs.vmc.users import ARCHETYPES, Users

OUT = Path("outputs/test_env")
PARAMS = ({"controller": "p", "kp": 100.0},          # q_diag index 1 = dtheta (STATE_KEYS 순서)
          {"controller": "lqr", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]},
          {"controller": "mpc", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]})


def rollout():
    env = VMCEnv()
    for params in PARAMS:
        tau = env.rollout(params, n=1, seed=0)[0]
        print(f"[{params['controller']:3s}] T={len(tau)}  channels={len(tau.channels)}  "
              f"pitch_rms={rms(tau.channels['dtheta']):.5f}  "
              f"long_rms={rms(tau.channels['ddx_com']):.4f}  "
              f"env_return={tau.channels['env_reward'].sum():.3f}")


def sweep():
    env = VMCEnv()
    for kp in (0.0, 50.0, 150.0):                    # kp=0 = 무개입 기준선
        taus = env.rollout({"controller": "p", "kp": kp}, n=2, seed=0)
        pr = np.mean([rms(t.channels["dtheta"]) for t in taus])
        print(f"kp={kp:5.0f}: pitch_rate_rms={pr:.5f}")


def _bank(env, episodes=5):
    taus = []
    for i, params in enumerate(({"controller": "p", "kp": 0.0}, {"controller": "p", "kp": 50.0},
                                {"controller": "p", "kp": 150.0},
                                {"controller": "lqr", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]})):
        taus += env.rollout(params, episodes, 100 * i)
    return taus


def feasible():
    taus = _bank(VMCEnv())
    base = np.asarray([t.channels["env_reward"].sum() for t in taus])
    specs = {**ARCHETYPES, "irrational_shaky": dict(step={"pitch_rate_sq": -1.0}, mu=1.0, T=0.25)}
    users = Users.designed(specs)
    print(f"bank: {len(taus)} eps  {users!r}")

    stats = users.summary(taus, ref=base)
    bad = [n for n, m in stats.items() if m["corr"] < -0.1 or m["R_std"] < 0.05]
    print(f"  verdict: {'OK' if not bad else 'X  ' + ', '.join(bad)}")
    print("  r_step mean(tau0): " + "  ".join(
        f"{u.name}={u.r_step(taus[0]).mean():+.3f}" for u in users))

    u, sweep, T0 = users["pitch_averse"], [], users["pitch_averse"].T
    for T in (0.2, 0.5, 1.0, 2.0):
        u.T = T
        p = u.p_good(taus)
        sweep.append(f"T={T}: informative={np.mean((0.05 < p) & (p < 0.95)):.2f}")
    u.T = T0
    print("  T sweep (pitch_averse): " + "  ".join(sweep))


def plot():
    OUT.mkdir(parents=True, exist_ok=True)
    env = VMCEnv()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for params in PARAMS:
        tau = env.rollout(params, n=1, seed=0)[0]
        t = np.arange(len(tau)) * tau.dt
        axes[0].plot(t, tau.channels["dtheta"], label=params["controller"], lw=0.8)
        axes[1].plot(t, tau.channels["ddx_com"], lw=0.8)
    axes[0].set_ylabel("pitch rate [rad/s]")
    axes[1].set_ylabel("long accel [m/s²]")
    axes[1].set_xlabel("time [s]")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "vmc_controller_response.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT / 'vmc_controller_response.png'}")


if __name__ == "__main__":
    for stage in sys.argv[1:] or ["rollout", "sweep", "feasible", "plot"]:
        print(f"===== {stage} =====")
        globals()[stage]()
