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

from envs.vmc import VMCEnv, episode_features
from envs.vmc.users import USERS, User
from oracle import calibrate
from oracle.base import sigmoid

OUT = Path("outputs/test_env")
PARAMS = ({"controller": "p", "kp": 100.0},          # q_diag index 1 = dtheta (STATE_KEYS 순서)
          {"controller": "lqr", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]},
          {"controller": "mpc", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]})


def rollout():
    env = VMCEnv()
    for params in PARAMS:
        tau = env.rollout(params, n=1, seed=0)[0]
        f = episode_features(tau)
        print(f"[{params['controller']:3s}] T={len(tau)}  channels={len(tau.channels)}  "
              f"pitch_rms={f['pitch_rate_rms']:.5f}  long_rms={f['long_accel_rms']:.4f}  "
              f"env_return={tau.channels['env_reward'].sum():.3f}")


def sweep():
    env = VMCEnv()
    for kp in (0.0, 50.0, 150.0):                    # kp=0 = 무개입 기준선
        taus = env.rollout({"controller": "p", "kp": kp}, n=2, seed=0)
        pr = np.mean([episode_features(t)["pitch_rate_rms"] for t in taus])
        print(f"kp={kp:5.0f}: pitch_rate_rms={pr:.5f}")


def _bank(env, episodes=5):
    taus = []
    for i, params in enumerate(({"controller": "p", "kp": 0.0}, {"controller": "p", "kp": 50.0},
                                {"controller": "p", "kp": 150.0},
                                {"controller": "lqr", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]})):
        taus += env.rollout(params, episodes, 100 * i)
    return taus


def feasible():
    """User 설계 보상의 feasible 점검:
      spread     — R(τ)가 뱅크에서 퍼지는가 (0이면 식별 불가)
      base_corr  — R이 base 승차감 보상과 양의 상관인가 (음수면 진동 선호 = 비합리)
      informative— p=σ(β·R)가 0/1로 포화되지 않은 비율 (β=1 기준)."""
    taus = _bank(VMCEnv())
    base = np.asarray([t.channels["env_reward"].sum() for t in taus])
    print(f"bank: {len(taus)} eps")

    cases = {**USERS, "irrational_shaky": User("irrational_shaky", step={"dtheta": -75.0})}
    calibrate(list(cases.values()), taus)
    for name, u in cases.items():
        R = np.asarray([u.R(t) for t in taus])
        p = sigmoid(R)
        base_corr = float(np.mean((R - R.mean()) * (base - base.mean()))
                          / max(R.std() * base.std(), 1e-12))
        informative = float(np.mean((0.05 < p) & (p < 0.95)))
        flag = "OK " if base_corr >= -0.1 and R.std() > 0.05 else "X  "
        print(f"  {flag}{name:17s} spread={R.std():.3f}  base_corr={base_corr:+.2f}  informative={informative:.2f}"
              f"  r_step_mean={u.r_step(taus[0]).mean():+.3f}")

    R = np.asarray([USERS["pitch_averse"].R(t) for t in taus])
    print("  beta sweep (pitch_averse): " + "  ".join(
        f"b={b}: informative={np.mean((0.05 < sigmoid(b * R)) & (sigmoid(b * R) < 0.95)):.2f}"
        for b in (1.0, 3.0, 10.0, 30.0)))


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
