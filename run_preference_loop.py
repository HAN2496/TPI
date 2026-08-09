from dataclasses import dataclass, field
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from core import Run
from envs.mujoco import (MPCPlanner, PolicyBank, Walker2dEnv, bank_report,
                         episode_features, params_label, record_video)
from oracle import synthetic
from reward.fully_bayesian.features import Features
from reward.fully_bayesian.model import Population


@dataclass
class Config:
    # ① 궤적 뱅크 — controller가 params_grid의 각 조합으로 rollout
    controller: str = "bank"       # "bank"=PPO 정책 뱅크 | "mpc"=predictive sampling MPC
    bank_run: str = "outputs/policy_banks/walker2d_competence_fast"  # bank 전용, 기존 뱅크 경로 필수
    params_grid: tuple = (
        {"profile": "baseline"},
        {"profile": "fast", "step": 51200},
        {"profile": "fast", "step": 151552},
        {"profile": "fast", "alpha": 0.33},
        {"profile": "fast", "alpha": 0.66},
    )
    mpc_grid: tuple = (            # mpc 전용 — 비용 가중치·목표치가 곧 params
        {"v_target": 1.0},
        {"v_target": 2.5},
        {"v_target": 1.5, "w_bounce": 5.0},
        {"v_target": 1.5, "w_smooth": 5.0},
        {"v_target": 1.5, "w_control": 2.0},
        {"h_target": 1.1},
    )
    mpc: dict = field(default_factory=dict)        # MPCPlanner defaults 오버라이드 (P/N/T/sigma/risk/가중치)
    video_mpc: dict = field(default_factory=dict)  # 영상용 추가 오버라이드 (mpc에 병합)
    episodes_per_params: int = 16
    horizon: int = 300
    action_noise_std: float = 0.05
    holdout_frac: float = 0.3
    video_episodes: int = 1        # params 조합당 영상 수 (0=끔) — 뱅크에 실제 든 에피소드를 재생·녹화
    video_horizon: int = 1250      # 영상 길이 [스텝] = 1250×0.008s ≈ 10초 (앞부분은 뱅크 에피소드와 동일, 낙상 시 조기 종료)

    # ② 가상 사용자
    n_users: int = 7
    beta: float = 1.0
    flip: float = 0.0

    timestamp: str = None
    seed: int = 42

    # ③ feature pipeline — walker 채널 위 manual stats (GT feature와 부분 겹침; cadence·survival은 미관측)
    standardize: bool = True
    include_bias: bool = True
    manual_stats: dict = field(default_factory=lambda: {
        "forward_speed": ["mean", "std"],
        "control_energy": ["mean"],
        "torso_angle": ["mean"],
        "torso_height": ["mean", "std"],
        "vertical_speed": ["rms"],
        "action_delta": ["rms"],
        "contact_force": ["p95_abs"],
        "foot_airborne": ["mean"],
        "foot_sync": ["mean"],
        "foot_switch": ["mean"],          # ~ cadence
        "foot_height_right": ["mean"],    # ~ clearance
        "foot_height_left": ["mean"],
        "fall_indicator": ["abs_peak"],
    })

    # ③ Gibbs
    n_burnin: int = 300
    n_samples: int = 800
    thin: int = 1
    niw_kappa0: float = 1.0
    niw_nu0: float = None
    niw_lambda0_scale: float = 1.0
    eps_var: float = None
    spike_slab: bool = True
    spike_slab_unit: str = "feature"
    spike_slab_a: float = 1.0
    spike_slab_b: float = 1.0
    newuser_n_iters: int = 8


def main(cfg=None):
    cfg = cfg or Config()
    run = Run("preference_loop", cfg)

    if cfg.controller == "mpc":
        source, grid = MPCPlanner(cfg.mpc), cfg.mpc_grid
    else:
        source, grid = PolicyBank(cfg.bank_run), cfg.params_grid
    env = Walker2dEnv(source, cfg.horizon, cfg.action_noise_std)
    taus = []
    for i, params in enumerate(grid):
        taus += env.rollout(params, cfg.episodes_per_params, cfg.seed + 1000 * i)
    lengths = [len(t) for t in taus]
    print(f"[bank] {len(taus)} episodes  len={min(lengths)}..{max(lengths)}")
    for label, s in bank_report(taus, run.plots / "bank").items():
        run.metrics[f"bank/{label}"] = s
        print(f"  {label:18s} fall={s['fall_rate']:.2f}  len={s['mean_len']:.0f}"
              f"  speed={s['speed_mean']:.2f}  env_return={s['env_return']:.0f}")
    vsource = (MPCPlanner({**cfg.mpc, **cfg.video_mpc})
               if cfg.controller == "mpc" else source)
    for i, params in enumerate(grid):
        for e in range(cfg.video_episodes):
            record_video(vsource, params, cfg.seed + 1000 * i + e,
                         run.plots / "bank" / f"{params_label(params)}_{e}.mp4",
                         cfg.video_horizon, cfg.action_noise_std)

    users = synthetic(taus, episode_features, cfg.n_users, cfg.seed, beta=cfg.beta, flip=cfg.flip)
    rng = np.random.default_rng(cfg.seed)
    ys = [u.feedback(taus, rng) for u in users]

    idx = rng.permutation(len(taus))
    n_hold = int(len(taus) * cfg.holdout_frac)
    hold_idx, train_idx = idx[:n_hold], idx[n_hold:]

    cols = tuple(cfg.manual_stats)
    phi = Features(SimpleNamespace(
        view=SimpleNamespace(cols=cols, fs=1.0 / taus[0].dt),
        standardize=cfg.standardize, include_bias=cfg.include_bias, manual_stats=cfg.manual_stats))
    X = [np.stack([t.channels[c] for c in cols], axis=-1)[None] for t in taus]  # 가변 길이 -> 에피소드별 (1,T,D)
    phi.fit([X[i] for i in train_idx], None)
    Z = np.concatenate([phi.transform(x) for x in X]).astype(np.float64)

    pop = Population(cfg)
    names = [u.name for u in users]
    pop.fit([Z[train_idx]] * len(users), [y[train_idx] for y in ys], phi.feature_names, names, phi.groups)

    R_hats, rows = [], []
    for u, y in zip(users, ys):
        R_hat = pop.user(u.name).reward(Z).mean(axis=0)          # PosteriorReward.R의 배치판 (같은 Z 재사용)
        R_gt = np.asarray([u.reward.R(t) for t in taus])
        m = dict(
            auroc_train=roc_auc_score(y[train_idx], R_hat[train_idx]),
            auroc_hold=roc_auc_score(y[hold_idx], R_hat[hold_idx]),
            pearson=float(np.corrcoef(R_hat[hold_idx], R_gt[hold_idx])[0, 1]),
            spearman=float(spearmanr(R_hat[hold_idx], R_gt[hold_idx]).statistic),
        )
        run.metrics[f"user/{u.name}"] = m
        R_hats.append((R_hat, R_gt))
        rows.append(m)
        print(f"  {u.name:22s} auroc train={m['auroc_train']:.3f} hold={m['auroc_hold']:.3f}"
              f"  corr(R_hat,R_gt) hold: pearson={m['pearson']:.3f} spearman={m['spearman']:.3f}")
    run.metrics["mean"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"[mean] {run.metrics['mean']}")

    run.plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(users), figsize=(3 * len(users), 3.2), sharex=False)
    for ax, u, (R_hat, R_gt), m in zip(np.atleast_1d(axes), users, R_hats, rows):
        ax.scatter(R_gt[train_idx], R_hat[train_idx], s=10, alpha=0.4, label="train")
        ax.scatter(R_gt[hold_idx], R_hat[hold_idx], s=14, label="hold")
        ax.set_title(f"{u.name}\nρ={m['spearman']:.2f}", fontsize=9)
        ax.set_xlabel("R_gt")
    axes[0].set_ylabel("R_hat (posterior mean logit)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run.plots / "recovery_scatter.png", dpi=150)
    plt.close(fig)

    run.finish()


if __name__ == "__main__":
    main()
