"""envs/vmc 스택과 synthetic 사용자 모집단 빠른 점검.

사용: python test_env.py [stage ...]   (기본: rollout sweep feasible generate plot)
  rollout  — 세 제어기(p/lqr/mpc)가 돌아가는지 + Trajectory 스키마 확인
  sweep    — 제어기 파라미터 → 응답 방향성 (kp ↑ ⇒ pitch_rate ↓)
  feasible — 설계한 가상 사용자 reward의 방향성과 정보량 점검
  generate — kp=30..300, seed=0..20 뱅크에서 Gaussian 사용자·피드백·파라미터 진단
  plot     — 제어기별 pitch rate·종가속 응답 시계열 저장 (outputs/test_env/)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm as normal_dist

from envs.vmc import VMCEnv
from envs.vmc.features import rms
from envs.vmc.users import ARCHETYPES, PARAMETER_NAMES, PREFERENCE_FEATURES, User, Users, sigmoid


OUT = Path("outputs/test/test_env")
GAINS = np.linspace(30.0, 300.0, 20)  # 테스트용: 양 끝점을 포함한 균일 grid
SEEDS = np.arange(21, dtype=int)       # 0..20 inclusive
N_USERS = 15
USER_POPULATION_SEED = 201
TARGET_GOOD = 0.5
USER_FEEDBACK_RATIO_RANGE = (0.3, 0.7)
PREFERRED_GAIN_RANGE = (50.0, 200.0)
PARAM_LABELS = ("bias", "pitch²", "long²")
ORACLE_A_LEVEL_SQUARED_USER = dict(
    name="ddx2_oracle_a_level",
    step={"pitch_rate_sq": 33.70208881, "long_accel_sq": 5.50478316},
    mu=-30.0,
)

PARAMS = (
    {"controller": "p", "kp": 100.0},
    {"controller": "lqr", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"controller": "mpc", "q_diag": [0, 3000.0, 0, 0, 0, 0, 0, 0, 0, 0]},
)


def rollout():
    env = VMCEnv()
    for params in PARAMS:
        tau = env.rollout(params, n=1, seed=0)[0]
        print(f"[{params['controller']:3s}] steps={len(tau)}  channels={len(tau.channels)}  "
              f"pitch_rms={rms(tau.channels['dtheta']):.5f}  "
              f"long_rms={rms(tau.channels['ddx_com']):.4f}  "
              f"env_return={tau.channels['env_reward'].sum():.3f}")


def sweep():
    env = VMCEnv()
    for kp in (30.0, 100.0, 200.0, 300.0):
        taus = env.rollout({"controller": "p", "kp": kp}, n=2, seed=0)
        pr = np.mean([rms(t.channels["dtheta"]) for t in taus])
        print(f"kp={kp:5.0f}: pitch_rate_rms={pr:.5f}")


def _bank(env, episodes=5):
    taus = []
    for i, params in enumerate((
        {"controller": "p", "kp": 30.0},
        {"controller": "p", "kp": 100.0},
        {"controller": "p", "kp": 200.0},
        {"controller": "p", "kp": 300.0},
    )):
        taus += env.rollout(params, episodes, 100 * i)
    return taus


def feasible():
    taus = _bank(VMCEnv())
    base = np.asarray([t.channels["env_reward"].sum() for t in taus])
    specs = {**ARCHETYPES, "irrational_shaky": dict(step={"pitch_rate_sq": -4.0}, mu=4.0)}
    users = Users.designed(specs)
    print(f"bank: {len(taus)} eps  {users!r}")

    stats = users.summary(taus, ref=base)
    bad = [n for n, m in stats.items() if m["corr"] < -0.1 or m["R_std"] < 0.05]
    print(f"  verdict: {'OK' if not bad else 'X  ' + ', '.join(bad)}")
    print("  r_step mean(tau0): " + "  ".join(
        f"{u.name}={u.r_step(taus[0]).mean():+.3f}" for u in users))

    u, values = users["pitch_averse"], []
    margin = u.R(taus) - u.mu
    for scale in (0.5, 1.0, 2.0, 4.0):
        p = sigmoid(scale * margin)
        values.append(f"scale={scale}: informative={np.mean((0.05 < p) & (p < 0.95)):.2f}")
    print("  parameter-scale sweep (pitch_averse): " + "  ".join(values))


def _axes_for_users(n, ncols=3, height=2.8, sharex=False, sharey=False):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.1 * ncols, height * nrows),
        squeeze=False, sharex=sharex, sharey=sharey,
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.remove()
    return fig, flat[:n]


def _save(fig, filename):
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def _mu_for_good_ratio(rewards, target):
    """Solve mean(sigmoid(R-mu))=target; probability decreases with mu."""
    rewards = np.asarray(rewards, dtype=float)
    lo = float(rewards.min() - 20.0)
    hi = float(rewards.max() + 20.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sigmoid(rewards - mid).mean() > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _plot_user_sigmoids(users, rewards, probabilities, feedback):
    """VMC oracle_analysis Reward-vs-Probability panel, one subplot per user."""
    fig, axes = _axes_for_users(len(users), height=3.0)
    for i, (ax, user) in enumerate(zip(axes, users)):
        r, p, y = rewards[i], probabilities[i], feedback[i]
        pad = max(0.05 * np.ptp(r), 1.0)
        r_line = np.linspace(r.min() - pad, r.max() + pad, 400)
        ax.scatter(r[y == 0], p[y == 0], color="#d62728", alpha=0.28, s=8, label="Bad (0)")
        ax.scatter(r[y == 1], p[y == 1], color="#2ca02c", alpha=0.28, s=8, label="Good (1)")
        ax.plot(r_line, sigmoid(0.5 * (r_line - user.mu)),
                color="#1f77b4", ls="--", lw=1.0, label="0.5x parameter scale")
        ax.plot(r_line, sigmoid(r_line - user.mu),
                color="black", lw=1.8, label="current parameter scale")
        ax.plot(r_line, sigmoid(2.0 * (r_line - user.mu)),
                color="#ff7f0e", ls=":", lw=1.0, label="2x parameter scale")
        ax.axvline(user.mu, color="0.5", ls=":", lw=0.8)
        weight_norm = np.linalg.norm(user.weight_vector())
        ax.set_title(f"{user.name}  ||w||={weight_norm:.2f}\ny_good={y.mean():.2f}", fontsize=9)
        ax.set_xlabel("episode reward R")
        ax.set_ylabel("p(good)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Per-user sharpness: reward parameter scale → sigmoid probability", y=1.002)
    _save(fig, "user_sigmoids.png")


def _plot_feedback_ratios(users, probabilities, feedback):
    actual_good = feedback.mean(axis=1)
    expected_good = probabilities.mean(axis=1)
    x = np.arange(len(users))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(users)), 4.5))
    ax.axhspan(*USER_FEEDBACK_RATIO_RANGE, color="#4c78a8", alpha=0.08,
               label="accepted 0.3-0.7")
    ax.bar(x - width / 2, actual_good, width, color="#2ca02c", alpha=0.8, label="observed good")
    ax.bar(x + width / 2, 1.0 - actual_good, width, color="#d62728", alpha=0.8, label="observed bad")
    ax.scatter(x - width / 2, expected_good, color="black", marker="D", s=22,
               zorder=3, label="expected good")
    ax.axhline(TARGET_GOOD, color="0.4", ls=":", lw=1.0, label=f"target={TARGET_GOOD:.1f}")
    ax.set_xticks(x, [u.name.replace("user_", "u") for u in users], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("feedback ratio")
    ax.set_title("Per-user Good/Bad feedback balance")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncols=5, fontsize=8)
    _save(fig, "user_feedback_ratios.png")


def _plot_mu_sensitivity(users, rewards):
    fig, axes = _axes_for_users(len(users), height=2.8, sharey=True)
    for ax, user, r in zip(axes, users, rewards):
        mu_target = _mu_for_good_ratio(r, TARGET_GOOD)
        lo = min(float(r.min()), user.mu, mu_target) - 3.0
        hi = max(float(r.max()), user.mu, mu_target) + 3.0
        mu_grid = np.linspace(lo, hi, 300)
        ratios = sigmoid(r[:, None] - mu_grid[None, :]).mean(axis=0)
        current = float(sigmoid(r - user.mu).mean())
        ax.plot(mu_grid, ratios, color="#1f77b4", lw=1.6)
        ax.scatter([user.mu], [current], color="black", s=25, label="current")
        ax.scatter([mu_target], [TARGET_GOOD], color="#ff7f0e", marker="*", s=70, label="target mu")
        ax.axhline(TARGET_GOOD, color="0.5", ls=":", lw=0.8)
        ax.set_title(f"{user.name}\nmu={user.mu:+.3f} → {mu_target:+.3f}", fontsize=9)
        ax.set_xlabel("mu")
        ax.set_ylabel("expected good ratio")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Per-user mu adjustment (target good={TARGET_GOOD:.2f})", y=1.002)
    _save(fig, "user_good_ratio_vs_mu.png")


def _plot_gain_landscape(users, rewards):
    """Compare gains on the same scenarios using per-user reward mean and variance."""
    fig, axes = _axes_for_users(len(users), height=2.7, sharex=True, sharey=True)
    preferred = []
    print("  common-scenario reward optimum:")
    for ax, user, reward in zip(axes, users, rewards):
        reward_gain = reward.reshape(len(GAINS), len(SEEDS))
        mean = reward_gain.mean(axis=1)
        std = reward_gain.std(axis=1)
        variance = reward_gain.var(axis=1)
        best = int(np.argmax(mean))
        preferred.append(float(GAINS[best]))
        weights = user.weight_vector()
        pitch_ratio = float(weights[0] / weights.sum())

        ax.fill_between(GAINS, mean - std, mean + std, color="#1f77b4", alpha=0.18,
                        label=f"scenario ±1 sd (N={len(SEEDS)})")
        ax.plot(GAINS, mean, "o-", ms=2.5, lw=1.2, color="#1f77b4", label="mean reward")
        ax.axvline(GAINS[best], color="#e45756", ls="--", lw=1.0, label="preferred kp")
        ax.scatter([GAINS[best]], [mean[best]], color="#e45756", s=22, zorder=3)
        ax.set_title(
            f"{user.name}: pitch ratio={pitch_ratio:.3f}\n"
            f"best kp={GAINS[best]:.1f}, Var={variance[best]:.3g}", fontsize=8,
        )
        ax.set_xlabel("kp")
        ax.set_ylabel("user reward R")
        ax.set_xlim(GAINS[0], GAINS[-1])
        ax.grid(alpha=0.2)
        print(f"    {user.name}: w_pitch={weights[0]:.4f} w_long={weights[1]:.4f} "
              f"ratio={pitch_ratio:.3f}  best_kp={GAINS[best]:.2f} "
              f"mean={mean[best]:+.4f} var={variance[best]:.5f}")
    axes[0].legend(fontsize=6.5)
    fig.suptitle(
        f"Common-scenario reward landscape: {len(SEEDS)} matched seeds per gain", y=1.002
    )
    _save(fig, "generate_kp_landscape.png")
    return np.asarray(preferred)


def _plot_gain_probability(users, probabilities, feedback):
    fig, axes = _axes_for_users(len(users), height=2.7, sharex=True, sharey=True)
    for ax, user, p, y in zip(axes, users, probabilities, feedback):
        p_gain = p.reshape(len(GAINS), len(SEEDS))
        y_gain = y.reshape(len(GAINS), len(SEEDS))
        mean, std = p_gain.mean(axis=1), p_gain.std(axis=1)
        ax.fill_between(GAINS, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                        color="#1f77b4", alpha=0.18, label="seed ±1 sd")
        ax.plot(GAINS, mean, "o-", ms=2.5, lw=1.2, color="#1f77b4", label="expected")
        ax.scatter(GAINS, y_gain.mean(axis=1), s=9, color="#2ca02c", alpha=0.7, label="observed")
        ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
        ax.set_title(user.name, fontsize=9)
        ax.set_xlabel("kp")
        ax.set_ylabel("p(good)")
        ax.set_xlim(GAINS[0], GAINS[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Good-probability landscape: kp={GAINS[0]:.0f}..{GAINS[-1]:.0f}, "
                 f"seeds={SEEDS[0]}..{SEEDS[-1]}", y=1.002)
    _save(fig, "generate_kp_probability_landscape.png")


def _plot_gain_feedback_ratio(users, rewards, probabilities, feedback):
    """Show the good-label ratio for every gain and matched seed bank."""
    fig, axes = _axes_for_users(len(users), height=2.7, sharex=True, sharey=True)
    for ax, user, reward, probability, labels in zip(
        axes, users, rewards, probabilities, feedback
    ):
        reward_gain = reward.reshape(len(GAINS), len(SEEDS))
        probability_gain = probability.reshape(len(GAINS), len(SEEDS))
        label_gain = labels.reshape(len(GAINS), len(SEEDS))
        preferred = int(np.argmax(reward_gain.mean(axis=1)))
        expected_good = probability_gain.mean(axis=1)
        observed_good = label_gain.mean(axis=1)

        ax.plot(GAINS, expected_good, color="#1f77b4", lw=1.3,
                label="expected good ratio")
        ax.plot(GAINS, observed_good, "o-", color="#2ca02c", ms=3.0, lw=0.8,
                alpha=0.8, label="observed good ratio")
        ax.axvline(GAINS[preferred], color="0.25", ls=":", lw=1.0,
                   label="reward-preferred kp")
        ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
        ax.set_title(
            f"{user.name}: best kp={GAINS[preferred]:.1f}\n"
            f"good ratio@best={observed_good[preferred]:.2f}",
            fontsize=8,
        )
        ax.set_xlabel("kp")
        ax.set_ylabel("good-label ratio")
        ax.set_xlim(GAINS[0], GAINS[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6.2)
    fig.suptitle(
        f"Common-scenario good-label ratio: {len(SEEDS)} matched seeds per gain",
        y=1.002,
    )
    _save(fig, "generate_kp_feedback_ratio.png")


def _plot_oracle_a_level_squared_user(taus):
    """Diagnose one ddx-squared user fitted to Oracle A-level feedback ratios."""
    user = User(**ORACLE_A_LEVEL_SQUARED_USER)
    rewards = user.R(taus).reshape(len(GAINS), len(SEEDS))
    probabilities = user.p_good(taus).reshape(len(GAINS), len(SEEDS))
    feedback = user.feedback(taus, np.random.default_rng(3)).reshape(len(GAINS), len(SEEDS))
    reward_mean, reward_std = rewards.mean(axis=1), rewards.std(axis=1)
    expected_good = probabilities.mean(axis=1)
    observed_good = feedback.mean(axis=1)
    best = int(np.argmax(reward_mean))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    axes[0].fill_between(GAINS, reward_mean - reward_std, reward_mean + reward_std,
                         color="#1f77b4", alpha=0.18, label=f"scenario ±1 sd (N={len(SEEDS)})")
    axes[0].plot(GAINS, reward_mean, "o-", color="#1f77b4", ms=3, lw=1.3,
                 label="mean reward")
    axes[0].axvline(GAINS[best], color="#e45756", ls="--", lw=1.0,
                    label="preferred kp")
    axes[0].set_ylabel("user reward R")
    axes[0].set_title("Squared-feature reward landscape")

    axes[1].plot(GAINS, expected_good, "o-", color="#1f77b4", ms=3, lw=1.3,
                 label="expected good ratio")
    axes[1].plot(GAINS, observed_good, "o-", color="#2ca02c", ms=3, lw=0.8,
                 alpha=0.75, label="observed good ratio")
    axes[1].axvline(GAINS[best], color="#e45756", ls="--", lw=1.0,
                    label="preferred kp")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("good-label ratio")
    axes[1].set_title("Oracle A-level ratio without threshold")

    for ax in axes:
        ax.set_xlabel("kp")
        ax.set_xlim(GAINS[0], GAINS[-1])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    weights = user.weight_vector()
    fig.suptitle(
        f"{user.name}: w_pitch={weights[0]:.3f}, w_long={weights[1]:.3f}, "
        f"mu={user.mu:.1f}, best kp={GAINS[best]:.1f}",
        y=1.02,
    )
    _save(fig, "ddx_squared_oracle_a_level_user.png")
    print(
        f"  squared Oracle-A-level user: best_kp={GAINS[best]:.2f}, "
        f"p_good(kp=30/best/300)={expected_good[0]:.3f}/"
        f"{expected_good[best]:.3f}/{expected_good[-1]:.3f}"
    )


def _plot_population_parameters(users, n_mc=10000):
    """Check the actual test cohort against its generating MVN population."""
    mean, cov = users.population_mean, users.population_cov
    sd = np.sqrt(np.diag(cov))
    active = users.parameter_matrix()
    n_users, d = active.shape
    rng = np.random.default_rng(2026)

    ncols = 3
    nrows = int(np.ceil((d + 2) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.6 * nrows), squeeze=False)
    flat = axes.ravel()
    qq_ax, corr_ax = flat[d], flat[d + 1]
    for ax in flat[d + 2:]:
        ax.remove()
    for j, name in enumerate(PARAM_LABELS):
        ax = flat[j]
        lo = min(mean[j] - 4.0 * sd[j], active[:, j].min() - 0.25 * sd[j])
        hi = max(mean[j] + 4.0 * sd[j], active[:, j].max() + 0.25 * sd[j])
        edges = np.linspace(lo, hi, 9)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])

        bin_probability = np.diff(normal_dist.cdf(edges, loc=mean[j], scale=sd[j]))
        outside_probability = max(0.0, 1.0 - bin_probability.sum())
        mc_counts = rng.multinomial(
            n_users, np.append(bin_probability, outside_probability), size=n_mc
        )[:, :-1]
        mc_density = mc_counts / (n_users * widths[None, :])
        low, high = np.quantile(mc_density, [0.025, 0.975], axis=0)
        ax.fill_between(centers, low, high, step="mid", color="0.75", alpha=0.45,
                        label=f"95% envelope (n={n_users})")

        actual_counts, _ = np.histogram(active[:, j], bins=edges)
        actual_density = actual_counts / (n_users * widths)
        ax.stairs(actual_density, edges, fill=True, color="#4c78a8", alpha=0.58,
                  label=f"actual cohort (n={n_users})")
        x = np.linspace(lo, hi, 300)
        ax.plot(x, normal_dist.pdf(x, loc=mean[j], scale=sd[j]), color="black", lw=1.6,
                label="target Normal")
        ax.scatter(active[:, j], np.zeros(len(active)), marker="|", color="#e45756", s=45,
                   label="individual users" if j == 0 else None)
        ax.set_title(
            f"{name}\ntarget {mean[j]:+.2f}±{sd[j]:.2f} | "
            f"cohort {active[:, j].mean():+.2f}±{active[:, j].std(ddof=1):.2f}",
            fontsize=8,
        )
        ax.grid(alpha=0.15)
        if j == 0:
            ax.legend(fontsize=6.5)

    diff = active - mean
    mahalanobis2 = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
    probs = (np.arange(n_users) + 0.5) / n_users
    expected = chi2.ppf(probs, df=d)
    observed = np.sort(mahalanobis2)
    mc_qq = np.sort(rng.chisquare(d, size=(n_mc, n_users)), axis=1)
    qq_low, qq_median, qq_high = np.quantile(mc_qq, [0.025, 0.5, 0.975], axis=0)
    qq_ax.fill_between(expected, qq_low, qq_high, color="0.75", alpha=0.45,
                       label=f"95% envelope (n={n_users})")
    qq_ax.plot(expected, qq_median, color="0.45", ls="--", lw=1.0, label="MC median")
    qq_ax.plot(expected, observed, "o-", color="#4c78a8", ms=4, lw=1.2,
               label="actual cohort")
    limit = max(expected[-1], observed[-1], qq_high[-1])
    qq_ax.plot([0, limit], [0, limit], color="black", lw=1.0)
    qq_corr = float(np.corrcoef(expected, observed)[0, 1])
    qq_ax.set_title(f"Actual cohort: Mahalanobis² ~ χ²({d})\nQQ corr={qq_corr:.4f}", fontsize=9)
    qq_ax.set_xlabel("theoretical quantile")
    qq_ax.set_ylabel("actual cohort quantile")
    qq_ax.grid(alpha=0.2)
    qq_ax.legend(fontsize=6.5)

    sample_cov = np.cov(active, rowvar=False)
    corr_error = sample_cov / np.sqrt(np.outer(np.diag(sample_cov), np.diag(sample_cov)))
    target_corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    corr_error -= target_corr
    vmax = max(0.10, float(np.max(np.abs(corr_error))))
    image = corr_ax.imshow(corr_error, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    corr_ax.set_xticks(range(len(PARAM_LABELS)), PARAM_LABELS, rotation=65, ha="right", fontsize=7)
    corr_ax.set_yticks(range(len(PARAM_LABELS)), PARAM_LABELS, fontsize=7)
    corr_ax.set_title(f"Actual {n_users}-user cohort − target correlation", fontsize=9)
    fig.colorbar(image, ax=corr_ax, fraction=0.046, pad=0.04)

    print(f"population cohort ({n_users} users): round-trip error={users.roundtrip_error():.3e}, "
          f"joint QQ corr={qq_corr:.5f}")
    fig.suptitle("Actual generated cohort vs target Gaussian population", y=1.005)
    _save(fig, "population_parameter_distribution.png")


def _plot_population_cohort_consistency(users, n_mc=10000):
    """Place this realized cohort within equal-size cohorts from the target MVN."""
    theta = users.parameter_matrix()
    mean, cov = users.population_mean, users.population_cov
    n_users, d = theta.shape
    inv_cov = np.linalg.inv(cov)
    diff = theta - mean
    mahalanobis2 = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
    user_percentiles = chi2.cdf(mahalanobis2, df=d)

    mean_diff = theta.mean(axis=0) - mean
    mean_stat = float(n_users * mean_diff @ inv_cov @ mean_diff)
    mean_percentile = float(chi2.cdf(mean_stat, df=d))

    chol = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(chol, diff.T).T
    whitened_cov = np.cov(whitened, rowvar=False)
    cov_error = float(np.linalg.norm(whitened_cov - np.eye(d)))

    rng = np.random.default_rng(2027)
    mc_mean_stats = rng.chisquare(d, size=n_mc)
    mc = rng.standard_normal((n_mc, n_users, d))
    mc -= mc.mean(axis=1, keepdims=True)
    mc_cov = np.einsum("mni,mnj->mij", mc, mc) / (n_users - 1)
    mc_cov_error = np.linalg.norm(mc_cov - np.eye(d), axis=(1, 2))
    cov_percentile = float(np.mean(mc_cov_error <= cov_error))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(n_users)
    axes[0].bar(x, 100.0 * user_percentiles, color="#4c78a8")
    axes[0].axhline(95.0, color="#ff7f0e", ls="--", lw=1.0, label="95th percentile")
    axes[0].axhline(99.0, color="#d62728", ls=":", lw=1.0, label="99th percentile")
    axes[0].set_xticks(x, [u.name.replace("user_", "u") for u in users], rotation=55, ha="right")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].set_ylabel("population percentile")
    axes[0].set_title("Per-user Mahalanobis percentile")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(mc_mean_stats, bins=45, density=True, color="0.7", alpha=0.7,
                 label=f"MVN cohorts (n={n_users})")
    xx = np.linspace(0.0, max(mc_mean_stats.max(), mean_stat) * 1.02, 400)
    axes[1].plot(xx, chi2.pdf(xx, df=d), color="black", lw=1.4, label=f"χ²({d})")
    axes[1].axvline(mean_stat, color="#e45756", lw=2.0, label="actual cohort")
    axes[1].set_title(f"Cohort mean consistency\nstat={mean_stat:.2f}, percentile={mean_percentile:.1%}")
    axes[1].set_xlabel("n · mean Mahalanobis²")
    axes[1].set_ylabel("density")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2)

    axes[2].hist(mc_cov_error, bins=45, density=True, color="0.7", alpha=0.7,
                 label=f"MVN cohorts (n={n_users})")
    axes[2].axvline(cov_error, color="#e45756", lw=2.0, label="actual cohort")
    axes[2].set_title(f"Cohort covariance consistency\nerror={cov_error:.2f}, percentile={cov_percentile:.1%}")
    axes[2].set_xlabel("||whitened sample covariance − I||F")
    axes[2].set_ylabel("density")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.2)

    outliers_95 = int(np.sum(user_percentiles > 0.95))
    outliers_99 = int(np.sum(user_percentiles > 0.99))
    print(f"cohort consistency: mean percentile={mean_percentile:.3f}, "
          f"cov percentile={cov_percentile:.3f}, users>95%={outliers_95}, users>99%={outliers_99}")
    fig.suptitle("Is the realized user cohort typical under the target MVN?", y=1.02)
    _save(fig, "population_cohort_consistency.png")


def _plot_user_parameters(users):
    theta = users.parameter_matrix()
    mean = users.population_mean
    sd = np.sqrt(np.diag(users.population_cov))
    z = (theta - mean) / sd
    limit = max(3.0, float(np.ceil(np.max(np.abs(z)) + 0.25)))
    fig, axes = _axes_for_users(len(users), height=2.8, sharey=True)
    x = np.arange(len(PARAMETER_NAMES))
    for ax, user, values in zip(axes, users, z):
        colors = np.where(values >= 0.0, "#4c78a8", "#e45756")
        ax.bar(x, values, color=colors, alpha=0.85)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axhline(1.0, color="0.6", ls=":", lw=0.7)
        ax.axhline(-1.0, color="0.6", ls=":", lw=0.7)
        ax.set_xticks(x, PARAM_LABELS, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("z-score vs population")
        ax.set_title(
            f"{user.name}  mu={user.mu:+.2f}, ||w||={np.linalg.norm(user.weight_vector()):.2f}",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Individual user parameter draws from the Gaussian population", y=1.002)
    _save(fig, "user_parameter_profiles.png")


def generate():
    OUT.mkdir(parents=True, exist_ok=True)
    env = VMCEnv()
    taus = []
    for kp in GAINS:
        taus += env.rollout({"controller": "p", "kp": float(kp)}, len(SEEDS), int(SEEDS[0]))
    actual_seeds = {int(t.meta["seed"]) for t in taus}
    if actual_seeds != set(SEEDS.tolist()):
        raise AssertionError(f"Expected seeds {SEEDS.tolist()}, got {sorted(actual_seeds)}")

    base = np.asarray([t.channels["env_reward"].sum() for t in taus])
    users = Users.gaussian(N_USERS, seed=USER_POPULATION_SEED)
    print(f"bank: {len(taus)} eps = {len(GAINS)} gains × {len(SEEDS)} seeds  "
          f"kp=[{GAINS[0]:.0f}, {GAINS[-1]:.0f}], seed=[{SEEDS[0]}, {SEEDS[-1]}]  {users!r}")
    users.summary(taus, ref=base)

    rewards = users.R(taus)
    probabilities = users.p_good(taus)
    feedback = users.feedback(taus, np.random.default_rng(2))
    expected_good = probabilities.mean(axis=1)
    observed_good = feedback.mean(axis=1)
    for label, ratios in (
        ("expected good", expected_good),
        ("expected bad", 1.0 - expected_good),
        ("observed good", observed_good),
        ("observed bad", 1.0 - observed_good),
    ):
        outside = [
            (user.name, float(ratio)) for user, ratio in zip(users, ratios)
            if not USER_FEEDBACK_RATIO_RANGE[0] <= ratio <= USER_FEEDBACK_RATIO_RANGE[1]
        ]
        if outside:
            raise AssertionError(
                f"{label} ratios outside {USER_FEEDBACK_RATIO_RANGE}: {outside}"
            )
    print("  user feedback and mu adjustment:")
    for i, user in enumerate(users):
        mu_target = _mu_for_good_ratio(rewards[i], TARGET_GOOD)
        print(f"    {user.name}: good={feedback[i].mean():.3f} bad={1-feedback[i].mean():.3f}  "
              f"E[good]={probabilities[i].mean():.3f}  mu={user.mu:+.3f}  "
              f"mu@{TARGET_GOOD:.0%}={mu_target:+.3f}")

    _plot_user_sigmoids(users, rewards, probabilities, feedback)
    _plot_feedback_ratios(users, probabilities, feedback)
    _plot_mu_sensitivity(users, rewards)
    preferred = _plot_gain_landscape(users, rewards)
    _plot_gain_probability(users, probabilities, feedback)
    _plot_gain_feedback_ratio(users, rewards, probabilities, feedback)
    _plot_oracle_a_level_squared_user(taus)
    outside = [
        (user.name, gain) for user, gain in zip(users, preferred)
        if not PREFERRED_GAIN_RANGE[0] <= gain <= PREFERRED_GAIN_RANGE[1]
    ]
    if outside:
        raise AssertionError(f"Preferred gains outside {PREFERRED_GAIN_RANGE}: {outside}")
    _plot_population_parameters(users)
    _plot_population_cohort_consistency(users)
    _plot_user_parameters(users)


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
    _save(fig, "vmc_controller_response.png")


if __name__ == "__main__":
    for stage in sys.argv[1:] or ["rollout", "sweep", "feasible", "generate", "plot"]:
        print(f"===== {stage} =====")
        globals()[stage]()
