from dataclasses import dataclass

from core import Run
from envs.vmc.preference import make_env as make_vmc_env
from preference_loop.artifacts import record_metrics, save_artifacts
from preference_loop.data import generate_data
from preference_loop.model import evaluate_population, fit_population
from preference_loop.plots import plot_results


@dataclass
class Config:
    environment: str = "vmc"

    # Random feedback scenarios sampled independently for every user.
    controller_bounds: tuple = ((30.0, 300.0),)
    preferred_controller_bounds: tuple = ((30.0, 300.0),)
    n_controller_candidates: int = 20
    n_feedback_scenarios: int = 10  # 10, 100
    n_evaluation_scenarios: int = 21
    rollout_seed: int = 0
    controller_optimizers: tuple = ("grid", "cmaes")

    # Same-budget user-count sweep
    n_train_users: int = 100  # 100, 10
    n_test_users: int = 50  # 10, 5
    user_population_seed: int = 201 # 201 / 20
    feedback_seed: int = 15543 # 145, 15543 / 2 (15543 -> test까지 비율 20 % 이내, 145는 train만 비율 20 % 이내)
    feedback_ratio_range: tuple = (0.3, 0.7)
    timestamp: str = None
    seed: int = 42

    # Hierarchical Polya-Gamma Gibbs inference.
    n_burnin: int = 5000
    n_samples: int = 10000
    thin: int = 1
    niw_kappa0: float = 0.1
    niw_nu0: float = None
    niw_lambda0_scale: float = 1.0
    eps_var: float = None
    # The synthetic population is one Gaussian, so exact-zero selection is off.
    spike_slab: bool = False
    spike_slab_unit: str = "feature"
    spike_slab_a: float = 1.0
    spike_slab_b: float = 1.0
    newuser_n_iters: int = 8

    # Controller optimization. "online" scores every candidate on the fixed
    # S_opt scenario set; "offline" regresses features on (controller,
    # scenario covariates) over the pooled train feedback log and averages
    # predictions over the logged scenarios. Both report regret against an
    # oracle run (same optimizer, true theta) on held-out S_eval.
    optimization_mode: str = "online"
    visualize_oracle: bool = False
    n_optimization_scenarios: int = 40
    surrogate_degree: int = 3
    cma_population_size: int = 8
    cma_generations: int = 15
    cma_sigma: float = 0.25
    cma_seed: int = 73


def main(cfg=None):
    cfg = Config() if cfg is None else cfg
    run = Run("preference_loop", cfg)
    env = {
        "vmc": make_vmc_env,
    }[cfg.environment](cfg)

    data = generate_data(cfg, env)
    population, fit_stats = fit_population(cfg, env, data)
    result = evaluate_population(cfg, env, data, population)

    save_artifacts(run, env, data, population, result)
    plot_results(cfg, env, run, data, population, result)
    record_metrics(run, env, data, fit_stats, result)

    run.finish()
    return run.metrics


if __name__ == "__main__":
    main()
