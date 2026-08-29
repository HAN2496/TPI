import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from preference_loop.data import derive_seed, sigmoid
from preference_loop.optimization import (
    controller_bank,
    controller_grid,
    optimize_all_users,
    scenario_seeds,
    user_evaluation,
)
from reward.fully_bayesian.model import Population


BASE_METRICS = (
    "auroc_feedback",
    "auroc_evaluation",
    "brier_feedback",
    "brier_evaluation",
    "pearson_evaluation",
    "spearman_evaluation",
    "theta_rmse",
    "theta_relative_error",
)


def fit_population(cfg, env, data):
    population = Population(cfg)
    names = list(data["train_names"])
    Zs = [data["feedback"][name]["Z"] for name in names]
    ys = [data["feedback"][name]["labels"] for name in names]

    started = time.time()
    stats = population.fit(Zs, ys, env.parameter_names, names, env.feature_groups)
    print(
        f"[fully_bayesian] {population.U}/{len(names)} train users used  "
        f"elapsed={time.time() - started:.1f}s"
    )
    return population, stats


def correlation(a, b, method):
    statistic = {
        "pearson": pearsonr,
        "spearman": spearmanr,
    }[method](a, b).statistic
    if not np.isfinite(statistic):
        raise FloatingPointError(f"Undefined {method} correlation")
    return float(statistic)


def auroc(labels, scores):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def candidate_statistics(
 candidates,
 controller_by_episode,
 true_logit,
 posterior_logit,
 probabilities,
 labels,
):
    true_mean, true_std = [], []
    posterior_mean = []
    expected_good, observed_good = [], []

    for candidate in candidates:
        selected = np.all(
            np.isclose(controller_by_episode, candidate[None, :]),
            axis=1,
        )
        true_values = true_logit[selected]
        posterior_candidate = posterior_logit[:, selected].mean(axis=1)
        true_mean.append(float(true_values.mean()))
        true_std.append(float(true_values.std()))
        posterior_mean.append(posterior_candidate)
        expected_good.append(float(probabilities[selected].mean()))
        observed_good.append(float(labels[selected].mean()))

    posterior_mean = np.stack(posterior_mean, axis=1)
    return {
        "true_mean": np.asarray(true_mean),
        "true_std": np.asarray(true_std),
        "estimate_mean": posterior_mean.mean(axis=0),
        "estimate_low": np.quantile(posterior_mean, 0.05, axis=0),
        "estimate_high": np.quantile(posterior_mean, 0.95, axis=0),
        "expected_good": np.asarray(expected_good),
        "observed_good": np.asarray(observed_good),
    }


def population_metrics(env, population, train_theta):
    cohort_mean = train_theta.mean(axis=0)
    cohort_cov = np.cov(train_theta, rowvar=False)
    return {
        "target_mean_rmse": float(np.sqrt(np.mean(
            (population.mu_bar - env.population_mean) ** 2
        ))),
        "target_mean_relative_error": float(
            np.linalg.norm(population.mu_bar - env.population_mean)
            / np.linalg.norm(env.population_mean)
        ),
        "cohort_mean_rmse": float(np.sqrt(np.mean(
            (population.mu_bar - cohort_mean) ** 2
        ))),
        "cohort_mean_relative_error": float(
            np.linalg.norm(population.mu_bar - cohort_mean)
            / np.linalg.norm(cohort_mean)
        ),
        "target_cov_fro_error": float(np.linalg.norm(
            population.Sigma_bar - env.population_cov
        )),
        "target_cov_relative_error": float(
            np.linalg.norm(population.Sigma_bar - env.population_cov)
            / np.linalg.norm(env.population_cov)
        ),
        "cohort_cov_fro_error": float(np.linalg.norm(
            population.Sigma_bar - cohort_cov
        )),
        "cohort_cov_relative_error": float(
            np.linalg.norm(population.Sigma_bar - cohort_cov)
            / np.linalg.norm(cohort_cov)
        ),
    }


def user_posterior(cfg, data, population, user_name, user_index):
    if (
        data["roles"][user_name] == "train"
        and user_name in population.user_names
    ):
        return population.user(user_name)

    posterior = population.new_user(
        user_name,
        seed=derive_seed(cfg.seed, 0, user_index),
    )
    posterior.fit(
        data["feedback"][user_name]["Z"],
        data["feedback"][user_name]["labels"],
        rng=np.random.default_rng(derive_seed(cfg.seed, 1, user_index)),
    )
    return posterior


def evaluate_population(cfg, env, data, population):
    started = time.time()
    bank = controller_bank(
        cfg,
        env,
        controller_grid(env, cfg.n_controller_candidates),
        scenario_seeds(cfg, 3, cfg.n_evaluation_scenarios),
    )
    posteriors = {
        name: user_posterior(cfg, data, population, name, i)
        for i, name in enumerate(data["user_names"])
    }
    theta_hats = {
        name: posterior.theta.mean(axis=0)
        for name, posterior in posteriors.items()
    }
    controllers_by_user = optimize_all_users(cfg, env, data, theta_hats)

    evaluations = {}
    metrics_by_role = {"train": [], "test": []}
    for i, user_name in enumerate(data["user_names"]):
        role = data["roles"][user_name]
        feedback = data["feedback"][user_name]
        posterior = posteriors[user_name]
        theta_true = data["theta_true"][i]
        theta_hat = theta_hats[user_name]
        theta_sd = posterior.theta.std(axis=0)
        evaluation = user_evaluation(cfg, bank, theta_true, i)
        feedback_logits = posterior.reward(feedback["Z"])
        posterior_feedback = feedback_logits.mean(axis=0)
        probability_feedback = sigmoid(feedback_logits).mean(axis=0)
        posterior_evaluation = posterior.reward(evaluation["Z"])
        logit_hat = posterior_evaluation.mean(axis=0)
        probability_evaluation = sigmoid(posterior_evaluation).mean(axis=0)
        candidate = candidate_statistics(
            evaluation["candidates"],
            evaluation["controller"],
            evaluation["logits"],
            posterior_evaluation,
            evaluation["probabilities"],
            evaluation["labels"],
        )

        true_best_index = int(np.argmax(candidate["true_mean"]))
        true_best = evaluation["candidates"][true_best_index].copy()
        preferred = env.preferred_controller_bounds
        if np.any(true_best < preferred[:, 0]) or np.any(
            true_best > preferred[:, 1]
        ):
            raise ValueError(
                f"{user_name} preferred controller "
                f"({env.format_controller(true_best)}) outside preferred bounds"
            )

        controllers = controllers_by_user[user_name]
        for method, optimization in controllers.items():
            true_logits = optimization["evaluation_rows"] @ theta_true
            oracle_logits = optimization["oracle_rows"] @ theta_true
            estimated_logits = optimization["evaluation_rows"] @ theta_hat
            optimization["controller_error"] = float(np.linalg.norm(
                optimization["parameters"] - optimization["oracle_parameters"]
            ))
            optimization["regret"] = float(
                oracle_logits.mean() - true_logits.mean()
            )
            optimization["true_reward_mean"] = float(
                true_logits.mean() - theta_true[env.bias_index]
            )
            optimization["true_reward_std"] = float(true_logits.std())
            optimization["estimated_reward_mean"] = float(
                estimated_logits.mean() - theta_hat[env.bias_index]
            )

        theta_error = theta_hat - theta_true
        metrics = {
            "auroc_feedback": auroc(
                feedback["labels"], posterior_feedback,
            ),
            "auroc_evaluation": auroc(
                evaluation["labels"], logit_hat,
            ),
            "brier_feedback": float(np.mean(
                (probability_feedback - feedback["labels"]) ** 2
            )),
            "brier_evaluation": float(np.mean(
                (probability_evaluation - evaluation["labels"]) ** 2
            )),
            "pearson_evaluation": correlation(
                logit_hat, evaluation["logits"], "pearson",
            ),
            "spearman_evaluation": correlation(
                logit_hat, evaluation["logits"], "spearman",
            ),
            "theta_rmse": float(np.sqrt(np.mean(theta_error ** 2))),
            "theta_relative_error": float(
                np.linalg.norm(theta_error) / np.linalg.norm(theta_true)
            ),
            "true_best_controller": true_best.tolist(),
        }
        for method, optimization in controllers.items():
            metrics[f"{method}_best_controller"] = optimization["parameters"].tolist()
            metrics[f"{method}_oracle_controller"] = optimization[
                "oracle_parameters"
            ].tolist()
            metrics[f"{method}_controller_error"] = optimization["controller_error"]
            metrics[f"{method}_regret"] = optimization["regret"]
        evaluations[user_name] = {
            **metrics,
            "role": role,
            "probability_feedback": probability_feedback,
            "probability_evaluation": probability_evaluation,
            "theta_hat": theta_hat,
            "theta_sd": theta_sd,
            "theta_samples": posterior.theta,
            "candidate": candidate,
            "evaluation": evaluation,
            "controllers": controllers,
        }
        metrics_by_role[role].append(metrics)
        controller_summary = "/".join(
            f"{method}=({env.format_controller(item['parameters'])})"
            for method, item in controllers.items()
        )
        print(
            f"  [{role}] {user_name}: "
            f"AUROC feedback/eval="
            f"{metrics['auroc_feedback']:.3f}/{metrics['auroc_evaluation']:.3f}  "
            f"controller true=({env.format_controller(true_best)})/"
            f"{controller_summary}"
        )

    print(
        f"[controller evaluation: {cfg.optimization_mode} "
        f"{', '.join(cfg.controller_optimizers)}] "
        f"{len(data['user_names'])} users x "
        f"{cfg.n_controller_candidates} candidates "
        f"x {cfg.n_evaluation_scenarios} scenarios  "
        f"elapsed={time.time() - started:.1f}s"
    )
    metric_names = BASE_METRICS + tuple(
        name
        for method in cfg.controller_optimizers
        for name in (
            f"{method}_controller_error",
            f"{method}_regret",
        )
    )
    mean_metrics = {
        role: {
            key: float(np.nanmean([row[key] for row in rows]))
            for key in metric_names
        }
        for role, rows in metrics_by_role.items()
    }
    train_theta = data["theta_true"][:len(data["train_names"])]
    return {
        "users": evaluations,
        "metrics_by_role": metrics_by_role,
        "theta_hat": np.stack([
            evaluations[name]["theta_hat"] for name in data["user_names"]
        ]),
        "population_metrics": population_metrics(
            env, population, train_theta,
        ),
        "mean_metrics": mean_metrics,
        "optimizer_names": tuple(cfg.controller_optimizers),
    }
