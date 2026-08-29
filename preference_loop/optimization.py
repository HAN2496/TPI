from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from cmaes import CMA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from preference_loop.data import (
    MAX_PARALLEL_USERS,
    derive_seed,
    sampled_labels,
    sigmoid,
)


def controller_grid(env, points_per_dimension):
    axes = [
        np.linspace(lower, upper, points_per_dimension)
        for lower, upper in env.controller_bounds
    ]
    return np.asarray(list(product(*axes)), dtype=float)


def scenario_seeds(cfg, namespace, n):
    return [derive_seed(cfg.rollout_seed, namespace, s) for s in range(n)]


def design_rows(env, controller, seeds):
    return np.stack([env.design_row(env.rollout(controller, seed)) for seed in seeds])


def controller_bank(cfg, env, candidates, seeds):
    with ProcessPoolExecutor(
        max_workers=min(len(candidates), MAX_PARALLEL_USERS)
    ) as executor:
        futures = [
            executor.submit(design_rows, env, controller, seeds)
            for controller in candidates
        ]
        blocks = [future.result() for future in futures]
    return {
        "candidates": candidates,
        "controller": np.repeat(candidates, len(seeds), axis=0),
        "Z": np.concatenate(blocks),
        "features": np.stack([block.mean(axis=0) for block in blocks]),
    }


def user_evaluation(cfg, bank, theta, user_index):
    logits = bank["Z"] @ theta
    probabilities = sigmoid(logits)
    labels = sampled_labels(
        probabilities,
        np.random.default_rng(derive_seed(cfg.feedback_seed, 1, user_index)),
    )
    return {**bank, "logits": logits, "probabilities": probabilities, "labels": labels}


SCENARIO_COVARIATES = (
    "initial_velocity_mps",
    "bump_position_m",
    "bump_half_width_m",
    "bump_height_m",
)


def feature_source(cfg, env, data):
    # "online" rolls every candidate out on the fixed S_opt seeds. "offline"
    # regresses features on (controller, scenario covariates) over the pooled
    # train feedback log, then averages predictions over the logged scenarios
    # so every candidate is scored on the same scenario list.
    if cfg.optimization_mode == "online":
        return "online", scenario_seeds(cfg, 2, cfg.n_optimization_scenarios)
    names = data["train_names"]
    controllers = np.concatenate([
        data["feedback"][name]["controller"] for name in names
    ])
    Z = np.concatenate([data["feedback"][name]["Z"] for name in names])
    scenarios = np.column_stack([
        np.concatenate([data["feedback"][name]["metadata"][key] for name in names])
        for key in SCENARIO_COVARIATES
    ])
    inputs = np.column_stack([controllers, scenarios])
    lower = inputs.min(axis=0)
    span = np.ptp(inputs, axis=0)
    surrogate = make_pipeline(
        PolynomialFeatures(cfg.surrogate_degree), Ridge(alpha=1e-3),
    )
    surrogate.fit((inputs - lower) / span, Z)
    return "offline", (surrogate, scenarios, lower, span)


def controller_features(env, features, controller):
    mode, payload = features
    if mode == "online":
        return design_rows(env, controller, payload).mean(axis=0)
    surrogate, scenarios, lower, span = payload
    inputs = np.column_stack([
        np.tile(controller, (len(scenarios), 1)), scenarios,
    ])
    return surrogate.predict((inputs - lower) / span).mean(axis=0)


def decode_controller(env, normalized):
    lower = env.controller_bounds[:, 0]
    upper = env.controller_bounds[:, 1]
    return lower + np.asarray(normalized, dtype=float) * (upper - lower)


def optimize_grid(cfg, env, theta, seed, features, grid):
    rewards = grid["features"] @ theta
    best = int(np.argmax(rewards))
    return {
        "method": "grid",
        "parameters": grid["candidates"][best].copy(),
        "candidate_index": best,
        "reward": rewards,
    }


def optimize_cmaes(cfg, env, theta, seed, features, grid):
    optimizer = CMA(
        mean=np.full(env.controller_dim, 0.5),
        sigma=cfg.cma_sigma,
        bounds=np.tile([0.0, 1.0], (env.controller_dim, 1)),
        seed=seed,
        population_size=cfg.cma_population_size,
    )
    generations, controllers, rewards = [], [], []
    for generation in range(cfg.cma_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            normalized = optimizer.ask()
            controller = decode_controller(env, normalized)
            reward = float(controller_features(env, features, controller) @ theta)
            solutions.append((normalized, -reward))
            generations.append(generation)
            controllers.append(controller)
            rewards.append(reward)
        optimizer.tell(solutions)
    return {
        "method": "cmaes",
        "parameters": decode_controller(env, optimizer.mean),
        "generation": np.asarray(generations, dtype=int),
        "controller": np.stack(controllers),
        "reward": np.asarray(rewards),
    }


OPTIMIZERS = {"grid": optimize_grid, "cmaes": optimize_cmaes}


def optimize_controllers(cfg, env, theta_hat, theta_true, user_index, features, grid):
    eval_seeds = scenario_seeds(cfg, 3, cfg.n_evaluation_scenarios)
    controllers = {}
    for method in cfg.controller_optimizers:
        optimization, oracle = [
            OPTIMIZERS[method](
                cfg, env, theta,
                derive_seed(cfg.cma_seed, user_index, run_index),
                features, grid,
            )
            for run_index, theta in enumerate((theta_hat, theta_true))
        ]
        optimization["oracle_parameters"] = oracle["parameters"]
        optimization["evaluation_rows"] = design_rows(
            env, optimization["parameters"], eval_seeds,
        )
        optimization["oracle_rows"] = design_rows(
            env, oracle["parameters"], eval_seeds,
        )
        controllers[method] = optimization
    return controllers


def optimize_all_users(cfg, env, data, theta_hats):
    features = feature_source(cfg, env, data)
    candidates = controller_grid(env, cfg.n_controller_candidates)
    if features[0] == "online":
        grid = controller_bank(cfg, env, candidates, features[1])
    else:
        grid = {
            "candidates": candidates,
            "features": np.stack([
                controller_features(env, features, controller)
                for controller in candidates
            ]),
        }
    names = data["user_names"]
    with ProcessPoolExecutor(
        max_workers=min(len(names), MAX_PARALLEL_USERS)
    ) as executor:
        futures = [
            executor.submit(
                optimize_controllers,
                cfg, env, theta_hats[name], data["theta_true"][i],
                i, features, grid,
            )
            for i, name in enumerate(names)
        ]
        return {name: future.result() for name, future in zip(names, futures)}
