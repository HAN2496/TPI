import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


MAX_PARALLEL_USERS = 15


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def derive_seed(base_seed, *keys):
    sequence = np.random.SeedSequence([base_seed, *keys])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def sampled_labels(probabilities, rng):
    return (rng.random(len(probabilities)) < probabilities).astype(np.int8)


def stack_metadata(rows):
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("Scenario metadata keys changed between rollouts")
    return {
        name: np.asarray([row[name] for row in rows])
        for name in keys
    }


def feedback_data(cfg, env, user_name, theta, user_index):
    controllers = []
    design_rows = []
    rewards = []
    metadata = []

    for scenario_index in range(cfg.n_feedback_scenarios):
        controller_rng = np.random.default_rng(derive_seed(
            cfg.rollout_seed, 0, user_index, scenario_index,
        ))
        controller = env.sample_controller(controller_rng)
        env_seed = derive_seed(
            cfg.rollout_seed, 1, user_index, scenario_index,
        )
        trajectory = env.rollout(controller, env_seed)
        trajectory.meta["user"] = user_name
        trajectory.meta["phase"] = "feedback"
        trajectory.meta["scenario_index"] = scenario_index
        design_row = env.design_row(trajectory)

        controllers.append(controller)
        design_rows.append(design_row)
        rewards.append(env.episode_reward(theta, design_row))
        metadata.append(env.scenario_metadata(trajectory))

    Z = np.stack(design_rows)
    rewards = np.asarray(rewards)
    logits = Z @ theta
    probabilities = sigmoid(logits)
    labels = sampled_labels(
        probabilities,
        np.random.default_rng(derive_seed(cfg.feedback_seed, 0, user_index)),
    )
    expected_good = float(probabilities.mean())
    observed_good = float(labels.mean())

    return {
        "controller": np.stack(controllers),
        "Z": Z,
        "rewards": rewards,
        "logits": logits,
        "probabilities": probabilities,
        "labels": labels,
        "expected_good": expected_good,
        "observed_good": observed_good,
        "metadata": stack_metadata(metadata),
    }


def collect_user_feedback(cfg, env, user_name, theta, user_index):
    return feedback_data(cfg, env, user_name, theta, user_index)


def generate_data(cfg, env):
    total_users = cfg.n_train_users + cfg.n_test_users
    user_names, theta_true = env.sample_users(
        total_users,
        seed=cfg.user_population_seed,
    )
    train_names = user_names[:cfg.n_train_users]
    test_names = user_names[cfg.n_train_users:]
    roles = {
        name: "train" if i < cfg.n_train_users else "test"
        for i, name in enumerate(user_names)
    }

    started = time.time()
    with ProcessPoolExecutor(
        max_workers=min(total_users, MAX_PARALLEL_USERS)
    ) as executor:
        futures = [
            executor.submit(
                collect_user_feedback,
                cfg,
                env,
                name,
                theta_true[i],
                i,
            )
            for i, name in enumerate(user_names)
        ]
        feedback = {
            name: future.result()
            for name, future in zip(user_names, futures)
        }

    for name in user_names:
        item = feedback[name]
        sampled_range = ", ".join(
            f"{parameter}={item['controller'][:, j].min():.1f}"
            f"..{item['controller'][:, j].max():.1f}"
            for j, parameter in enumerate(env.controller_names)
        )
        print(
            f"  [{roles[name]}] {name}: {sampled_range}  "
            f"expected/observed good="
            f"{item['expected_good']:.3f}/{item['observed_good']:.3f}"
        )

    print(
        f"[feedback data] {total_users} users x {cfg.n_feedback_scenarios} "
        f"independent scenarios  elapsed={time.time() - started:.1f}s"
    )
    return {
        "user_names": user_names,
        "theta_true": theta_true,
        "train_names": train_names,
        "test_names": test_names,
        "roles": roles,
        "feedback": feedback,
    }
