import csv

import joblib
import numpy as np


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_artifacts(run, env, data, population, result):
    names = data["user_names"]
    feedback = [data["feedback"][name] for name in names]
    arrays = {
        "user_names": np.asarray(names),
        "user_roles": np.asarray([data["roles"][name] for name in names]),
        "theta_true": data["theta_true"],
        "population_mean": env.population_mean,
        "population_cov": env.population_cov,
        "parameter_names": np.asarray(env.parameter_names),
        "controller_names": np.asarray(env.controller_names),
        "feedback_controller": np.stack([
            item["controller"] for item in feedback
        ]),
        "feedback_Z": np.stack([item["Z"] for item in feedback]),
        "feedback_rewards": np.stack([
            item["rewards"] for item in feedback
        ]),
        "feedback_logits": np.stack([
            item["logits"] for item in feedback
        ]),
        "feedback_probabilities": np.stack([
            item["probabilities"] for item in feedback
        ]),
        "feedback_labels": np.stack([
            item["labels"] for item in feedback
        ]),
    }
    metadata_names = tuple(feedback[0]["metadata"])
    for metadata_name in metadata_names:
        arrays[f"feedback_metadata_{metadata_name}"] = np.stack([
            item["metadata"][metadata_name] for item in feedback
        ])
    np.savez_compressed(run.dir / "preference_data.npz", **arrays)

    joblib.dump(
        population.state_dict(),
        run.dir / "fully_bayesian_state.joblib",
    )
    np.savez_compressed(
        run.dir / "personalized_posteriors.npz",
        user_names=np.asarray(names),
        theta_samples=np.stack([
            result["users"][name]["theta_samples"] for name in names
        ]),
    )
    if "cmaes" in result["optimizer_names"]:
        np.savez_compressed(
            run.dir / "cmaes_optimization.npz",
            user_names=np.asarray(names),
            generation=np.stack([
                result["users"][name]["controllers"]["cmaes"]["generation"]
                for name in names
            ]),
            controller=np.stack([
                result["users"][name]["controllers"]["cmaes"]["controller"]
                for name in names
            ]),
            estimated_reward=np.stack([
                result["users"][name]["controllers"]["cmaes"]["reward"]
                for name in names
            ]),
            final_controller=np.stack([
                result["users"][name]["controllers"]["cmaes"]["parameters"]
                for name in names
            ]),
        )

    user_rows = []
    for i, name in enumerate(names):
        item = data["feedback"][name]
        evaluation = result["users"][name]
        row = {
            "user": name,
            "role": evaluation["role"],
            "expected_good": item["expected_good"],
            "observed_good": item["observed_good"],
        }
        for method, optimization in evaluation["controllers"].items():
            row[f"{method}_controller_error"] = optimization[
                "controller_error"
            ]
            row[f"{method}_regret"] = optimization["regret"]
            row[f"{method}_true_reward_mean"] = optimization[
                "true_reward_mean"
            ]
            row[f"{method}_true_reward_std"] = optimization[
                "true_reward_std"
            ]
            row[f"{method}_estimated_reward_mean"] = optimization[
                "estimated_reward_mean"
            ]
        for j, parameter_name in enumerate(env.parameter_names):
            row[f"true_{parameter_name}"] = data["theta_true"][i, j]
            row[f"estimated_{parameter_name}"] = evaluation["theta_hat"][j]
            row[f"posterior_sd_{parameter_name}"] = evaluation["theta_sd"][j]
        for j, controller_name in enumerate(env.controller_names):
            row[f"true_best_{controller_name}"] = evaluation[
                "true_best_controller"
            ][j]
            for method, optimization in evaluation["controllers"].items():
                row[f"{method}_best_{controller_name}"] = optimization[
                    "parameters"
                ][j]
                row[f"{method}_oracle_{controller_name}"] = optimization[
                    "oracle_parameters"
                ][j]
        user_rows.append(row)
    write_csv(run.dir / "users.csv", user_rows)

    controller_rows = []
    for name in names:
        evaluation = result["users"][name]
        candidate = evaluation["candidate"]
        for j, controller in enumerate(
            evaluation["evaluation"]["candidates"]
        ):
            row = {
                "user": name,
                "role": evaluation["role"],
                "n_random_scenarios": len(
                    evaluation["evaluation"]["controller"]
                ) // len(evaluation["evaluation"]["candidates"]),
                "true_logit_mean": candidate["true_mean"][j],
                "true_logit_std": candidate["true_std"][j],
                "estimated_logit_mean": candidate["estimate_mean"][j],
                "estimated_logit_q05": candidate["estimate_low"][j],
                "estimated_logit_q95": candidate["estimate_high"][j],
                "expected_good_ratio": candidate["expected_good"][j],
                "observed_good_ratio": candidate["observed_good"][j],
            }
            for k, controller_name in enumerate(env.controller_names):
                row[controller_name] = controller[k]
            controller_rows.append(row)
    write_csv(run.dir / "controller_summary.csv", controller_rows)


def record_metrics(run, env, data, fit_stats, result):
    for role, rows in result["metrics_by_role"].items():
        names = data[f"{role}_names"]
        for name, metrics in zip(names, rows):
            run.metrics[f"{role}/{name}"] = metrics

    expected_good = np.asarray([
        data["feedback"][name]["expected_good"]
        for name in data["user_names"]
    ])
    observed_good = np.asarray([
        data["feedback"][name]["observed_good"]
        for name in data["user_names"]
    ])
    true_best = np.stack([
        result["users"][name]["true_best_controller"]
        for name in data["user_names"]
    ])
    data_metrics = {
        "n_train_users": len(data["train_names"]),
        "n_test_users": len(data["test_names"]),
        "n_feedback_scenarios_per_user": len(
            data["feedback"][data["user_names"][0]]["labels"]
        ),
        "expected_good_min": float(expected_good.min()),
        "expected_good_max": float(expected_good.max()),
        "observed_good_min": float(observed_good.min()),
        "observed_good_max": float(observed_good.max()),
    }
    for j, controller_name in enumerate(env.controller_names):
        data_metrics[f"true_preferred_{controller_name}_min"] = float(
            true_best[:, j].min()
        )
        data_metrics[f"true_preferred_{controller_name}_max"] = float(
            true_best[:, j].max()
        )
    run.metrics["data"] = data_metrics
    run.metrics["fully_bayesian"] = fit_stats
    run.metrics["population"] = result["population_metrics"]
    run.metrics["train_mean"] = result["mean_metrics"]["train"]
    run.metrics["test_mean"] = result["mean_metrics"]["test"]

    print(f"[train mean] {result['mean_metrics']['train']}")
    print(f"[test mean] {result['mean_metrics']['test']}")
    print(f"[population] {result['population_metrics']}")
