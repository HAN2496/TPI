import argparse
import csv
import json

import numpy as np
import torch
from scipy.optimize import minimize

from .state_space import calibrate, metrics
from .models import estimate, model_spec
from .run import OUTPUT, data, training_split, write_csv


STAGES = ((3.0, 100.0), (6.0, 300.0), (12.0, 1000.0), (20.0, 5000.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    cfg, x, y, ids, test = data()
    az, bounce = x[:, :, 1], y[:, :, 0]
    train, validation, validation_driver = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]
    start, base_bounds = model_spec("matern32")
    checkpoint = torch.load(OUTPUT / "matern32_lstm.pt", map_location="cpu", weights_only=False)
    current = np.asarray(checkpoint["matern32_parameters"], float)

    def objective(p):
        velocity = estimate("matern32", az[fit_index], p, cfg.fs)[..., 1]
        gain, offset = calibrate(velocity, bounce[fit_index])
        error = gain * velocity + offset - bounce[fit_index]
        return np.sqrt(np.mean(error**2)) / bounce[fit_index].std()

    def validation_score(p):
        velocity = estimate("matern32", az[train], p, cfg.fs)[..., 1]
        gain, offset = calibrate(velocity, bounce[train])
        prediction = gain * estimate("matern32", az[validation], p, cfg.fs)[..., 1] + offset
        corr, rmse, lag = metrics(bounce[validation], prediction, cfg.fs)
        nrmse = np.sqrt(np.mean((prediction - bounce[validation]) ** 2)) / bounce[validation].std()
        return gain, offset, nrmse, np.median(corr), np.median(rmse), 1000 * np.median(lag)

    rows, fitted = [], current
    if args.evaluate_only:
        with (OUTPUT / "matern32_bound_sensitivity.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["success"] = row["success"].lower() == "true"
            for key in set(row) - {"start", "success"}:
                row[key] = float(row[key])
    else:
        for stage, (f_max, lambda_max) in enumerate(STAGES):
            bounds = list(base_bounds)
            bounds[0] = (bounds[0][0], f_max)
            bounds[-1] = (bounds[-1][0], np.log(lambda_max))
            warm = np.clip(fitted, [v[0] for v in bounds], [v[1] for v in bounds])
            alternate = np.array(start, copy=True)
            alternate[0] = min(0.7 * f_max, f_max - 1e-4)
            alternate[-1] = np.log(min(0.3 * lambda_max, lambda_max - 1e-4))
            candidates = []
            for start_name, initial in (("warm", warm), ("alternate", alternate)):
                result = minimize(objective, initial, method="Powell", bounds=bounds,
                                  options={"maxiter": 100, "ftol": 1e-6, "xtol": 1e-5})
                candidates.append((result.fun, start_name, result))
                print(stage, f_max, lambda_max, start_name, result.success, result.nit, result.nfev,
                      result.fun, result.x[0], np.exp(result.x[-1]), flush=True)
            _, selected_start, result = min(candidates, key=lambda value: value[0])
            fitted = result.x
            gain, offset, val_nrmse, val_corr, val_rmse, val_lag = validation_score(fitted)
            rows.append(dict(stage=stage, f_max=f_max, lambda_max=lambda_max, start=selected_start,
                             success=bool(result.success), iterations=result.nit, evaluations=result.nfev,
                             train_objective=result.fun, f=fitted[0], zeta=fitted[1],
                             q_v=np.exp(fitted[2]), sigma_d2=np.exp(fitted[3]), r=np.exp(fitted[4]),
                             lambda_=np.exp(fitted[5]), f_fraction=fitted[0] / f_max,
                             lambda_fraction=np.exp(fitted[5]) / lambda_max, gain=gain, offset=offset,
                             validation_nrmse=val_nrmse, validation_corr_median=val_corr,
                             validation_rmse_median=val_rmse, validation_lag_median_ms=val_lag))
            write_csv(OUTPUT / "matern32_bound_sensitivity.csv", rows)
            print("validation", rows[-1], flush=True)

    converged = [row for row in rows if row["success"] and row["f_fraction"] < .95
                 and row["lambda_fraction"] < .95]
    selected = min(converged or rows, key=lambda row: row["validation_nrmse"])
    def parameters(row):
        return np.array([row["f"], row["zeta"], np.log(row["q_v"]), np.log(row["sigma_d2"]),
                         np.log(row["r"]), np.log(row["lambda_"])])

    selected_p = parameters(selected)

    def test_score(name, p):
        velocity = estimate("matern32", az[train], p, cfg.fs)[..., 1]
        gain, offset = calibrate(velocity, bounce[train])
        prediction = gain * estimate("matern32", az[test], p, cfg.fs)[..., 1] + offset
        corr, rmse, lag = metrics(bounce[test], prediction, cfg.fs)
        return dict(name=name, gain=gain, offset=offset, corr_p10=np.percentile(corr, 10),
                    corr_median=np.median(corr), corr_p90=np.percentile(corr, 90),
                    rmse_p10=np.percentile(rmse, 10), rmse_median=np.median(rmse),
                    rmse_p90=np.percentile(rmse, 90), lag_median_ms=1000 * np.median(lag))

    current_validation = validation_score(current)
    summary = dict(validation_driver=validation_driver, train_episodes=len(train),
                   validation_episodes=len(validation), test_episodes=int(test.sum()),
                   current_validation=dict(nrmse=current_validation[2], corr_median=current_validation[3],
                                           rmse_median=current_validation[4],
                                           lag_median_ms=current_validation[5]),
                   selected_stage=int(selected["stage"]), selected_parameters=selected_p.tolist(),
                   validation=rows,
                   test=[test_score("current", current),
                         test_score("same_bound_refit", parameters(rows[0])),
                         test_score("expanded_converged", selected_p)])
    (OUTPUT / "matern32_bound_sensitivity.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                             encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
