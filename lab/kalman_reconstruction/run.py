import argparse
import csv
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar

from loader import Dataset
from run_reconstruction import Config, load

from .state_space import GRAVITY, bandpass, calibrate, highpass, metrics, waveform_metrics
from .models import PITCH, ROAD_STATES, SPECS, estimate, estimate_half, estimate_kinematic, estimate_pitch
from .models import half_car, model, model_spec, quarter_car
from .iri import spatial_results
from .viz import plot_grid, plot_hybrid_detail, plot_lstm_kf_detail, plot_metric_grid, plot_model_free
from .viz import plot_spatial, plot_waveforms
from .hybrid import ResidualLSTM, fit as fit_residual_lstm, kf_features, normalize, predict as predict_residual
from .model_free import MODELS as MODEL_FREE_MODELS
from .model_free import build as build_model_free, causal_check, fit as fit_model_free
from .model_free import normalize as normalize_model_free, predict as predict_model_free


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "outputs"
BASE = ("rw", "ou", "matern32", "matern52", "qc2")


def data():
    cfg = Config()
    x, y, ids = load(Dataset(ROOT / "datasets"), cfg)
    test = np.array([value.split()[0] in cfg.test for value in ids])
    return cfg, x, y, ids, test


def training_split(ids, test):
    drivers = np.array([value.split()[0] for value in ids])
    names, counts = np.unique(drivers[~test], return_counts=True)
    validation_driver = names[np.argmin(np.abs(counts - .1 * np.sum(~test)))]
    validation = np.flatnonzero((drivers == validation_driver) & ~test)
    train = np.flatnonzero(~test & (drivers != validation_driver))
    return train, validation, validation_driver


def fit_model(name, x, y, fit_index, fs, maxiter=40, warm=None):
    spec = SPECS[name]
    target = y[fit_index, :, spec["target"]]
    best = [np.inf, None]

    def objective(p):
        output = spec["run"](x[fit_index], p, fs)[..., spec["output"]]
        gain, offset = calibrate(output, target)
        cost = np.sqrt(np.mean((gain * output + offset - target) ** 2)) / target.std()
        cost = cost if np.isfinite(cost) else 1e6
        if cost < best[0]:
            best[:] = cost, p.copy()
        return cost

    start, bounds = model_spec(name)
    starts = [start] if warm is None else [start, np.clip(warm, [b[0] for b in bounds], [b[1] for b in bounds])]
    fit = min([minimize(objective, s, method="Powell", bounds=bounds, options={"maxiter": maxiter}) for s in starts],
              key=lambda value: value.fun)
    fit.fun, fit.x = best
    return fit


def evaluate(raw, target, test, fs):
    gain, offset = calibrate(raw[~test], target[~test])
    pred = gain * raw[test] + offset
    corr, rmse, lag = metrics(target[test], pred, fs)
    return dict(gain=gain, offset=offset, pred=pred, corr=corr, rmse=rmse, lag=lag)


def metric_row(name, execution, loss, p, result, roll=None):
    corr, rmse, lag = result["corr"], result["rmse"], result["lag"]
    cp, rp = np.percentile(corr, [10, 50, 90]), np.percentile(rmse, [10, 50, 90])
    latent = np.exp(p[5]) if name in ("ou", "matern32", "matern52") else np.nan
    vehicle = name.startswith(("qc2", "hc8"))
    return dict(model=name, execution=execution, loss=loss, f=p[0], zeta=p[1], lambda_=latent,
                mass_ratio=20 / 3 if vehicle else np.nan, kt_ks=np.exp(p[2]) if vehicle else np.nan,
                roll_inertia_ratio=np.exp(p[3]) if name.startswith("hc8") else np.nan,
                corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2], rmse_p10=rp[0],
                rmse_median=rp[1], rmse_p90=rp[2], lag_median_ms=1000 * np.median(lag),
                roll_corr_median=np.median(roll["corr"]) if roll else np.nan,
                roll_rmse_median=np.median(roll["rmse"]) if roll else np.nan,
                roll_lag_median_ms=1000 * np.median(roll["lag"]) if roll else np.nan)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def base_panels(name, state, state_space, fit, bounce, az, road=None):
    panels = [("v_s -> Bounce" if name == "qc2" else "v -> Bounce",
               fit["gain"] * state[:, 1] + fit["offset"], bounce),
              ("a_z posterior", (state @ state_space.H.T)[:, 0], (az - 1) * GRAVITY)]
    labels = {"rw": ("z", "d"), "ou": ("z", "d"), "matern32": ("z", "d", "d_dot"),
              "matern52": ("z", "d", "d_dot", "d_ddot"), "qc2": ("z_s", "z_u", "v_u")}
    indices = {"rw": (0, 2), "ou": (0, 2), "matern32": (0, 2, 3),
               "matern52": (0, 2, 3, 4), "qc2": (0, 2, 3)}
    panels += [(label, state[:, index], None) for label, index in zip(labels[name], indices[name])]
    if road is not None:
        panels.append(("road input", road, None))
    return panels


def classical_plots(cfg, az, lat, bounce, roll, ids, results, fits):
    median = plot_waveforms(bounce, results, ids, cfg.fs, OUTPUT / "kf_models.png", "rw")
    rows = []
    for name in BASE:
        value = estimate(name, az[median:median + 1], fits[name]["p"], cfg.fs, road=name == "qc2")
        state, road = value if name == "qc2" else (value, None)
        rows.append((name, base_panels(name, state[0], model(name, fits[name]["p"], cfg.fs), fits[name],
                                              bounce[median], az[median], road[0, :, 0] if road is not None else None)))
    plot_grid(rows, ids[median], cfg.fs, OUTPUT / "kf_states_median.png", "KF state estimates")

    qp, hp, qsp, hsp = (fits[name]["p"] for name in ("qc2", "hc8", "qc2_rts", "hc8_rts"))
    qs, qr = estimate("qc2", az[median:median + 1], qp, cfg.fs, road=True)
    qss = estimate("qc2", az[median:median + 1], qsp, cfg.fs, smooth=True)
    hs, hr = estimate_half(az[median:median + 1], lat[median:median + 1], hp, cfg.fs, road=True, full=True)
    hss = estimate_half(az[median:median + 1], lat[median:median + 1], hsp, cfg.fs, smooth=True, full=True)
    qs, qr, qss, hs, hr, hss = qs[0], qr[0], qss[0], hs[0], hr[0], hss[0]
    az_obs, lat_obs = (az[median] - 1) * GRAVITY, highpass(lat[median:median + 1] * GRAVITY, .5, cfg.fs)[0]
    qc = lambda name, state, state_space: [("v_s -> Bounce", fits[name]["gain"] * state[:, 1] + fits[name]["offset"], bounce[median]),
                                    ("a_z", (state @ state_space.H.T)[:, 0], az_obs), ("z_s", state[:, 0], None),
                                    ("z_u", state[:, 2], None), ("v_u", state[:, 3], None)]
    hc = lambda name, state, state_space: [("v_s -> Bounce", fits[name]["gain"] * state[:, 1] + fits[name]["offset"], bounce[median]),
                                    ("theta_dot -> Roll", fits[name]["roll_gain"] * state[:, 3] + fits[name]["roll_offset"], roll[median]),
                                    ("a_z", state @ state_space.H[0], az_obs), ("a_lat", state @ state_space.H[1], lat_obs),
                                    ("z_s", state[:, 0], None), ("theta", state[:, 2], None)]
    extension_rows = [("qc2 online", qc("qc2", qs, quarter_car(qp, cfg.fs)) + [("road", qr[:, 0], None)]),
                      ("qc2 RTS", qc("qc2_rts", qss, quarter_car(qsp, cfg.fs))),
                      ("hc8 online", hc("hc8", hs, half_car(hp, cfg.fs)) +
                       [("road left", hr[:, 0], None), ("road right", hr[:, 1], None)]),
                      ("hc8 RTS", hc("hc8_rts", hss, half_car(hsp, cfg.fs)))]
    plot_grid(extension_rows, ids[median], cfg.fs, OUTPUT / "kf_extensions_median.png", "QC/HC estimates")


def run_classical(loaded=None):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(~test)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]
    az, lat, speed = x[:, :, 1], x[:, :, 3], x[:, :, 5:9].mean(2)
    bounce, roll = y[:, :, 0], y[:, :, 1]
    fits, results, rows = {}, {}, []

    for name in BASE:
        fit = fit_model(name, x, y, fit_index, cfg.fs)
        result = evaluate(estimate(name, az, fit.x, cfg.fs)[..., 1], bounce, test, cfg.fs)
        fits[name], results[name] = dict(p=fit.x, **{k: result[k] for k in ("gain", "offset")}), result
        rows.append(metric_row(name, "online", fit.fun, fit.x, result))
        print(f"{name:9s} corr={np.median(result['corr']):.3f} rmse={np.median(result['rmse']):.3f}")

    qp = fits["qc2"]["p"]
    hc_parameters = lambda value, base=qp: np.array([base[0], base[1], base[2], value[0], base[3], base[4], value[1]])
    def hc_objective(value, smooth=False, base=qp):
        output = estimate_half(az[fit_index], lat[fit_index], hc_parameters(value, base), cfg.fs, smooth)
        gain, offset = calibrate(output[..., 1], roll[fit_index])
        return np.sqrt(np.mean((gain * output[..., 1] + offset - roll[fit_index]) ** 2)) / roll[fit_index].std()
    hc_fit = minimize(hc_objective, [np.log(1960 / 600), -3], method="Powell",
                      bounds=[(np.log(.5), np.log(20)), (-12, 4)], options={"maxiter": 40})
    hp = hc_parameters(hc_fit.x)
    hc_output = estimate_half(az, lat, hp, cfg.fs)
    hc_bounce, hc_roll = evaluate(hc_output[..., 0], bounce, test, cfg.fs), evaluate(hc_output[..., 1], roll, test, cfg.fs)
    fits["hc8"] = dict(p=hp, gain=hc_bounce["gain"], offset=hc_bounce["offset"],
                       roll_gain=hc_roll["gain"], roll_offset=hc_roll["offset"])
    results["hc8"] = hc_bounce
    rows.append(metric_row("hc8", "online", hc_fit.fun, hp, hc_bounce, hc_roll))

    def rts_objective(p):
        velocity = estimate("qc2", az[fit_index], p, cfg.fs, smooth=True)[..., 1]
        gain, offset = calibrate(velocity, bounce[fit_index])
        return np.sqrt(np.mean((gain * velocity + offset - bounce[fit_index]) ** 2)) / bounce[fit_index].std()
    start = qp.copy()
    start[3] = np.clip(start[3] + 2, -20, -4)
    qfit = minimize(rts_objective, start, method="Powell", bounds=model_spec("qc2")[1], options={"maxiter": 40})
    qresult = evaluate(estimate("qc2", az, qfit.x, cfg.fs, smooth=True)[..., 1], bounce, test, cfg.fs)
    fits["qc2_rts"], results["qc2_rts"] = dict(p=qfit.x, gain=qresult["gain"], offset=qresult["offset"]), qresult
    rows.append(metric_row("qc2_rts", "offline", qfit.fun, qfit.x, qresult))

    hc_smooth_parameters = lambda value: hc_parameters(value, qfit.x)
    def hc_smooth_objective(value):
        output = estimate_half(az[fit_index], lat[fit_index], hc_smooth_parameters(value), cfg.fs, smooth=True)
        gain, offset = calibrate(output[..., 1], roll[fit_index])
        return np.sqrt(np.mean((gain * output[..., 1] + offset - roll[fit_index]) ** 2)) / roll[fit_index].std()
    hsfit = minimize(hc_smooth_objective, hc_fit.x, method="Powell",
                     bounds=[(np.log(.5), np.log(20)), (-12, 4)], options={"maxiter": 40})
    hsp = hc_smooth_parameters(hsfit.x)
    hsout = estimate_half(az, lat, hsp, cfg.fs, smooth=True)
    hs_bounce, hs_roll = evaluate(hsout[..., 0], bounce, test, cfg.fs), evaluate(hsout[..., 1], roll, test, cfg.fs)
    fits["hc8_rts"] = dict(p=hsp, gain=hs_bounce["gain"], offset=hs_bounce["offset"],
                           roll_gain=hs_roll["gain"], roll_offset=hs_roll["offset"])
    results["hc8_rts"] = hs_bounce
    rows.append(metric_row("hc8_rts", "offline", hsfit.fun, hsp, hs_bounce, hs_roll))

    classical_plots(cfg, az[test], lat[test], bounce[test], roll[test], ids[test], results, fits)
    _, qc_road = estimate("qc2", az[test], qp, cfg.fs, road=True)
    _, hc_road = estimate_half(az[test], lat[test], hp, cfg.fs, road=True)
    qc_road = qc_road[..., 0]
    spatial, iri_rows = spatial_results(speed[test], qc_road, hc_road, ids[test], cfg.fs)
    np.savez_compressed(OUTPUT / "qc2_road_posterior.npz", ids=ids[test], road=qc_road.astype(np.float32), fs=cfg.fs)
    np.savez_compressed(OUTPUT / "hc8_road_posterior.npz", ids=ids[test], left=hc_road[..., 0].astype(np.float32),
                        right=hc_road[..., 1].astype(np.float32), fs=cfg.fs)
    np.savez_compressed(OUTPUT / "kf_spatial_iri.npz", ids=ids[test], spatial_step_m=.1, iri_window_m=40, **spatial)
    np.savez(OUTPUT / "kf_parameters.npz", qc2_online=qp, hc8_online=hp, qc2_rts=qfit.x, hc8_rts=hsp)
    write_csv(OUTPUT / "kf_metrics.csv", rows)
    write_csv(OUTPUT / "kf_iri_summary.csv", iri_rows)
    plot_spatial(speed[test], spatial, results, ids[test], cfg.fs, OUTPUT / "kf_spatial_iri_median.png")
    print(f"episodes train={len(train)} test={test.sum()} outputs={OUTPUT}")
    return loaded or (cfg, x, y, ids, test)


def run_pitch(loaded=None):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(~test)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]
    pitch = y[:, :, 2]
    names = ("pitch_hc", "pitch_hc_ou", "pitch_hc_osc", "pitch_road", "pitch_road_osc",
             "pitch_delay", "pitch_delay_osc")
    fits, results, rows = {}, {}, []
    for name in names:
        started = time.perf_counter()
        warm = None
        if name.endswith(("_ou", "_osc")):
            tail = model_spec(name)[0][14:]
            tail[-1] = -8.0
            warm = np.r_[fits[name.rsplit("_", 1)[0]].x[:14], tail]
        fit = fit_model(name, x, y, fit_index, cfg.fs, maxiter=300, warm=warm)
        result = evaluate(estimate_pitch(name, x, fit.x, cfg.fs)[..., 3], pitch, test, cfg.fs)
        fits[name], results[name] = fit, result
        p = fit.x
        cp, rp = np.percentile(result["corr"], [10, 50, 90]), np.percentile(result["rmse"], [10, 50, 90])
        dist = dict(lambda_d=np.nan, omega_d=np.nan, zeta_d=np.nan, sigma_d2=np.nan)
        if name.endswith("_ou"):
            dist |= dict(lambda_d=np.exp(p[14]), sigma_d2=np.exp(p[15]))
        if name.endswith("_osc"):
            dist |= dict(omega_d=np.exp(p[14]), zeta_d=p[15], sigma_d2=np.exp(p[16]))
        rows.append(dict(model=name, execution="online", loss=fit.fun, f=p[0], zeta=p[1], eps=p[2],
                         pitch_inertia_ratio=np.exp(p[3]), kt_ks=np.exp(p[4]), g_u=p[5], beta=p[6],
                         lambda_front=p[7], lambda_rear=p[8], **dist,
                         gain=result["gain"], offset=result["offset"],
                         corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2], rmse_p10=rp[0], rmse_median=rp[1],
                         rmse_p90=rp[2], lag_median_ms=1000 * np.median(result["lag"]),
                         fit_seconds=time.perf_counter() - started))
        print(f"{name:14s} corr={cp[1]:.3f} rmse={rp[1]:.3f} lag={rows[-1]['lag_median_ms']:.0f} ms "
              f"f={p[0]:.2f} zeta={p[1]:.2f} gain={result['gain']:.3f} ({rows[-1]['fit_seconds']:.0f}s)")

    median = plot_waveforms(pitch[test], results, ids[test], cfg.fs, OUTPUT / "pitch_models.png", names[-1])
    index = np.flatnonzero(test)[median]
    grid_rows = [(name, pitch_panels(name, fits[name].x, results[name], x, index, pitch, cfg.fs))
                 for name in names]
    plot_grid(grid_rows, ids[index], cfg.fs, OUTPUT / "pitch_states_median.png", "Pitch KF states")
    plot_metric_grid({name: waveform_metrics(pitch[test], results[name]["pred"], cfg.fs) for name in names},
                     OUTPUT / "pitch_metric_grid.png")
    write_csv(OUTPUT / "pitch_metrics.csv", rows)
    np.savez(OUTPUT / "pitch_parameters.npz", **{name: fits[name].x for name in names})
    print(f"train={len(train)} test={test.sum()} outputs={OUTPUT}")


def pitch_panels(name, p, result, x, index, pitch, fs):
    dist_name, road, _, ax = PITCH[name]
    nroad = ROAD_STATES[road]
    state_space = model(name, p, fs)
    state = estimate_pitch(name, x[index:index + 1], p, fs)[0]
    front = x[index, :, 5:7].mean(1) / 3.6
    panels = [("theta_dot -> Pitch", result["gain"] * state[:, 3] + result["offset"], pitch[index]),
              ("a_z posterior", state @ state_space.H[2], (x[index, :, 1] - 1) * GRAVITY),
              ("front wheel posterior", state @ state_space.H[0], front),
              ("z_s", state[:, 0], None), ("theta", state[:, 2], None), ("v", state[:, 8], None)]
    if nroad:
        panels.append(("r_f (road)", state[:, 9], None))
    if nroad == 2:
        panels.append(("r_r (road)", state[:, 10], None))
    if dist_name != "none":
        panels.append(("d (latent pitch moment)", state[:, 9 + nroad], None))
    if ax:
        n_states = len(state_space.A)
        panels += [("a_x posterior", state @ state_space.H[3], x[index, :, 4] * GRAVITY),
                   ("a_b (long accel)", state[:, n_states - 3], None),
                   ("grade", state[:, n_states - 2], None)]
    return panels


def run_pitch2(loaded=None):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(~test)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]
    pitch = y[:, :, 2]
    stored = np.load(OUTPUT / "pitch_parameters.npz")
    anchor = "pitch_road_osc"
    params = {anchor: stored[anchor]}
    results, rows = {}, []

    def report(name, loss, seconds):
        result = results[name]
        cp = np.percentile(result["corr"], [10, 50, 90])
        rp = np.percentile(result["rmse"], [10, 50, 90])
        rows.append(dict(model=name, loss=loss, corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2],
                         rmse_p10=rp[0], rmse_median=rp[1], rmse_p90=rp[2],
                         lag_median_ms=1000 * np.median(result["lag"]), gain=result["gain"],
                         offset=result["offset"], fit_seconds=seconds,
                         parameters=" ".join(f"{value:.9g}" for value in params[name])))
        print(f"{name:14s} corr={cp[1]:.3f} rmse={rp[1]:.3f} loss={loss:.3f} ({seconds:.0f}s)")

    results[anchor] = evaluate(estimate_pitch(anchor, x, params[anchor], cfg.fs)[..., 3], pitch, test, cfg.fs)
    report(anchor, np.nan, 0.0)
    for name, parent, extension in (("pitch_tq", anchor, np.zeros(4)),
                                    ("pitch_ax", "pitch_tq", np.r_[np.log(20), .01, np.log(.2), -8., -10., -3.]),
                                    ("pitch_axou", "pitch_ax", np.r_[np.log(.01)]),
                                    ("pitch_eps", "pitch_ax", None)):
        started = time.perf_counter()
        warm = np.r_[params[parent], extension] if extension is not None else \
            np.r_[params[parent], params[parent][2]]
        if name == "pitch_axou":
            warm[9] += np.log(1 / (2 * .01))
        fit = fit_model(name, x, y, fit_index, cfg.fs, maxiter=300, warm=warm)
        params[name] = fit.x
        results[name] = evaluate(estimate_pitch(name, x, fit.x, cfg.fs)[..., 3], pitch, test, cfg.fs)
        report(name, fit.fun, time.perf_counter() - started)

    names = tuple(params)
    median = plot_waveforms(pitch[test], results, ids[test], cfg.fs, OUTPUT / "pitch2_models.png", names[-1])
    index = np.flatnonzero(test)[median]
    grid_rows = [(name, pitch_panels(name, params[name], results[name], x, index, pitch, cfg.fs))
                 for name in names]
    plot_grid(grid_rows, ids[index], cfg.fs, OUTPUT / "pitch2_states_median.png", "Pitch KF stage 2 states")
    plot_metric_grid({name: waveform_metrics(pitch[test], results[name]["pred"], cfg.fs) for name in names},
                     OUTPUT / "pitch2_metric_grid.png")
    write_csv(OUTPUT / "pitch2_metrics.csv", rows)
    np.savez(OUTPUT / "pitch2_parameters.npz", **params)
    print(f"train={len(train)} test={test.sum()} outputs={OUTPUT}")


def run_oscillator(loaded=None):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    az, bounce = x[:, :, 1], y[:, :, 0]
    train, _, _ = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]
    results, rows = {}, []
    for name in ("oscillator", "rw"):
        fitted = fit_model(name, x, y, fit_index, cfg.fs)
        velocity = estimate(name, az, fitted.x, cfg.fs)[..., 1]
        gain, offset = calibrate(velocity[train], bounce[train])
        prediction = gain * velocity + offset
        result = scored(prediction, bounce, test, cfg.fs)
        results[name] = result
        cp = np.percentile(result["corr"], [10, 50, 90])
        rp = np.percentile(result["rmse"], [10, 50, 90])
        rows.append(dict(model=name, corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2],
                         rmse_p10=rp[0], rmse_median=rp[1], rmse_p90=rp[2],
                         lag_median_ms=1000 * np.median(result["lag"]), gain=gain, offset=offset,
                         parameters=" ".join(f"{value:.9g}" for value in fitted.x)))
        print(f"{name:12s} corr={cp[1]:.3f} rmse={rp[1]:.3f} lag={1000*np.median(result['lag']):.1f} ms")
    plot_waveforms(bounce[test], results, ids[test], cfg.fs, OUTPUT / "oscillator_models.png", "rw")
    write_csv(OUTPUT / "oscillator_metrics.csv", rows)
    np.savez_compressed(OUTPUT / "oscillator_predictions.npz", ids=ids[test], target=bounce[test].astype(np.float32),
                        oscillator=results["oscillator"]["pred"].astype(np.float32),
                        rw=results["rw"]["pred"].astype(np.float32), fs=cfg.fs)
    print(f"train={len(train)} test={test.sum()} outputs={OUTPUT}")


def scored(prediction, target, test, fs):
    corr, rmse, lag = metrics(target[test], prediction[test], fs)
    return dict(pred=prediction[test], corr=corr, rmse=rmse, lag=lag)


def run_hybrid(args, loaded=None, kf_name="matern32"):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    az, bounce = x[:, :, 1], y[:, :, 0]
    train, validation, validation_driver = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]

    fitted = fit_model(kf_name, x, y, fit_index, cfg.fs)
    state_space = model(kf_name, fitted.x, cfg.fs)
    state = estimate(kf_name, az, fitted.x, cfg.fs)
    gain, offset = calibrate(state[train, :, 1], bounce[train])
    baseline = gain * state[..., 1] + offset
    residual = bounce - baseline
    feature = kf_features(state, az, state_space)
    feature, residual_norm, x_mean, x_std, y_mean, y_std = normalize(feature, residual, train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = ResidualLSTM(feature.shape[-1], args.hidden_size).to(device)
    best, epochs = fit_residual_lstm(network, feature, residual_norm.astype(np.float32), train, validation,
                                     device, args.epochs, args.batch_size)
    correction = predict_residual(network, feature, device, args.batch_size) * y_std + y_mean
    hybrid = baseline + correction
    hybrid_name = f"{kf_name}_lstm"
    results = {kf_name: scored(baseline, bounce, test, cfg.fs),
               hybrid_name: scored(hybrid, bounce, test, cfg.fs)}
    rows = []
    for name, result in results.items():
        cp, rp = np.percentile(result["corr"], [10, 50, 90]), np.percentile(result["rmse"], [10, 50, 90])
        rows.append(dict(model=name, corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2],
                         rmse_p10=rp[0], rmse_median=rp[1], rmse_p90=rp[2],
                         lag_median_ms=1000 * np.median(result["lag"])))
        print(f"{name:14s} corr={cp[1]:.3f} rmse={rp[1]:.3f}")
    plot_waveforms(bounce[test], results, ids[test], cfg.fs, OUTPUT / f"{hybrid_name}_models.png", kf_name)
    predicted_state = np.zeros_like(state)
    predicted_state[:, 1:] = state[:, :-1] @ state_space.A.T
    innovation = (az - 1) * GRAVITY - (predicted_state @ state_space.H.T)[..., 0]
    plot_hybrid_detail(bounce[test], baseline[test], hybrid[test], correction[test], innovation[test], ids[test],
                       results[kf_name]["corr"], cfg.fs, kf_name, OUTPUT / f"{hybrid_name}_median.png")
    write_csv(OUTPUT / f"{hybrid_name}_metrics.csv", rows)
    np.savez_compressed(OUTPUT / f"{hybrid_name}_predictions.npz", ids=ids[test],
                        baseline=baseline[test].astype(np.float32), hybrid=hybrid[test].astype(np.float32),
                        correction=correction[test].astype(np.float32), fs=cfg.fs)
    cpu_state = {name: value.detach().cpu() for name, value in network.state_dict().items()}
    checkpoint = dict(model=cpu_state, hidden_size=args.hidden_size, input_size=feature.shape[-1],
                      kf_model=kf_name, kf_parameters=fitted.x, gain=gain, offset=offset,
                      x_mean=x_mean, x_std=x_std, residual_mean=y_mean, residual_std=y_std,
                      validation_driver=validation_driver, validation_mse=best, epochs=epochs,
                      real_data_only=True)
    checkpoint[f"{kf_name}_parameters"] = fitted.x
    torch.save(checkpoint, OUTPUT / f"{hybrid_name}.pt")
    network.cpu()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f"train={len(train)} validation={len(validation)} ({validation_driver}) test={test.sum()} outputs={OUTPUT}")


def run_calibration_ablation(loaded=None):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    az, bounce = x[:, :, 1], y[:, :, 0]
    train, _, _ = training_split(ids, test)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]
    rows, predictions = [], {}

    for name in ("rw", "matern32"):
        start, bounds = model_spec(name)

        def loss(p, calibrated):
            velocity = estimate(name, az[fit_index], p, cfg.fs)[..., 1]
            gain, offset = calibrate(velocity, bounce[fit_index]) if calibrated else (1., 0.)
            return np.sqrt(np.mean((gain * velocity + offset - bounce[fit_index]) ** 2)) / bounce[fit_index].std()

        calibrated_fit = minimize(lambda p: loss(p, True), start, method="Powell", bounds=bounds,
                                   options={"maxiter": 40})
        candidates = [minimize(lambda p: loss(p, False), initial, method="Powell", bounds=bounds,
                               options={"maxiter": 40}) for initial in (start, calibrated_fit.x)]
        constrained_fit = min(candidates, key=lambda fit: fit.fun)

        calibrated_state = estimate(name, az, calibrated_fit.x, cfg.fs)[..., 1]
        gain, offset = calibrate(calibrated_state[train], bounce[train])
        variants = {
            "calibrated": (gain * calibrated_state + offset, calibrated_fit, gain, offset),
            "same_kf_raw": (calibrated_state, calibrated_fit, 1., 0.),
            "constrained": (estimate(name, az, constrained_fit.x, cfg.fs)[..., 1], constrained_fit, 1., 0.),
        }
        for variant, (prediction, fit, fitted_gain, fitted_offset) in variants.items():
            result = scored(prediction, bounce, test, cfg.fs)
            cp = np.percentile(result["corr"], [10, 50, 90])
            rp = np.percentile(result["rmse"], [10, 50, 90])
            key = f"{name}_{variant}"
            predictions[key] = result["pred"].astype(np.float32)
            rows.append(dict(model=name, variant=variant, train_loss=loss(fit.x, variant == "calibrated"),
                             gain=fitted_gain, offset=fitted_offset, f=fit.x[0], zeta=fit.x[1],
                             corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2],
                             rmse_p10=rp[0], rmse_median=rp[1], rmse_p90=rp[2],
                             lag_median_ms=1000 * np.median(result["lag"]),
                             parameters=" ".join(f"{value:.9g}" for value in fit.x)))
            print(f"{name:8s} {variant:12s} corr={cp[1]:.3f} rmse={rp[1]:.3f} "
                  f"gain={fitted_gain:.3f} offset={fitted_offset:.3f}")

    write_csv(OUTPUT / "calibration_ablation_metrics.csv", rows)
    np.savez_compressed(OUTPUT / "calibration_ablation_predictions.npz", ids=ids[test],
                        target=bounce[test].astype(np.float32), fs=cfg.fs, **predictions)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in zip(axes, ("corr_median", "rmse_median"),
                                 ("median correlation", "median RMSE")):
        positions = np.arange(3)
        for index, name in enumerate(("rw", "matern32")):
            values = [next(row[metric] for row in rows if row["model"] == name and row["variant"] == variant)
                      for variant in ("calibrated", "same_kf_raw", "constrained")]
            ax.bar(positions + (index - .5) * .36, values, width=.36, label=name.upper())
        ax.set_xticks(positions, ("calibrated", "same KF raw", "constrained"))
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.25)
    axes[0].legend()
    fig.suptitle("Affine calibration ablation on held-out test episodes")
    fig.tight_layout()
    fig.savefig(OUTPUT / "calibration_ablation.png", dpi=160)
    plt.close(fig)
    print(f"train={len(train)} test={test.sum()} outputs={OUTPUT}")


def run_lstm_kf(args, loaded=None):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, validation, validation_driver = training_split(ids, test)
    target = y[:, :, 0].astype(np.float32)
    acceleration = ((x[:, :, 1] - 1) * GRAVITY).astype(np.float64)
    filtered = bandpass(acceleration, .2, 25, cfg.fs)
    fit_index = train[np.linspace(0, len(train) - 1, min(300, len(train))).astype(int)]

    def kinematic_loss(log_ratio):
        state = estimate_kinematic(filtered[fit_index], cfg.fs, np.exp(log_ratio))
        gain, offset = calibrate(state[..., 1], target[fit_index])
        return np.sqrt(np.mean((gain * state[..., 1] + offset - target[fit_index]) ** 2)) / target[fit_index].std()

    kinematic_fit = minimize_scalar(kinematic_loss, bounds=(-8, 16), method="bounded",
                                    options={"maxiter": 30, "xatol": .02})
    process_var = np.exp(kinematic_fit.x)
    kinematic_state = estimate_kinematic(filtered, cfg.fs, process_var)
    gain, offset = calibrate(kinematic_state[train, :, 1], target[train])
    kinematic_prediction = gain * kinematic_state[..., 1] + offset

    features = acceleration[..., None].astype(np.float32)
    features, normalized_target, x_mean, x_std, y_mean, y_std = \
        normalize_model_free(features, target, train)
    network = build_model_free("lstm_online", 1, features.shape[1], args.hidden_size)
    causality_error = causal_check(network, 1)
    if causality_error > 1e-6:
        raise RuntimeError(f"LSTM failed causality check: {causality_error}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    validation_mse, best_epoch = fit_model_free(network, features, normalized_target, train, validation,
                                                 device, args.epochs, args.batch_size)
    lstm_prediction = predict_model_free(network, features, device, args.batch_size) * y_std + y_mean
    velocity_var = np.var(lstm_prediction[validation] - target[validation]) + 1e-8

    def fusion_loss(log_process_var):
        state = estimate_kinematic(filtered[validation], cfg.fs, np.exp(log_process_var), velocity=lstm_prediction[validation],
                                   velocity_gain=gain, velocity_offset=offset, velocity_var=velocity_var)
        prediction = gain * state[..., 1] + offset
        return np.sqrt(np.mean((prediction - target[validation]) ** 2)) / target[validation].std()

    fusion_fit = minimize_scalar(fusion_loss, bounds=(-8, 16), method="bounded",
                                 options={"maxiter": 30, "xatol": .02})
    fused_state = estimate_kinematic(filtered, cfg.fs, np.exp(fusion_fit.x), velocity=lstm_prediction,
                                     velocity_gain=gain, velocity_offset=offset, velocity_var=velocity_var)
    fused_prediction = gain * fused_state[..., 1] + offset
    results = {"kinematic_kf": scored(kinematic_prediction, target, test, cfg.fs),
               "imu_lstm": scored(lstm_prediction, target, test, cfg.fs),
               "lstm_kf": scored(fused_prediction, target, test, cfg.fs)}
    rows = []
    for name, result in results.items():
        cp = np.percentile(result["corr"], [10, 50, 90])
        rp = np.percentile(result["rmse"], [10, 50, 90])
        rows.append(dict(model=name, corr_p10=cp[0], corr_median=cp[1], corr_p90=cp[2],
                         rmse_p10=rp[0], rmse_median=rp[1], rmse_p90=rp[2],
                         lag_median_ms=1000 * np.median(result["lag"]),
                         process_var=process_var if name == "kinematic_kf" else np.exp(fusion_fit.x)
                         if name == "lstm_kf" else np.nan, velocity_var=velocity_var if name == "lstm_kf" else np.nan,
                         validation_mse=validation_mse if name != "kinematic_kf" else np.nan,
                         best_epoch=best_epoch if name != "kinematic_kf" else np.nan))
        print(f"{name:14s} corr={cp[1]:.3f} rmse={rp[1]:.3f}")
    plot_waveforms(target[test], results, ids[test], cfg.fs, OUTPUT / "kinematic_lstm_kf_models.png", "lstm_kf")
    plot_lstm_kf_detail(target[test], results, kinematic_state[test], fused_state[test], filtered[test], ids[test],
                        cfg.fs, OUTPUT / "kinematic_lstm_kf_median.png")
    write_csv(OUTPUT / "kinematic_lstm_kf_metrics.csv", rows)
    np.savez_compressed(OUTPUT / "kinematic_lstm_kf_predictions.npz", ids=ids[test], fs=cfg.fs,
                        recorded=target[test], kinematic=kinematic_prediction[test], lstm=lstm_prediction[test],
                        fused=fused_prediction[test], kinematic_state=kinematic_state[test].astype(np.float32),
                        fused_state=fused_state[test].astype(np.float32))
    cpu_state = {name: value.detach().cpu() for name, value in network.state_dict().items()}
    torch.save(dict(model=cpu_state, hidden_size=args.hidden_size, input_size=1, x_mean=x_mean, x_std=x_std,
                    y_mean=y_mean, y_std=y_std, kinematic_process_var=process_var,
                    fusion_process_var=np.exp(fusion_fit.x), acceleration_var=1., velocity_var=velocity_var,
                    gain=gain, offset=offset, validation_driver=validation_driver,
                    validation_mse=validation_mse, best_epoch=best_epoch, seed=args.seed,
                    real_data_only=True), OUTPUT / "kinematic_lstm_kf.pt")
    network.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"train={len(train)} validation={len(validation)} ({validation_driver}) test={test.sum()} outputs={OUTPUT}")


def run_model_free(args, loaded=None):
    cfg, x, y, ids, test = loaded or data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, validation, validation_driver = training_split(ids, test)
    features, target = x.astype(np.float32), y[:, :, 0].astype(np.float32)
    features, target_norm, x_mean, x_std, y_mean, y_std = normalize_model_free(features, target, train)
    requested = tuple(value.strip() for value in args.models.split(",") if value.strip())
    unknown = set(requested) - set(MODEL_FREE_MODELS)
    if unknown:
        raise ValueError(f"unknown model-free models: {sorted(unknown)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, rows, checkpoints, predictions = {}, [], {}, {}
    for name in requested:
        network = build_model_free(name, features.shape[-1], features.shape[1], args.hidden_size)
        causality_error = causal_check(network, features.shape[-1]) if name.endswith("online") else np.nan
        if name.endswith("online") and causality_error > 1e-6:
            raise RuntimeError(f"{name} failed causality check: {causality_error}")
        parameters = sum(value.numel() for value in network.parameters())
        batch_size = min(args.batch_size, 8 if "transformer" in name else 16 if "unet" in name else args.batch_size)
        network = network.to(device)
        started = time.perf_counter()
        val_mse, best_epoch = fit_model_free(network, features, target_norm, train, validation, device,
                                              args.epochs, batch_size)
        seconds = time.perf_counter() - started
        prediction = predict_model_free(network, features[test], device, batch_size) * y_std + y_mean
        corr, rmse, lag = metrics(target[test], prediction, cfg.fs)
        results[name] = dict(pred=prediction, corr=corr, rmse=rmse, lag=lag)
        predictions[name] = prediction.astype(np.float32)
        cp, rp = np.percentile(corr, [10, 50, 90]), np.percentile(rmse, [10, 50, 90])
        rows.append(dict(model=name, execution="online" if name.endswith("online") else "offline",
                         parameters=parameters, best_epoch=best_epoch, validation_mse=val_mse,
                         train_seconds=seconds, causality_error=causality_error, corr_p10=cp[0],
                         corr_median=cp[1], corr_p90=cp[2], rmse_p10=rp[0], rmse_median=rp[1],
                         rmse_p90=rp[2], lag_median_ms=1000 * np.median(lag)))
        checkpoints[name] = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        print(f"{name:20s} corr={cp[1]:.3f} rmse={rp[1]:.3f} epoch={best_epoch} params={parameters}")
        network.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(OUTPUT / "model_free_metrics.csv", rows)
    np.savez_compressed(OUTPUT / "model_free_predictions.npz", ids=ids[test], fs=cfg.fs, **predictions)
    torch.save(dict(models=checkpoints, model_names=requested, hidden_size=args.hidden_size,
                    input_channels=cfg.x_channels, episode_length=features.shape[1], x_mean=x_mean, x_std=x_std,
                    y_mean=y_mean, y_std=y_std, validation_driver=validation_driver, seed=args.seed,
                    real_data_only=True), OUTPUT / "model_free.pt")
    plot_model_free(target[test], results, ids[test], cfg.fs, OUTPUT / "model_free_models.png")
    print(f"train={len(train)} validation={len(validation)} ({validation_driver}) test={test.sum()} outputs={OUTPUT}")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("classical", "pitch", "pitch2", "hybrid", "rw-hybrid", "ou-hybrid",
                                               "oscillator", "calibration-ablation", "lstm-kf", "model-free", "all"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default=",".join(MODEL_FREE_MODELS))
    return parser.parse_args()


def main():
    args = arguments()
    loaded = data()
    if args.experiment in ("classical", "all"):
        run_classical(loaded)
    if args.experiment == "pitch":
        run_pitch(loaded)
    if args.experiment == "pitch2":
        run_pitch2(loaded)
    if args.experiment in ("hybrid", "all"):
        run_hybrid(args, loaded)
    if args.experiment == "rw-hybrid":
        run_hybrid(args, loaded, "rw")
    if args.experiment == "ou-hybrid":
        run_hybrid(args, loaded, "ou")
    if args.experiment == "calibration-ablation":
        run_calibration_ablation(loaded)
    if args.experiment == "oscillator":
        run_oscillator(loaded)
    if args.experiment == "lstm-kf":
        run_lstm_kf(args, loaded)
    if args.experiment in ("model-free", "all"):
        run_model_free(args, loaded)


if __name__ == "__main__":
    main()
