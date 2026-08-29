import argparse
from pathlib import Path

import numpy as np
import torch

from loader import Dataset
from run_reconstruction import Config as ReconstructionConfig, load
from reward.reconstruction.ou import affine_calibration, build_ou, fit_ou
from reward.reconstruction.sequence import (
    ResidualLSTM, build_model_free, fit_model_free, fit_residual_lstm,
    normalize_model_free, normalize_residual, ou_features,
    predict_model_free, predict_residual_lstm,
)


TRAIN = ("강신길", "조현석", "한규택", "박재일", "이지환")
TEST = ("김재호", "김진명", "김태근", "신민철", "이강근")
FS = 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("ou", "ou_lstm", "gru", "bilstm"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    cfg = ReconstructionConfig(test=TEST)
    xr, yr, ids = load(Dataset("datasets"), cfg)
    drivers = np.array([value.split()[0] for value in ids])
    train = np.isin(drivers, TRAIN)
    validation = drivers == "조현석"
    test = np.isin(drivers, TEST)
    az, bounce = xr[:, :, 1], yr[:, :, 0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method in ("ou", "ou_lstm"):
        fit = fit_ou(az[train], bounce[train], FS, maxiter=40)
        state_space = build_ou(fit.parameters, FS)
        state = fit.states(az, FS)
        gain, offset = affine_calibration(state[train, :, 1], bounce[train])
        baseline = gain * state[:, :, 1] + offset
        if args.method == "ou_lstm":
            features = ou_features(state, az, state_space)
            features, residual, *_ = normalize_residual(features, bounce - baseline, train)
            model = ResidualLSTM(features.shape[-1], args.hidden)
            fit_residual_lstm(model, features, residual.astype(np.float32),
                              np.flatnonzero(train), np.flatnonzero(validation), device,
                              epochs=args.epochs, verbose=True)
            prediction = baseline + predict_residual_lstm(model, features, device)
        else:
            prediction = baseline
        params = fit.parameters
    else:
        features, target, *_ = normalize_model_free(xr, bounce, train)
        name = "gru_online" if args.method == "gru" else "bilstm_offline"
        model = build_model_free(name, features.shape[-1], features.shape[1], args.hidden)
        fit_model_free(model, features, target.astype(np.float32), np.flatnonzero(train),
                       np.flatnonzero(validation), device, epochs=args.epochs, verbose=True)
        prediction = predict_model_free(model, features, device)
        params = np.array([], dtype=float)

    output = Path(args.out or f"outputs/reconstruction/{args.method}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reconstructed_signals.npz", ids=ids, channels=np.array(["Bounce_rate_6D"]),
                        **{args.method: prediction[:, None].astype(np.float32)}, parameters=params)
    corr = np.array([np.corrcoef(prediction[i], bounce[i])[0, 1] for i in np.flatnonzero(test)])
    rmse = np.sqrt(np.mean((prediction[test] - bounce[test]) ** 2, axis=1))
    np.savetxt(output / "metrics.csv", np.array([[np.nanpercentile(corr, 10), np.median(corr), np.nanpercentile(corr, 90),
                                                   np.nanpercentile(rmse, 10), np.median(rmse), np.nanpercentile(rmse, 90)]]),
               delimiter=",", header="corr_p10,corr_median,corr_p90,rmse_p10,rmse_median,rmse_p90", comments="")
    print(f"method={args.method} test_corr={np.median(corr):.4f} test_rmse={np.median(rmse):.4f} output={output}")


if __name__ == "__main__":
    main()
