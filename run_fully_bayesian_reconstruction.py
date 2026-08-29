import argparse

from run_fully_bayesian import Config, ReconstructionConfig, main


def run(method, burnin=500, samples=1500):
    cfg = Config()
    cfg.timestamp = None
    cfg.n_burnin = burnin
    cfg.n_samples = samples
    cfg.manual_stats = {
        "Bounce_rate_6D": ["impulse_abs", "abs_peak_deriv", "p2p_deriv", "wrms_z_deriv"],
        "IMU_VerAccelVal": ["p2p", "wrms_z", "crest", "vdv"],
        "IMU_LongAccelVal": ["std", "wrms_xy", "impulse_abs", "crest"],
    }
    cfg.recon = ReconstructionConfig(methods=(method,), timestamp=method)
    main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("ou", "ou_lstm", "gru", "bilstm"))
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1500)
    args = parser.parse_args()
    run(args.method, args.burnin, args.samples)
