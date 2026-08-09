from dataclasses import dataclass

import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from loader import Dataset, View
from core import (Run, split_ctx, grid, Track,
                      evaluate_predictions, plot_sequential_auroc, plot_training_curves)
from reward.vpl.vae_binary import VAEModel
from reward.vpl.trainer import VPLTrainerBinary, Annealer
from reward.vpl.utils import PreferenceDataset, convert_to_binary_context
from reward.vpl import viz


@dataclass
class Config:
    train: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test: tuple = ("강신길",)
    view: View = View(
        features=("IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"),
        around=(0, 2), downsample=5, smooth=(12.0, 2))
    context_size: int = 8
    normalize: bool = False
    balanced: bool = True
    val_size: float = 0.1
    timestamp: str = "test"

    latent_dim: int = 8
    hidden_dim: int = 128
    kl_weight: float = 0.05
    flow_prior: bool = True
    reward_scaling: object = "T"

    lr: float = 0.001
    weight_decay: float = 0.0
    n_epochs: int = 500
    batch_size: int = 128
    eval_freq: int = 10
    early_stop: bool = False
    patience: int = 20
    min_delta: float = 0.0

    anneal_shape: str = "cosine"
    anneal_cycles: int = 4

    device: str = "cuda"
    seed: int = 42
    verbose: int = 1


def load_data(cfg, ds):
    """드라이버별 (X, y). normalize=True면 train 통계로 전체 정규화."""
    data = {n: cfg.view(ds[n]) for n in dict.fromkeys(cfg.train + cfg.test)}
    if cfg.normalize:
        concat = np.concatenate([data[n][0] for n in cfg.train], axis=0)
        mu = concat.mean(axis=(0, 1))
        sd = concat.std(axis=(0, 1)) + 1e-6
        data = {n: ((X - mu) / sd, y) for n, (X, y) in data.items()}
    return {n: data[n] for n in cfg.train}, {n: data[n] for n in cfg.test}


def build_queries(cfg, train_data):
    queries = defaultdict(list)
    for name, (X, y) in train_data.items():
        perm = np.random.permutation(len(X))
        dd = convert_to_binary_context(X[perm], y[perm], name, cfg.context_size, balanced=cfg.balanced)
        for key in ("observations", "labels", "driver_name"):
            queries[key].extend(dd[key])
    return {k: np.stack(v) for k, v in queries.items()}


def build_model(cfg, queries):
    xb = queries["observations"]
    annotation_size, size_segment, obs_dim = xb.shape[1], xb.shape[2], xb.shape[3]
    encoder_input = annotation_size * (size_segment * obs_dim + 1)
    decoder_input = obs_dim + cfg.latent_dim
    reward_scaling = size_segment if cfg.reward_scaling == "T" else cfg.reward_scaling
    return VAEModel(
        encoder_input=encoder_input, decoder_input=decoder_input,
        latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
        annotation_size=annotation_size, size_segment=size_segment,
        kl_weight=cfg.kl_weight, flow_prior=cfg.flow_prior,
        annealer=Annealer(cfg.n_epochs // cfg.anneal_cycles, cfg.anneal_shape,
                          cyclical=cfg.anneal_cycles > 1),
        reward_scaling=reward_scaling,
    ).to(cfg.device)


def train(cfg, run, model, queries):
    idx = np.arange(len(queries["observations"]))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_size, random_state=cfg.seed)
    train_loader = DataLoader(PreferenceDataset({k: v[tr_idx] for k, v in queries.items()}),
                              batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(PreferenceDataset({k: v[va_idx] for k, v in queries.items()}),
                            batch_size=len(va_idx), shuffle=False)

    trainer = VPLTrainerBinary(model, run.dir, {
        "device": cfg.device, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
        "n_epochs": cfg.n_epochs, "eval_freq": cfg.eval_freq,
        "early_stop": cfg.early_stop, "patience": cfg.patience, "min_delta": cfg.min_delta,
    })
    print("[1] Training VPL Binary...")
    metrics, _ = trainer.train(train_loader, val_loader, verbose=cfg.verbose)
    plot_training_curves(dict(metrics), run.plots / "training_curves.png", title="VPL Binary")
    model.load_state_dict(torch.load(trainer.best_model_path, weights_only=True))


def encode_z(cfg, model, X, y, name, device):
    ctx = convert_to_binary_context(X, y, name, cfg.context_size, balanced=False)
    loader = DataLoader(PreferenceDataset(ctx), batch_size=len(ctx["labels"]), shuffle=False)
    means = []
    with torch.no_grad():
        for b in loader:
            m, _ = model.encode(b["observations"].to(device).float(),
                                b["labels"].to(device).float())
            means.append(m.cpu().numpy())
    return np.mean(np.concatenate(means, axis=0), axis=0)


def eval_driver(cfg, model, name, X, y, plots=None):
    """context X[:t]로 z 추정 → 고정 holdout(뒷 절반) 예측. plots 주면 snapshot 저장."""
    device = torch.device(cfg.device)
    (ctx_X, ctx_y), (hold_X, hold_y) = split_ctx(X, y)
    if len(ctx_y) < 1 or len(np.unique(hold_y)) < 2:
        print(f"  [Skip] {name}: holdout lacks both classes")
        return None

    T = X.shape[1]
    hold_obs = torch.tensor(hold_X, dtype=torch.float32, device=device).unsqueeze(1)
    pcts = {max(1, int(len(y) * p)): int(p * 100) for p in (0.1, 0.2, 0.3, 0.4, 0.5)}
    pcts = {t: p for t, p in pcts.items() if t <= len(ctx_y)}

    trk = Track(hold_y)
    z_traj = []
    for t in grid(len(ctx_y)):
        z = encode_z(cfg, model, ctx_X[:t], ctx_y[:t], name, device)
        z_traj.append(z)
        z_t = torch.tensor(z, dtype=torch.float32, device=device).view(1, 1, 1, -1).expand(len(hold_X), 1, T, -1)
        with torch.no_grad():
            step_r = model.decode(hold_obs, z_t).squeeze(1).squeeze(-1)
            probs = torch.sigmoid(step_r.mean(dim=1)).cpu().numpy()
        trk.add(t, probs)
        if plots and t in pcts:
            snap = plots / "snapshots" / f"context_{pcts[t]}pct"
            evaluate_predictions([hold_y], [probs], snap, [name], save_name="metrics",
                                 title=f"VPL Binary - {name} (Context {pcts[t]}%)")
            viz.plot_latent_z(z, snap / "latent_z.png", title=f"Latent z — Context {pcts[t]}%")

    return dict(name=name, holdout_y=hold_y, trk=trk, z_traj=z_traj)


def evaluate(cfg, run, model, train_data, test_data):
    model.eval()
    device = torch.device(cfg.device)
    plots = run.plots
    test_z = {}

    for name, (X, y) in test_data.items():
        d = eval_driver(cfg, model, name, X, y, plots=plots)
        if d is None:
            continue
        trk = d["trk"]
        plot_sequential_auroc([trk.ts], [trk.aurocs], plots, [name], save_name=f"seq_auroc_{name}")
        m = evaluate_predictions([d["holdout_y"]], [trk.final[2]], plots, [name],
                                 save_name=f"metrics_{name}", title=f"VPL Binary - {name} (test)")[0]
        run.metrics[f"test/{name}"] = m
        print(f"  {name}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")
        test_z[name] = np.stack(d["z_traj"])
        viz.plot_z_evolution(d["z_traj"], trk.ts, name, plots / f"z_evolution_{name}.png")

    print("\n[2] Evaluating training drivers...")
    for name, (X, y) in train_data.items():
        d = eval_driver(cfg, model, name, X, y)
        if d is None:
            continue
        trk = d["trk"]
        plot_sequential_auroc([trk.ts], [trk.aurocs], plots / "train", [name], save_name=f"seq_auroc_{name}")
        m = evaluate_predictions([d["holdout_y"]], [trk.final[2]], plots / "train", [name],
                                 save_name=f"metrics_{name}", title=f"VPL Binary - {name} (train)")[0]
        run.metrics[f"train/{name}"] = m
        print(f"  {name}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

    z_by_driver = viz.collect_z_binary(model, train_data, cfg.context_size, device)
    viz.plot_z_space({**z_by_driver, **test_z}, list(test_data), plots / "z_space.png")
    viz.plot_z_dims(z_by_driver, plots / "z_dims.png")
    viz.plot_reward_dist(model, train_data, z_by_driver, device, plots / "reward_model")


def main(cfg=None):
    cfg = cfg or Config()
    run = Run("vpl", cfg)
    ds = Dataset("datasets")

    train_data, test_data = load_data(cfg, ds)
    queries = build_queries(cfg, train_data)
    model = build_model(cfg, queries)

    if run.eval_only:
        model.load_state_dict(torch.load(run.dir / "best_model.pt",
                                         map_location=cfg.device, weights_only=True))
    else:
        train(cfg, run, model, queries)

    evaluate(cfg, run, model, train_data, test_data)
    run.finish()


if __name__ == "__main__":
    main()
