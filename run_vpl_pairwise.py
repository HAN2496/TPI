from dataclasses import dataclass

import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from loader import Dataset, View
from core import (Run, evaluate_predictions, plot_sequential_auroc,
                      plot_training_curves, compute_sequential_aurocs)
from reward.vpl.vae import VAEModel
from reward.vpl.trainer import VPLTrainer, Annealer
from reward.vpl.utils import PreferenceDataset, convert_to_pairwise, compute_step_rewards
from reward.vpl import viz


@dataclass
class Config:
    train: tuple = ("김진명", "김태근", "조현석", "한규택", "박재일", "이지환")
    test: tuple = ("강신길",)
    view: View = View(
        features=("IMU_VerAccelVal", "Pitch_rate_6D", "Bounce_rate_6D", "IMU_LongAccelVal"),
        around=(0, 2), downsample=5)
    context_size: int = 64
    tie_ratio: float = 0.0
    normalize: bool = True   # False면 encoder log_var 폭주로 NaN (원시 스케일 입력)
    val_size: float = 0.1
    timestamp: str = None

    latent_dim: int = 8
    hidden_dim: int = 128
    kl_weight: float = 0.1
    flow_prior: bool = False
    reward_scaling: object = "T"

    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 300
    batch_size: int = 32
    eval_freq: int = 10
    early_stop: bool = False
    patience: int = 20
    min_delta: float = 0.0

    anneal_shape: str = "linear"
    anneal_cycles: int = 4

    device: str = "cuda"
    seed: int = 42
    verbose: int = 1


def load_data(cfg, ds):
    data = {n: cfg.view(ds[n]) for n in dict.fromkeys(cfg.train + cfg.test)}
    for n, (X, y) in data.items():
        assert (y == 1).any() and (y == 0).any(), f"{n}: pairwise needs both classes"
    if cfg.normalize:
        concat = np.concatenate([data[n][0] for n in cfg.train], axis=0)
        mu = concat.mean(axis=(0, 1))
        sd = concat.std(axis=(0, 1)) + 1e-6
        data = {n: ((X - mu) / sd, y) for n, (X, y) in data.items()}
    return {n: data[n] for n in cfg.train}, {n: data[n] for n in cfg.test}


def build_queries(cfg, train_data):
    queries = defaultdict(list)
    for name, (X, y) in train_data.items():
        dd = convert_to_pairwise(X, y, name, cfg.context_size, cfg.tie_ratio)
        print(f"  {name}: {len(dd['observations'])} queries")
        for key in ("observations", "observations_2", "labels", "driver_name"):
            queries[key].extend(dd[key])
    return {k: np.stack(v) for k, v in queries.items()}


def build_model(cfg, queries):
    xb = queries["observations"]
    annotation_size, size_segment, obs_dim = xb.shape[1], xb.shape[2], xb.shape[3]
    encoder_input = annotation_size * (2 * size_segment * obs_dim + 1)
    decoder_input = obs_dim + cfg.latent_dim   # forward()가 스텝 단위 decode (구버전의 seg*obs+latent는 버그)
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

    trainer = VPLTrainer(model, run.dir, {
        "device": cfg.device, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
        "n_epochs": cfg.n_epochs, "eval_freq": cfg.eval_freq,
        "early_stop": cfg.early_stop, "patience": cfg.patience, "min_delta": cfg.min_delta,
    })
    print("[1] Training VPL Pairwise...")
    metrics, best_acc = trainer.train(train_loader, val_loader, verbose=cfg.verbose)
    plot_training_curves(dict(metrics), run.plots / "training_curves.png", title="VPL Pairwise")
    run.metrics["train/val"] = {"accuracy": float(best_acc)}


def eval_driver(cfg, model, name, X, y):
    """순차 예측: context[0..t-1]로 z 추정 → episode t 예측."""
    device = torch.device(cfg.device)
    if len(np.unique(y)) < 2:
        print(f"  [Skip] {name}: needs both classes")
        return None
    probs, labels, ctx_sizes = [], [], []
    for t in range(1, len(X)):
        if len(np.unique(y[:t])) < 2:   # pairwise는 context에 양 클래스 필요
            continue
        ctx = convert_to_pairwise(X[:t], y[:t], name, cfg.context_size, cfg.tie_ratio)
        loader = DataLoader(PreferenceDataset(ctx), batch_size=len(ctx["labels"]), shuffle=False)
        means = []
        with torch.no_grad():
            for b in loader:
                m, _ = model.encode(b["observations"].to(device).float(),
                                    b["observations_2"].to(device).float(),
                                    b["labels"].to(device).float())
                means.append(m.cpu().numpy())
        z = np.mean(np.concatenate(means, axis=0), axis=0)
        step_rewards = compute_step_rewards(model, X[t:t+1], z, device)
        probs.append(torch.sigmoid(torch.tensor(step_rewards.mean(axis=1))).item())
        labels.append(int(y[t]))
        ctx_sizes.append(t)
    return (np.array(probs), np.array(labels), ctx_sizes) if probs else None


def evaluate(cfg, run, model, queries, train_data, test_data):
    model.eval()
    device = torch.device(cfg.device)
    plots = run.plots

    print("[2] Evaluating test drivers...")
    for name, (X, y) in test_data.items():
        result = eval_driver(cfg, model, name, X, y)
        if result is None:
            continue
        p, l, ctx = result
        seq_x, seq_y = compute_sequential_aurocs(p.tolist(), l.tolist(), ctx)
        plot_sequential_auroc([seq_x], [seq_y], plots, [name], save_name=f"seq_auroc_{name}")
        m = evaluate_predictions([l], [p], plots, [name], save_name=f"metrics_{name}",
                                 title=f"VPL Pairwise - {name} (test)")[0]
        run.metrics[f"test/{name}"] = m
        print(f"  {name}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  Brier={m['brier']:.4f}")

    z_by_driver = viz.collect_z_pairwise(model, queries, device)
    viz.plot_z_space(z_by_driver, list(test_data), plots / "viz" / "z_space.png")
    viz.plot_z_dims(z_by_driver, plots / "viz" / "z_dims.png")
    viz.plot_reward_dist(model, train_data, z_by_driver, device, plots / "viz")


def main(cfg=None):
    cfg = cfg or Config()
    run = Run("vpl_pairwise", cfg)
    ds = Dataset("datasets")

    train_data, test_data = load_data(cfg, ds)
    queries = build_queries(cfg, train_data)
    model = build_model(cfg, queries)

    if run.eval_only:
        model.load_state_dict(torch.load(run.dir / "best_model.pt",
                                         map_location=cfg.device, weights_only=True))
    else:
        train(cfg, run, model, queries)

    evaluate(cfg, run, model, queries, train_data, test_data)
    run.finish()


if __name__ == "__main__":
    main()
