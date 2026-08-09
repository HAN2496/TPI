"""Driver-held-out hyperparameter sweep for the reconstruction LSTM."""

import argparse
import copy
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from loader import Dataset
from reward.reconstruction.methods import LSTM
from run_reconstruction import Config, load


DEFAULT_OUT = ROOT / "outputs" / "lab" / "reconstruction" / "lstm_sweep" / "results.jsonl"


def parse_list(text, cast):
    return [cast(v) for v in text.split(",") if v]


def standardize(xr, yr, train):
    xf = xr[train].reshape(-1, xr.shape[2])
    yf = yr[train].reshape(-1, yr.shape[2])
    mx, sx = xf.mean(0), xf.std(0) + 1e-8
    my, sy = yf.mean(0), yf.std(0) + 1e-8
    x = torch.from_numpy(((xr - mx) / sx).transpose(0, 2, 1).astype(np.float32))
    y = torch.from_numpy(((yr - my) / sy).transpose(0, 2, 1).astype(np.float32))
    return x, y


@torch.no_grad()
def evaluate(model, x, y, indices, device, batch_size):
    model.eval()
    pred, true = [], []
    for b in torch.from_numpy(indices).split(batch_size):
        xb = x[b].to(device)
        pred.append(model(xb).cpu())
        true.append(y[b])
    pred = torch.cat(pred).numpy()
    true = torch.cat(true).numpy()
    mse = float(np.mean((pred - true) ** 2))
    a = pred - pred.mean(2, keepdims=True)
    b = true - true.mean(2, keepdims=True)
    corr = (a * b).sum(2) / np.sqrt((a * a).sum(2) * (b * b).sum(2) + 1e-12)
    median = np.nanmedian(corr, axis=0)
    score = float((median[0] + median[2]) / 2)
    return {"score": score, "mse": mse, "corr": median.tolist()}


def train_one(x, y, train_idx, val_idx, cfg, args, device):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = LSTM(x.shape[1], y.shape[1], cfg["hidden"], cfg["layers"],
                 cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=args.weight_decay)
    best, best_state, stale = None, None, 0
    started = time.time()

    generator = torch.Generator().manual_seed(cfg["seed"])
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        total = 0.0
        order = torch.from_numpy(train_idx)[torch.randperm(len(train_idx), generator=generator)]
        for b in order.split(args.batch_size):
            xb, yb = x[b].to(device), y[b].to(device)
            loss = F.mse_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += loss.item() * len(b)

        metrics = evaluate(model, x, y, val_idx, device, args.batch_size)
        metrics["train_mse"] = total / len(train_idx)
        improved = (args.fixed_epochs or best is None or
                    metrics["score"] > best["score"] + args.min_delta)
        if improved:
            best = {**metrics, "epoch": epoch}
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 2 == 0 or improved:
            print(f"  epoch={epoch:02d} train={metrics['train_mse']:.4f} val={metrics['mse']:.4f} "
                  f"corr={metrics['corr']} score={metrics['score']:.5f}", flush=True)
        if not args.fixed_epochs and epoch >= args.min_epochs and stale >= args.patience:
            break

    model.load_state_dict(best_state)
    return model, {**best, "seconds": time.time() - started,
                   "params": sum(p.numel() for p in model.parameters())}


def result_key(result):
    fields = ("tag", "val_drivers", "hidden", "layers", "dropout", "lr", "seed",
              "batch_size", "weight_decay", "grad_clip", "max_epochs", "fixed_epochs")
    return tuple(json.dumps(result.get(k, False), ensure_ascii=False, sort_keys=True) for k in fields)


def read_completed(path):
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(result_key(json.loads(line)))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--val-drivers", required=True,
                        help="Comma-separated driver names; use 'TEST' only for final evaluation")
    parser.add_argument("--hidden", default="64,96,128,160")
    parser.add_argument("--layers", default="2")
    parser.add_argument("--dropouts", default="0.1")
    parser.add_argument("--lrs", default="0.0003,0.001")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--fixed-epochs", action="store_true",
                        help="Report the final epoch instead of selecting an epoch on validation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    base = Config()
    xr, yr, ids = load(Dataset(ROOT / "datasets"), base)
    drivers = np.array([s.split()[0] for s in ids])
    val_drivers = list(base.test) if args.val_drivers == "TEST" else args.val_drivers.split(",")
    train_mask = ~np.isin(drivers, list(base.test) + val_drivers)
    val_mask = np.isin(drivers, val_drivers)
    if not train_mask.any() or not val_mask.any():
        raise ValueError(f"empty split: train={train_mask.sum()} val={val_mask.sum()}")
    train_idx, val_idx = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    x, y = standardize(xr, yr, train_mask)
    print(f"split train={len(train_idx)} val={len(val_idx)} val_drivers={val_drivers}", flush=True)

    configs = itertools.product(parse_list(args.hidden, int), parse_list(args.layers, int),
                                parse_list(args.dropouts, float), parse_list(args.lrs, float),
                                parse_list(args.seeds, int))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set() if args.rerun else read_completed(args.output)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for hidden, layers, dropout, lr, seed in configs:
        record = {
            "tag": args.tag, "val_drivers": val_drivers, "hidden": hidden,
            "layers": layers, "dropout": dropout, "lr": lr, "seed": seed,
            "batch_size": args.batch_size, "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip, "max_epochs": args.max_epochs,
            "fixed_epochs": args.fixed_epochs,
        }
        if result_key(record) in completed:
            print(f"skip completed {record}", flush=True)
            continue
        print(f"\nSTART {record}", flush=True)
        _, metrics = train_one(x, y, train_idx, val_idx, record, args, device)
        result = {**record, **metrics}
        with args.output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"DONE {json.dumps(result, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
