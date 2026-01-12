import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.model.vpl_new.vae import VAEModel
from src.model.vpl_new.trainer import VPLTrainer

from src.utils.logger import ExperimentLogger
from src.utils.vpl_pairwise_dataset import create_vpl_datset_new, PreferenceDataset

FLAGS = {
    # data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_name": "강신길",
    "time_range": (5, 7),
    "downsample": 5,
    "tie_ratio": 0.0,

    "sampling_method": "balanced", # "balanced" or "sqrt" or "natural"
    "context_size": 128,
    "steps_per_epoch": 100,

    # model
    "hidden_dim": 256,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "early_stop": False,
    "patience": 10,
    "min_delta": 3e-4,

    "latent_dim": 32,
    "kl_weight": 1.0,
    "flow_prior": False,
    "use_annealing": False,
    "annealer_baseline": 0.0,
    "annealer_type": "cosine",
    "annealer_cycles":4,

    # Training
    "n_epochs": 500,
    "eval_freq": 50,
    "save_freq": 50,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "verbose": 1,
}

def collate_fn(batch):
    return {
        'observations': torch.stack([torch.from_numpy(item['observations']).float() for item in batch]),
        'observations_2': torch.stack([torch.from_numpy(item['observations_2']).float() for item in batch]),
        'labels': torch.stack([torch.from_numpy(item['labels']).float() for item in batch]),
        'driver_names': [item['driver_name'] for item in batch]
    }

def plot_history(metrics, logger):
    os.makedirs(logger.run_dir / "plots", exist_ok=True)

    for key in metrics.keys():
        if not key.startswith("train/") and not key.startswith("eval/"):
            continue

        plt.figure()
        plt.plot(metrics[key])
        plt.title(key)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.grid()
        plt.savefig(logger.run_dir / "plots" / f"{key}.png")
        plt.close()

def main():

    print("Creating dataset...")
    train_datasets = create_vpl_datset_new(
        test_deriver_name=FLAGS["test_driver_name"],
        features=FLAGS["features"],
        time_range=FLAGS["time_range"],
        downsample=FLAGS["downsample"],
        tie_ratio=FLAGS["tie_ratio"]
    )

    print(f"\nTotal drivers: {len(train_datasets)}")
    for i, data in enumerate(train_datasets[:3]):
        print(f"  Driver {data['driver_name']}: {data['observations'].shape[0]} pairs")

    np.random.shuffle(train_datasets)
    n_total = len(train_datasets)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val

    train_drivers = train_datasets[:n_train]
    val_drivers = train_datasets[n_train:]

    train_dataset = PreferenceDataset(train_drivers, context_size=FLAGS["context_size"])
    val_dataset = PreferenceDataset(val_drivers, context_size=FLAGS["context_size"])

    data_counts = [len(d['observations']) for d in train_drivers]

    if FLAGS["sampling_method"] == "balanced":
        weights = [1.0 / c for c in data_counts]
        print(f"Sampling Method: Balanced (All drivers have equal probability)")
    elif FLAGS["sampling_method"] == "sqrt":
        weights = [math.sqrt(c) for c in data_counts]
        weights = [1.0 / math.sqrt(c) for c in data_counts]
        print(f"Sampling Method: Square Root (Proportional to sqrt(N))")
    else:
        weights = [1.0 for _ in data_counts]
        print(f"Sampling Method: Natural (Proportional to dataset size)")

    total_samples = FLAGS["steps_per_epoch"] * FLAGS["batch_size"]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=total_samples,
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS["batch_size"],
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FLAGS["batch_size"],
        collate_fn=collate_fn,
        num_workers=0
    )

    sample_batch = next(iter(train_loader))
    B, Nmax, T, d = sample_batch['observations'].shape
    encoder_input = 2 * T * d + 1
    decoder_input = d + FLAGS["latent_dim"]

    print(f"\nModel dimensions:")
    print(f"  Input shape: (B={B}, Nmax={Nmax}, T={T}, d={d})")
    print(f"  Encoder input: {encoder_input} (per pair)")
    print(f"  Decoder input: {decoder_input}")

    from src.model.vpl_new.trainer import Annealer
    annealer = None
    if FLAGS["use_annealing"]:
        annealer = Annealer(
            total_steps=FLAGS["n_epochs"] // FLAGS["annealer_cycles"],
            shape=FLAGS["annealer_type"],
            baseline=FLAGS["annealer_baseline"],
            cyclical=FLAGS["annealer_cycles"] > 1
        )

    model = VAEModel(
        encoder_input=encoder_input,
        decoder_input=decoder_input,
        latent_dim=FLAGS["latent_dim"],
        hidden_dim=FLAGS["hidden_dim"],
        annotation_size=Nmax,
        size_segment=T,
        kl_weight=FLAGS["kl_weight"],
        flow_prior=FLAGS["flow_prior"],
        annealer=annealer,
        reward_scaling=T
    ).to(FLAGS["device"])

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")

    logger = ExperimentLogger(
        log_dir="artifacts/vpl_pairwise_new/",
        experiment_name="vpl_pairwise_new_experiment",
        add_timestamp=True
    )

    train_config = {
        'lr': FLAGS["lr"],
        'weight_decay': FLAGS["weight_decay"],
        'n_epochs': FLAGS["n_epochs"],
        'early_stop': FLAGS["early_stop"],
        'patience': FLAGS["patience"],
        'min_delta': FLAGS["min_delta"]
    }

    trainer = VPLTrainer(
        model=model,
        logger=logger,
        config=train_config,
        device=FLAGS["device"]
    )

    print("\nStarting training...")
    metrics, val_accuracy = trainer.train(train_loader, val_loader)

    print(f"\nTraining complete!")
    print(f"Final validation accuracy: {val_accuracy:.4f}")
    print(f"Results saved to: {logger.log_dir}")

    print("\nPlotting training curves...")
    plot_history(metrics, logger)

    print("\n" + "="*70)
    print("Testing on test driver")
    print("="*70)





    from src.model.vpl_new.utils import get_test_latent, compute_step_rewards
    from src.utils.utils import _load_dataset_sequences
    from src.utils.visualization import plot_roc_curve, plot_test_step_rewards
    from sklearn.metrics import roc_auc_score

    model.load_state_dict(torch.load(logger.run_dir / "best_model.pt"))
    model.eval()

    print(f"\nTest driver: {FLAGS['test_driver_name']}")

    test_driver_dataset = create_vpl_datset_new(
        test_deriver_name=None,
        features=FLAGS["features"],
        time_range=FLAGS["time_range"],
        downsample=FLAGS["downsample"],
        tie_ratio=FLAGS["tie_ratio"]
    )

    test_driver_data = None
    for data in test_driver_dataset:
        if data['driver_name'] == FLAGS['test_driver_name']:
            test_driver_data = data
            break

    if test_driver_data is None:
        print(f"Warning: Test driver {FLAGS['test_driver_name']} not found in dataset!")
    else:
        print(f"Estimating latent z for {FLAGS['test_driver_name']}...")
        z_mean = get_test_latent(model, test_driver_data, FLAGS["device"])
        print(f"  z shape: {z_mean.shape}")

        X, y = _load_dataset_sequences(
            FLAGS['test_driver_name'],
            FLAGS["time_range"],
            FLAGS["downsample"],
            {'features': FLAGS["features"]}
        )

        print(f"Computing step rewards for {len(X)} episodes...")
        step_rewards = compute_step_rewards(model, X, z_mean, FLAGS["device"])
        print(f"  Step rewards shape: {step_rewards.shape}")

        mean_rewards = step_rewards.mean(axis=1)
        print(f"\nReward statistics:")
        print(f"  Mean: {mean_rewards.mean():.4f} ± {mean_rewards.std():.4f}")
        print(f"  Range: [{mean_rewards.min():.4f}, {mean_rewards.max():.4f}]")

        true_mask = (y == 1)
        false_mask = (y == 0)
        true_rewards = mean_rewards[true_mask]
        false_rewards = mean_rewards[false_mask]

        if len(true_rewards) > 0 and len(false_rewards) > 0:
            print(f"\nLabel separation:")
            print(f"  True episodes: {true_rewards.mean():.4f} ± {true_rewards.std():.4f}")
            print(f"  False episodes: {false_rewards.mean():.4f} ± {false_rewards.std():.4f}")
            print(f"  Difference: {true_rewards.mean() - false_rewards.mean():.4f}")

        if len(np.unique(y)) > 1:
            auroc = roc_auc_score(y, mean_rewards)
            print(f"\nAUROC: {auroc:.4f}")

            plot_roc_curve(
                y, mean_rewards,
                save_path=logger.run_dir / f"test_{FLAGS['test_driver_name']}_roc.png",
                title=f"Test Driver ROC - {FLAGS['test_driver_name']}"
            )
            print(f"  ROC curve saved")
        else:
            print("\nWarning: Only one class in test data, cannot compute AUROC")

        plot_test_step_rewards(
            step_rewards, y, FLAGS['test_driver_name'],
            n_samples=10,
            save_path=logger.run_dir / f"test_{FLAGS['test_driver_name']}_step_rewards.png"
        )
        print(f"  Step rewards plot saved")

    print("\n" + "="*70)
    print("All done!")
    print("="*70)


if __name__ == "__main__":
    main()