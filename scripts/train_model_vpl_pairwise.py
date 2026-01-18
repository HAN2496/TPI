import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.model.vpl_new.vae import VAEModel
from src.model.vpl_new.trainer import VPLTrainer

from src.utils.logger import ExperimentLogger
from src.utils.vpl_pairwise_dataset import create_vpl_dataset_new, PreferenceDataset, create_vpl_dataset_new_2

FLAGS = {
    # data
    "features": ["IMU_VerAccelVal", "Bounce_rate_6D", "Pitch_rate_6D", "IMU_LongAccelVal"],
    "test_driver_name": "강신길",
    "time_range": (5, 7),
    "downsample": 5,
    "tie_ratio": 0.0,

    "context_size": 128,
    "val_size": 0.1,
    "use_test_context": True,  # True: use context_size for test, False: use all pairs at once

    # model
    "hidden_dim": 128,
    "batch_size": 64,
    "latent_dim": 32,
    "kl_weight": 1.0,
    "flow_prior": False,
    "use_annealing": True,
    "annealer_baseline": 0.0,
    "annealer_type": "cosine", # linear, cosine, logistic
    "annealer_cycles":4,

    # Trainer
    "lr": 1e-3,
    "weight_decay": 0.0,
    "early_stop": False,
    "patience": 10,
    "min_delta": 3e-4,

    "n_epochs": 500,
    "eval_freq": 10,

    "device": "cuda", # cuda, cpu
    "verbose": 1,
}

def collate_fn(batch):
    return {
        'observations': torch.stack([torch.from_numpy(item['observations']).float() for item in batch]),
        'observations_2': torch.stack([torch.from_numpy(item['observations_2']).float() for item in batch]),
        'labels': torch.stack([torch.from_numpy(item['labels']).float() for item in batch]),
    }

def plot_history(metrics, logger):
    plot_keys = [k for k in metrics.keys() if k.startswith("train/") or k.startswith("eval/")]

    num_plots = len(plot_keys)
    if num_plots == 0:
        return

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten() 

    for i, key in enumerate(plot_keys):
        ax = axes[i]
        ax.plot(metrics[key])
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    save_path = logger.log_dir / "training_curves.png"
    os.makedirs(save_path.parent, exist_ok=True) 

    plt.savefig(save_path)
    plt.close()


def main():

    print("Creating dataset...")
    trainval_data, test_driver_data = create_vpl_dataset_new_2(
        test_driver_names=[FLAGS["test_driver_name"]],
        features=FLAGS["features"],
        time_range=FLAGS["time_range"],
        downsample=FLAGS["downsample"],
        tie_ratio=FLAGS["tie_ratio"]
    )

    print(f"Total train pairs: {len(trainval_data)}")

    # Shuffle all pairs
    np.random.shuffle(trainval_data)

    # Split into train/val (pairs level, not driver level)
    n_total = len(trainval_data)
    n_val = max(1, int(n_total * FLAGS["val_size"]))
    n_train = n_total - n_val

    train_data = trainval_data[:n_train]
    val_data = trainval_data[n_train:]

    print(f"\nTrain/Val split:")
    print(f"  Train pairs: {len(train_data)}")
    print(f"  Val pairs: {len(val_data)}")

    # Create fixed context datasets (group pairs into queries)
    train_dataset = PreferenceDataset(train_data, FLAGS["context_size"])
    val_dataset = PreferenceDataset(val_data, FLAGS["context_size"])

    print(f"\nQuery statistics:")
    print(f"  Train queries: {len(train_dataset)}")
    print(f"  Val queries: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),  # Process entire val set at once
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
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

    trainer = VPLTrainer(
        model=model,
        logger=logger,
        config=FLAGS
    )

    print("\nStarting training...")
    # metrics, val_accuracy = trainer.train(train_loader, val_loader)

    # print(f"\nTraining complete!")
    # print(f"Final validation accuracy: {val_accuracy:.4f}")
    # print(f"Results saved to: {logger.log_dir}")

    # print("\nPlotting training curves...")
    # plot_history(metrics, logger)

    # print("\n" + "="*70)
    # print("Testing on test driver")
    # print("="*70)





    from src.model.vpl_new.utils import compute_step_rewards
    from src.utils.utils import _load_dataset_sequences
    from src.utils.visualization import plot_roc_curve, plot_test_step_rewards, visualize_all_driver_latents, visualize_episode_probabilities
    from src.utils.data_loader import DatasetManager
    from sklearn.metrics import roc_auc_score

    model.load_state_dict(torch.load(logger.log_dir / "best_model.pt"))
    model.eval()

    # 테스트 결과 저장용
    test_probability_results = {}

    # Prepare all driver data for latent visualization
    print("\n" + "="*70)
    print("Preparing all driver data for latent visualization")
    print("="*70)

    all_driver_data = {}
    manager = DatasetManager("datasets", downsample=FLAGS["downsample"])
    all_driver_names = manager.keys()

    for driver_name in all_driver_names:
        X, y = _load_dataset_sequences(
            driver_name,
            FLAGS["time_range"],
            FLAGS["downsample"],
            {'features': FLAGS["features"]}
        )

        true_mask = (y == 1)
        false_mask = (y == 0)
        true_episodes = X[true_mask]
        false_episodes = X[false_mask]

        if len(true_episodes) == 0 or len(false_episodes) == 0:
            print(f"Skipping {driver_name}: no True or False episodes")
            continue

        driver_obs = []
        driver_obs_2 = []
        driver_labels = []

        for true_ep in true_episodes:
            for false_ep in false_episodes:
                driver_obs.append(true_ep)
                driver_obs_2.append(false_ep)
                driver_labels.append(1.0)

        driver_obs = np.stack(driver_obs, axis=0)
        driver_obs_2 = np.stack(driver_obs_2, axis=0)
        driver_labels = np.array(driver_labels).reshape(-1, 1)

        all_driver_data[driver_name] = {
            'observations': driver_obs,
            'observations_2': driver_obs_2,
            'labels': driver_labels
        }
        print(f"  {driver_name}: {len(driver_obs)} pairs")

    for test_driver_name, test_data in test_driver_data.items():
        print(f"\nTest driver: {test_driver_name}")
        print(f"  Test pairs: {len(test_data['observations'])}")

        # Convert to individual pairs
        test_pairs = []
        for i in range(len(test_data['observations'])):
            test_pairs.append({
                'observations': test_data['observations'][i],
                'observations_2': test_data['observations_2'][i],
                'labels': test_data['labels'][i]
            })

        # Create test dataset with context_size
        if FLAGS["use_test_context"]:
            test_context_size = FLAGS["context_size"]
            print(f"  Using context_size={test_context_size} for test inference")
        else:
            test_context_size = len(test_pairs)  # Use all pairs at once
            print(f"  Using all {test_context_size} pairs at once for test inference")

        test_dataset = PreferenceDataset(test_pairs, test_context_size)
        print(f"  Test queries: {len(test_dataset)}")

        # Get average z from all queries
        print(f"Estimating latent z for {test_driver_name}...")
        all_z = []
        for query_data in test_dataset:
            obs1 = torch.from_numpy(query_data['observations']).float().to(FLAGS["device"]).unsqueeze(0)
            obs2 = torch.from_numpy(query_data['observations_2']).float().to(FLAGS["device"]).unsqueeze(0)
            labels = torch.from_numpy(query_data['labels']).float().to(FLAGS["device"]).unsqueeze(0)

            with torch.no_grad():
                mean, _ = model.encode(obs1, obs2, labels)
            all_z.append(mean.squeeze(0).cpu().numpy())

        z_mean = np.mean(all_z, axis=0)
        print(f"  z shape: {z_mean.shape} (averaged over {len(all_z)} queries)")

        X, y = _load_dataset_sequences(
            test_driver_name,
            FLAGS["time_range"],
            FLAGS["downsample"],
            {'features': FLAGS["features"]}
        )

        print(f"Computing step rewards for {len(X)} episodes...")
        step_rewards = compute_step_rewards(model, X, z_mean, FLAGS["device"])
        print(f"  Step rewards shape: {step_rewards.shape}")

        # 모델과 동일하게 scaling 적용: r_hat = sum(r_t) / scaling
        T = step_rewards.shape[1]  # timesteps
        mean_rewards = step_rewards.sum(axis=1) / T
        print(f"\nReward statistics (scaled):")
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
                save_path=logger.log_dir / f"test_{test_driver_name}_roc.png",
                title=f"Test Driver ROC - {test_driver_name}"
            )
            print(f"  ROC curve saved")
        else:
            auroc = 0.0
            print("\nWarning: Only one class in test data, cannot compute AUROC")

        plot_test_step_rewards(
            step_rewards, y, test_driver_name,
            n_samples=10,
            save_path=logger.log_dir / f"test_{test_driver_name}_step_rewards.png"
        )
        print(f"  Step rewards plot saved")

        # 확률 시각화를 위한 결과 저장
        test_probability_results[test_driver_name] = {
            'mean_rewards': mean_rewards,
            'y_true': y,
            'auroc': auroc
        }

    print("\n" + "="*70)
    print("Visualizing latent space for all drivers")
    print("="*70)

    visualize_all_driver_latents(
        model=model,
        all_driver_data=all_driver_data,
        device=FLAGS["device"],
        save_path=logger.log_dir / "all_drivers_latent_space.png",
        context_size=FLAGS["context_size"]
    )

    print("\n" + "="*70)
    print("Visualizing episode probabilities")
    print("="*70)

    visualize_episode_probabilities(
        test_results=test_probability_results,
        save_path=logger.log_dir / "episode_probabilities.png"
    )

    print("\n" + "="*70)
    print("All done!")
    print("="*70)


if __name__ == "__main__":
    main()