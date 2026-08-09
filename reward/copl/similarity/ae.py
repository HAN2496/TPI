import copy
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from .base import ItemSimilarityBuilder, standardize_fit, standardize_apply, median_heuristic_gamma


class Conv1dAE(nn.Module):
    def __init__(self, obs_dim, seq_len, latent_dim=8, hidden_channels=32, kernel_size=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.enc_conv1 = nn.Conv1d(obs_dim, hidden_channels, kernel_size, padding=1)
        self.enc_conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size, padding=1)
        self.enc_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_z = nn.Linear(hidden_channels * 2, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, hidden_channels * 2 * seq_len)
        self.dec_conv1 = nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(hidden_channels, obs_dim, kernel_size, padding=1)
        self._hidden_channels = hidden_channels

    def encode(self, x):
        h = F.leaky_relu(self.enc_conv1(x), 0.2)
        h = F.leaky_relu(self.enc_conv2(h), 0.2)
        h = self.enc_pool(h).squeeze(-1)
        return self.fc_z(h)

    def decode(self, z):
        h = F.leaky_relu(self.dec_fc(z), 0.2)
        h = h.view(-1, self._hidden_channels * 2, self.seq_len)
        h = F.leaky_relu(self.dec_conv1(h), 0.2)
        return self.dec_conv2(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class AESimilarity(ItemSimilarityBuilder):
    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)
        self.mu_stats, self.sd_stats = standardize_fit(X)
        Xs = standardize_apply(X, self.mu_stats, self.sd_stats)
        self._T, self._D = T, D
        self.metric = cfg.ae_metric
        self.temperature = cfg.ae_temperature

        if self.metric == "kl" or self.metric.startswith("wasserstein"):
            raise ValueError("AE similarity does not support kl/wasserstein metric. Use cosine/laplacian/rbf.")

        device = torch.device(cfg.device)

        latent_dim = cfg.ae_latent_dim
        ae_epochs = cfg.ae_epochs
        ae_lr = cfg.ae_lr
        ae_batch_size = cfg.ae_batch_size
        ae_hidden_channels = cfg.ae_hidden_channels
        val_size = cfg.val_size

        X_tensor = torch.tensor(Xs.reshape(N, T, D).transpose(0, 2, 1), dtype=torch.float32)
        train_idx, val_idx = train_test_split(np.arange(N), test_size=val_size, random_state=cfg.seed)

        train_loader = DataLoader(TensorDataset(X_tensor[train_idx]), batch_size=ae_batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_tensor[val_idx]), batch_size=ae_batch_size, shuffle=False)

        self.ae = Conv1dAE(D, T, latent_dim, ae_hidden_channels).to(device)
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=ae_lr)

        if cfg.verbose > 0:
            print(f"  [AE] Training: latent_dim={latent_dim}, epochs={ae_epochs}, "
                  f"lr={ae_lr}, val_size={val_size}, metric={self.metric}")

        best_val_loss = float("inf")
        best_state = None
        self.history = {"train/loss": [], "val/loss": []}

        for epoch in range(ae_epochs):
            self.ae.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                x_hat, _ = self.ae(batch_x)
                loss = F.mse_loss(x_hat, batch_x, reduction="mean")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_idx)

            self.ae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    x_hat, _ = self.ae(batch_x)
                    loss = F.mse_loss(x_hat, batch_x, reduction="mean")
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_idx)

            self.history["train/loss"].append(train_loss)
            self.history["val/loss"].append(val_loss)

            if cfg.verbose > 1 or (cfg.verbose > 0 and (epoch + 1) % 10 == 0):
                print(f"    epoch {epoch+1}/{ae_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.ae.state_dict())

        if best_state is not None:
            self.ae.load_state_dict(best_state)
            if cfg.verbose > 0:
                print(f"  [AE] Best val_loss={best_val_loss:.4f}")

        self.ae.eval()
        return self.build_graph(item_series, cfg)

    def build_graph(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)
        Xs = standardize_apply(X, self.mu_stats, self.sd_stats)
        X_tensor = torch.tensor(Xs.reshape(N, T, D).transpose(0, 2, 1), dtype=torch.float32)
        device = next(self.ae.parameters()).device

        self.ae.eval()
        with torch.no_grad():
            Z = self.ae.encode(X_tensor.to(device)).cpu().numpy()
        self.Z_train = Z

        self.gamma = None
        if self.metric == "cosine":
            Aii_norm = self.build_cosine_knn_graph(Z, knn_k=cfg.knn_k, mutual=cfg.mutual, temperature=self.temperature)
        elif self.metric == "laplacian":
            gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
            self.gamma = gamma_med * cfg.gamma_mul
            Aii_norm = self.build_laplacian_knn_graph(Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)
        elif self.metric == "rbf":
            gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
            self.gamma = gamma_med * cfg.gamma_mul
            Aii_norm = self.build_knn_graph(Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)
        else:
            raise ValueError(f"AE similarity does not support metric '{self.metric}'.")

        self.Aii_norm = Aii_norm

        meta = {
            "method": "ae",
            "metric": self.metric,
            "latent_dim": self.ae.latent_dim,
            "knn_k": cfg.knn_k,
            "mutual": cfg.mutual,
        }
        if self.gamma is not None:
            meta["gamma"] = float(self.gamma)
        if cfg.verbose > 0:
            print(f"  [AE] meta: {meta}")

        return {"Aii_norm": Aii_norm, "Z_train": Z, "gamma": self.gamma, "meta": meta}

    def save(self, path):
        torch.save({
            "ae_state_dict": self.ae.state_dict(),
            "obs_dim": self.ae.obs_dim, "seq_len": self.ae.seq_len,
            "latent_dim": self.ae.latent_dim, "hidden_channels": self.ae._hidden_channels,
            "mu_stats": self.mu_stats, "sd_stats": self.sd_stats,
            "metric": self.metric, "temperature": self.temperature,
            "gamma": self.gamma, "_T": self._T, "_D": self._D,
            "history": self.history,
        }, path)

    def load(self, path, device):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.ae = Conv1dAE(ck["obs_dim"], ck["seq_len"], ck["latent_dim"], ck["hidden_channels"]).to(device)
        self.ae.load_state_dict(ck["ae_state_dict"])
        self.ae.eval()
        self.mu_stats, self.sd_stats = ck["mu_stats"], ck["sd_stats"]
        self.metric, self.temperature, self.gamma = ck["metric"], ck["temperature"], ck["gamma"]
        self._T, self._D = ck["_T"], ck["_D"]
        self.history = ck["history"]

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        Xs = standardize_apply(Xf, self.mu_stats, self.sd_stats)
        X_tensor = torch.tensor(Xs.reshape(n_test, T, D).transpose(0, 2, 1), dtype=torch.float32)
        device = next(self.ae.parameters()).device
        self.ae.eval()
        with torch.no_grad():
            z = self.ae.encode(X_tensor.to(device))
        return z.cpu().numpy()

    def visualize(self, save_dir, item_series, item_owner_uid,
                  train_drivers, feature_names, item_labels):
        from core.plots import plot_training_curves
        from ..viz import (plot_vae_reconstruction, plot_vae_feature_mse,
                                     plot_vae_latent, plot_driver_similarity_matrix,
                                     plot_ae_orig_paper, plot_ae_recon_paper)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(self.history, save_dir / "ae_curves.png",
                             title="AE Similarity Training")
        plot_vae_reconstruction(self.ae, item_series, feature_names,
                                self.mu_stats, self.sd_stats, save_dir / "ae_recon.png")
        plot_ae_orig_paper(self.ae, item_series, feature_names,
                           self.mu_stats, self.sd_stats, save_dir / "ae_orig_paper.png")
        plot_ae_recon_paper(self.ae, item_series, feature_names,
                            self.mu_stats, self.sd_stats, save_dir / "ae_recon_paper.png")
        plot_vae_feature_mse(self.ae, item_series, feature_names,
                             self.mu_stats, self.sd_stats, save_dir / "ae_mse.png")
        plot_vae_latent(self.Z_train, item_owner_uid, train_drivers,
                        save_dir / "latent_by_driver.png")
        plot_driver_similarity_matrix(self.Aii_norm, item_owner_uid, train_drivers,
                                      save_dir / "driver_sim_matrix.png")

    def get_affinity(self, Z_query, Z_target, k):
        from sklearn.neighbors import NearestNeighbors
        if self.metric == "cosine":
            norm_q = np.linalg.norm(Z_query, axis=1, keepdims=True) + 1e-12
            norm_t = np.linalg.norm(Z_target, axis=1, keepdims=True) + 1e-12
            Z_q_norm = Z_query / norm_q
            Z_t_norm = Z_target / norm_t
            nnbrs = NearestNeighbors(n_neighbors=min(k, Z_t_norm.shape[0]), metric="euclidean")
            nnbrs.fit(Z_t_norm)
            dist, nbr = nnbrs.kneighbors(Z_q_norm, return_distance=True)
            cos_sim = 1.0 - 0.5 * (dist ** 2)
            return nbr, self._compute_cosine_affinity(cos_sim, self.temperature)
        elif self.metric == "laplacian":
            nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
            nnbrs.fit(Z_target)
            dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
            return nbr, self._compute_laplacian_affinity(dist, self.gamma)
        else:
            nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
            nnbrs.fit(Z_target)
            dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
            return nbr, self._compute_rbf_affinity(dist ** 2, self.gamma)