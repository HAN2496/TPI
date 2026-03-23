import copy
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from .base import ItemSimilarityBuilder, standardize_fit, standardize_apply, median_heuristic_gamma


class Conv1dVAE(nn.Module):
    def __init__(self, obs_dim, seq_len, latent_dim=8, hidden_channels=32, kernel_size=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.enc_conv1 = nn.Conv1d(obs_dim, hidden_channels, kernel_size, padding=1)
        self.enc_conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size, padding=1)
        self.enc_pool = nn.AdaptiveAvgPool1d(1)

        enc_out_dim = hidden_channels * 2
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, hidden_channels * 2 * seq_len)
        self.dec_conv1 = nn.ConvTranspose1d(hidden_channels * 2, hidden_channels, kernel_size, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(hidden_channels, obs_dim, kernel_size, padding=1)
        self._hidden_channels = hidden_channels

    def encode(self, x):
        h = F.leaky_relu(self.enc_conv1(x), 0.2)
        h = F.leaky_relu(self.enc_conv2(h), 0.2)
        h = self.enc_pool(h).squeeze(-1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = F.leaky_relu(self.dec_fc(z), 0.2)
        h = h.view(-1, self._hidden_channels * 2, self.seq_len)
        h = F.leaky_relu(self.dec_conv1(h), 0.2)
        return self.dec_conv2(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(x, x_hat, mu, logvar, kl_weight=1.0):
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


class VAESimilarity(ItemSimilarityBuilder):
    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)
        self.mu_stats, self.sd_stats = standardize_fit(X)
        Xs = standardize_apply(X, self.mu_stats, self.sd_stats)
        self._T, self._D = T, D
        self.metric = cfg.vae_metric
        self.temperature = cfg.vae_temperature

        device = torch.device(cfg.device)

        latent_dim = cfg.vae_latent_dim
        vae_epochs = cfg.vae_epochs
        vae_lr = cfg.vae_lr
        vae_kl_weight = cfg.vae_kl_weight
        vae_batch_size = cfg.vae_batch_size
        vae_hidden_channels = cfg.vae_hidden_channels
        val_size = cfg.val_size

        X_tensor = torch.tensor(Xs.reshape(N, T, D).transpose(0, 2, 1), dtype=torch.float32)
        train_idx, val_idx = train_test_split(np.arange(N), test_size=val_size, random_state=cfg.seed)

        train_loader = DataLoader(TensorDataset(X_tensor[train_idx]), batch_size=vae_batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_tensor[val_idx]), batch_size=vae_batch_size, shuffle=False)

        self.vae = Conv1dVAE(D, T, latent_dim, vae_hidden_channels).to(device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

        if cfg.verbose > 0:
            print(f"  [VAE] Training: latent_dim={latent_dim}, epochs={vae_epochs}, "
                  f"kl_weight={vae_kl_weight}, lr={vae_lr}, val_size={val_size}, metric={self.metric}")

        best_val_loss = float("inf")
        best_state = None
        self.history = {"train/loss": [], "val/loss": [],
                        "train/recon": [], "val/recon": [],
                        "train/kl": [], "val/kl": []}

        for epoch in range(vae_epochs):
            self.vae.train()
            train_loss = train_recon = train_kl = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                x_hat, mu, logvar = self.vae(batch_x)
                loss, recon, kl = Conv1dVAE.loss_function(batch_x, x_hat, mu, logvar, vae_kl_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n = batch_x.size(0)
                train_loss  += loss.item()  * n
                train_recon += recon.item() * n
                train_kl    += kl.item()    * n
            train_loss  /= len(train_idx)
            train_recon /= len(train_idx)
            train_kl    /= len(train_idx)

            self.vae.eval()
            val_loss = val_recon = val_kl = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    x_hat, mu, logvar = self.vae(batch_x)
                    loss, recon, kl = Conv1dVAE.loss_function(batch_x, x_hat, mu, logvar, vae_kl_weight)
                    n = batch_x.size(0)
                    val_loss  += loss.item()  * n
                    val_recon += recon.item() * n
                    val_kl    += kl.item()    * n
            val_loss  /= len(val_idx)
            val_recon /= len(val_idx)
            val_kl    /= len(val_idx)

            self.history["train/loss"].append(train_loss)
            self.history["val/loss"].append(val_loss)
            self.history["train/recon"].append(train_recon)
            self.history["val/recon"].append(val_recon)
            self.history["train/kl"].append(train_kl)
            self.history["val/kl"].append(val_kl)

            if cfg.verbose > 1 or (cfg.verbose > 0 and (epoch + 1) % 10 == 0):
                print(f"    epoch {epoch+1}/{vae_epochs}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}  "
                      f"recon={val_recon:.4f}  kl={val_kl:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.vae.state_dict())

        if best_state is not None:
            self.vae.load_state_dict(best_state)
            if cfg.verbose > 0:
                print(f"  [VAE] Best val_loss={best_val_loss:.4f}")

        self.vae.eval()
        return self.build_graph(item_series, cfg)

    def build_graph(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)
        Xs = standardize_apply(X, self.mu_stats, self.sd_stats)
        X_tensor = torch.tensor(Xs.reshape(N, T, D).transpose(0, 2, 1), dtype=torch.float32)
        device = next(self.vae.parameters()).device

        self.vae.eval()
        with torch.no_grad():
            mu_z, logvar_z = self.vae.encode(X_tensor.to(device))
            Z = mu_z.cpu().numpy()
            Z_sigma = np.exp(0.5 * logvar_z.cpu().numpy())
        self.Z_train = Z
        self.Z_sigma_train = Z_sigma

        self.gamma = None
        if self.metric == "cosine":
            Aii_norm = self.build_cosine_knn_graph(Z, knn_k=cfg.knn_k, mutual=cfg.mutual, temperature=self.temperature)
        elif self.metric in ("wasserstein", "wasserstein_laplacian"):
            gamma_med = median_heuristic_gamma(np.concatenate([Z, Z_sigma], axis=1), seed=cfg.seed)
            self.gamma = gamma_med * cfg.gamma_mul
            Aii_norm = self.build_wasserstein_knn_graph(Z, Z_sigma, knn_k=cfg.knn_k, gamma=self.gamma,
                                                        mutual=cfg.mutual, kernel=self.metric)
        elif self.metric == "laplacian":
            gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
            self.gamma = gamma_med * cfg.gamma_mul
            Aii_norm = self.build_laplacian_knn_graph(Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)
        else:  # rbf
            gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
            self.gamma = gamma_med * cfg.gamma_mul
            Aii_norm = self.build_knn_graph(Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)

        meta = {
            "method": "vae",
            "metric": self.metric,
            "latent_dim": self.vae.latent_dim,
            "knn_k": cfg.knn_k,
            "mutual": cfg.mutual,
        }
        if self.gamma is not None:
            meta["gamma"] = float(self.gamma)
        if cfg.verbose > 0:
            print(f"  [VAE] meta: {meta}")

        return {"Aii_norm": Aii_norm, "Z_train": Z, "gamma": self.gamma, "meta": meta}

    def save(self, path):
        torch.save({
            "vae_state_dict": self.vae.state_dict(),
            "obs_dim": self.vae.obs_dim, "seq_len": self.vae.seq_len,
            "latent_dim": self.vae.latent_dim, "hidden_channels": self.vae._hidden_channels,
            "mu_stats": self.mu_stats, "sd_stats": self.sd_stats,
            "metric": self.metric, "temperature": self.temperature,
            "gamma": self.gamma, "_T": self._T, "_D": self._D,
            "history": self.history,
        }, path)

    def load(self, path, device):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.vae = Conv1dVAE(ck["obs_dim"], ck["seq_len"], ck["latent_dim"], ck["hidden_channels"]).to(device)
        self.vae.load_state_dict(ck["vae_state_dict"])
        self.vae.eval()
        self.mu_stats, self.sd_stats = ck["mu_stats"], ck["sd_stats"]
        self.metric, self.temperature, self.gamma = ck["metric"], ck["temperature"], ck["gamma"]
        self._T, self._D = ck["_T"], ck["_D"]
        self.history = ck["history"]

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        Xs = standardize_apply(Xf, self.mu_stats, self.sd_stats)
        X_tensor = torch.tensor(Xs.reshape(n_test, T, D).transpose(0, 2, 1), dtype=torch.float32)
        device = next(self.vae.parameters()).device
        self.vae.eval()
        with torch.no_grad():
            mu_z, _ = self.vae.encode(X_tensor.to(device))
        return mu_z.cpu().numpy()

    def visualize(self, save_dir, item_series, item_owner_uid,
                  train_drivers, feature_names, item_labels):
        from ....evaluation import plot_training_curves
        from ..visualization import plot_vae_reconstruction, plot_vae_feature_mse, plot_vae_latent
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_training_curves(self.history, save_dir / "vae_curves.png",
                             title="VAE Similarity Training")
        plot_vae_reconstruction(self.vae, item_series, feature_names,
                                self.mu_stats, self.sd_stats, save_dir / "vae_recon.png")
        plot_vae_feature_mse(self.vae, item_series, feature_names,
                             self.mu_stats, self.sd_stats, save_dir / "vae_mse.png")
        plot_vae_latent(self.Z_train, item_owner_uid, train_drivers,
                        save_dir / "latent_by_driver.png")

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
        elif self.metric in ("wasserstein_laplacian", "laplacian"):
            nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
            nnbrs.fit(Z_target)
            dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
            return nbr, self._compute_laplacian_affinity(dist, self.gamma)
        else:  # rbf, wasserstein
            nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
            nnbrs.fit(Z_target)
            dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
            return nbr, self._compute_rbf_affinity(dist ** 2, self.gamma)
