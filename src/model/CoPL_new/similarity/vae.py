"""
VAE 기반 Item-Item 유사도 그래프 빌더.

1D-CNN VAE로 시계열의 latent representation을 학습한 후,
latent space에서 cosine similarity 기반 kNN 그래프를 구축합니다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.model.CoPL_new.similarity.base import (
    ItemSimilarityBuilder, standardize_fit, standardize_apply, median_heuristic_gamma
)


# =========================
# 1D-CNN VAE Model
# =========================
class Conv1dVAE(nn.Module):
    """
    1D-CNN Variational Autoencoder for time-series data.

    Encoder: Conv1d layers → flatten → mu, log_var
    Decoder: Linear → unflatten → ConvTranspose1d → reconstruction
    """

    def __init__(self, obs_dim: int, seq_len: int, latent_dim: int = 8,
                 hidden_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # ---------- Encoder ----------
        # Input: (B, obs_dim, seq_len) -- channels-first
        self.enc_conv1 = nn.Conv1d(obs_dim, hidden_channels, kernel_size, padding=1)
        self.enc_conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size, padding=1)
        self.enc_pool = nn.AdaptiveAvgPool1d(1)  # → (B, hidden_channels*2, 1)

        enc_out_dim = hidden_channels * 2
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # ---------- Decoder ----------
        self.dec_fc = nn.Linear(latent_dim, hidden_channels * 2 * seq_len)
        self.dec_conv1 = nn.ConvTranspose1d(
            hidden_channels * 2, hidden_channels, kernel_size, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(
            hidden_channels, obs_dim, kernel_size, padding=1)

        self._hidden_channels = hidden_channels

    def encode(self, x: torch.Tensor):
        """x: (B, obs_dim, seq_len) → mu, logvar: (B, latent_dim)"""
        h = F.leaky_relu(self.enc_conv1(x), 0.2)
        h = F.leaky_relu(self.enc_conv2(h), 0.2)
        h = self.enc_pool(h).squeeze(-1)  # (B, hidden*2)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """z: (B, latent_dim) → x_hat: (B, obs_dim, seq_len)"""
        h = F.leaky_relu(self.dec_fc(z), 0.2)
        h = h.view(-1, self._hidden_channels * 2, self.seq_len)
        h = F.leaky_relu(self.dec_conv1(h), 0.2)
        return self.dec_conv2(h)  # no activation (MSE loss)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    @staticmethod
    def loss_function(x, x_hat, mu, logvar, kl_weight: float = 1.0):
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


# =========================
# VAE Similarity Builder
# =========================
import copy
from sklearn.model_selection import train_test_split

class VAESimilarity(ItemSimilarityBuilder):
    """
    VAE 기반 유사도 그래프 빌더.

    학습 아이템 시계열로 1D-CNN VAE를 학습한 후,
    encoder의 mu (latent mean)를 item representation으로 사용합니다.
    latent space에서 cosine similarity 또는 RBF kNN으로 A_ii를 구축합니다.
    """

    def fit(self, item_series: np.ndarray, cfg) -> dict:
        N, T, D = item_series.shape
        X = item_series.reshape(N, T * D).astype(np.float32)

        # 표준화 (테스트 변환에서도 사용)
        self.mu_stats, self.sd_stats = standardize_fit(X)
        Xs = standardize_apply(X, self.mu_stats, self.sd_stats)
        self._T, self._D = T, D
        
        # Metric setting (Default: cosine)
        self.metric = getattr(cfg, "vae_metric", "cosine")
        self.temperature = getattr(cfg, "vae_temperature", 0.2)

        # VAE 학습
        device = getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu") # Use cfg.device if available
        if isinstance(device, str): device = torch.device(device)
            
        latent_dim = getattr(cfg, "vae_latent_dim", 8)
        vae_epochs = getattr(cfg, "vae_epochs", 50)
        vae_lr = getattr(cfg, "vae_lr", 1e-3)
        vae_kl_weight = getattr(cfg, "vae_kl_weight", 0.1)
        vae_batch_size = getattr(cfg, "vae_batch_size", 128)
        vae_hidden_channels = getattr(cfg, "vae_hidden_channels", 32)
        val_size = getattr(cfg, "val_size", 0.1)

        # (N, T*D) → (N, T, D) → (N, D, T) for Conv1d (channels-first)
        X_tensor = torch.tensor(
            Xs.reshape(N, T, D).transpose(0, 2, 1),
            dtype=torch.float32)

        # Train/Val Split
        train_idx, val_idx = train_test_split(
            np.arange(N), test_size=val_size, random_state=cfg.seed)
        
        train_ds = TensorDataset(X_tensor[train_idx])
        val_ds = TensorDataset(X_tensor[val_idx])
        
        train_loader = DataLoader(train_ds, batch_size=vae_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=vae_batch_size, shuffle=False)

        self.vae = Conv1dVAE(
            obs_dim=D, seq_len=T, latent_dim=latent_dim,
            hidden_channels=vae_hidden_channels,
        ).to(device)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=vae_lr)

        if cfg.verbose > 0:
            print(f"  [VAE] Training: latent_dim={latent_dim}, epochs={vae_epochs}, "
                  f"kl_weight={vae_kl_weight}, lr={vae_lr}, val_size={val_size}, metric={self.metric}")

        best_val_loss = float("inf")
        best_state = None
        
        for epoch in range(vae_epochs):
            # Train
            self.vae.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                x_hat, mu, logvar = self.vae(batch_x)
                loss, recon, kl = Conv1dVAE.loss_function(
                    batch_x, x_hat, mu, logvar, kl_weight=vae_kl_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            train_loss /= len(train_idx)
            
            # Val
            self.vae.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    x_hat, mu, logvar = self.vae(batch_x)
                    loss, recon, kl = Conv1dVAE.loss_function(
                        batch_x, x_hat, mu, logvar, kl_weight=vae_kl_weight)
                    val_loss += loss.item() * batch_x.size(0)
            val_loss /= len(val_idx)

            if cfg.verbose > 1 or (cfg.verbose > 0 and (epoch + 1) % 10 == 0):
                print(f"    epoch {epoch+1}/{vae_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.vae.state_dict())
        
        if best_state is not None:
            self.vae.load_state_dict(best_state)
            if cfg.verbose > 0:
                print(f"  [VAE] Loaded best model with val_loss={best_val_loss:.4f}")

        # Encode: 모든 학습 아이템의 latent representation 추출
        self.vae.eval()
        with torch.no_grad():
            X_all = X_tensor.to(device)
            mu_z, _ = self.vae.encode(X_all)
            Z = mu_z.cpu().numpy()  # (N, latent_dim)

        self.gamma = None
        if self.metric == "cosine":
            # Cosine KNN Graph
            Aii_norm = self.build_cosine_knn_graph(
                Z, knn_k=cfg.knn_k, mutual=cfg.mutual, temperature=self.temperature)
        else:
            # RBF KNN Graph
            gamma_med = median_heuristic_gamma(Z, seed=cfg.seed)
            self.gamma = gamma_med * getattr(cfg, "gamma_mul", 1.0)
            Aii_norm = self.build_knn_graph(
                Z, knn_k=cfg.knn_k, gamma=self.gamma, mutual=cfg.mutual)

        meta = {
            "method": "vae",
            "metric": self.metric,
            "latent_dim": latent_dim,
            "vae_epochs": vae_epochs,
            "vae_best_val_loss": float(best_val_loss),
            "knn_k": cfg.knn_k,
            "mutual": cfg.mutual,
        }
        if self.gamma is not None:
             meta["gamma"] = float(self.gamma)
             
        if cfg.verbose > 0:
            print(f"  [VAE] meta: {meta}")

        return {
            "Aii_norm": Aii_norm,
            "Z_train": Z,
            "gamma": self.gamma,
            "meta": meta,
        }

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        n_test, T, D = X_test.shape
        Xf = X_test.reshape(n_test, T * D).astype(np.float32)
        Xs = standardize_apply(Xf, self.mu_stats, self.sd_stats)

        # (n_test, T*D) → (n_test, D, T)
        X_tensor = torch.tensor(
            Xs.reshape(n_test, T, D).transpose(0, 2, 1),
            dtype=torch.float32)

        device = next(self.vae.parameters()).device
        self.vae.eval()
        with torch.no_grad():
            mu_z, _ = self.vae.encode(X_tensor.to(device))
        return mu_z.cpu().numpy()

    def get_affinity(self, Z_query: np.ndarray, Z_target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors
        
        if self.metric == "cosine":
            # Cosine similarity based affinity
            # Z should be normalized for cosine? 
            # Or use metric='cosine' in NearestNeighbors.
            # build_cosine_knn_graph logic uses normalized Z + euclidean logic or direct dot product.
            # For consistent behavior: normalize -> euclidean -> dist^2 -> weight

            norm_q = np.linalg.norm(Z_query, axis=1, keepdims=True) + 1e-12
            Z_q_norm = Z_query / norm_q

            norm_t = np.linalg.norm(Z_target, axis=1, keepdims=True) + 1e-12
            Z_t_norm = Z_target / norm_t

            nnbrs = NearestNeighbors(n_neighbors=min(k, Z_t_norm.shape[0]), metric="euclidean")
            nnbrs.fit(Z_t_norm)
            dist, nbr = nnbrs.kneighbors(Z_q_norm, return_distance=True)

            # dist^2 = 2(1 - cos) => cos = 1 - dist^2 / 2
            cos_sim = 1.0 - 0.5 * (dist ** 2)

            w = self._compute_cosine_affinity(cos_sim, self.temperature)
            return nbr, w
        else:
            # RBF
            nnbrs = NearestNeighbors(n_neighbors=min(k, Z_target.shape[0]), metric="euclidean")
            nnbrs.fit(Z_target)
            dist, nbr = nnbrs.kneighbors(Z_query, return_distance=True)
            w = self._compute_rbf_affinity(dist ** 2, self.gamma)
            return nbr, w
