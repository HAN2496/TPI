import torch
import torch.nn as nn

from src.model.vpl.vae import VAEModel


class VAEPairwiseModel(VAEModel):
    """
    VAE model for pairwise preference learning.
    Extends VAEModel to support both pairwise training and binary inference.
    """

    def __init__(self, encoder_input, decoder_input, latent_dim, hidden_dim,
                 annotation_size, size_segment, kl_weight=1.0, learned_prior=False,
                 flow_prior=False, annealer=None, reward_scaling=1.0):
        """
        Initialize VAE Pairwise Model.

        Args:
            encoder_input: Encoder input dimension for pairwise: set_len * (2 * T * d + 1)
            decoder_input: Decoder input dimension: d + latent_dim (same as binary)
            latent_dim: Latent dimension
            hidden_dim: Hidden layer dimension
            annotation_size: Number of pairs per query (set_len)
            size_segment: Number of timesteps per episode (T)
            kl_weight: KL divergence weight
            learned_prior: Whether to use learned prior
            flow_prior: Whether to use flow prior
            annealer: Annealer for KL weight scheduling
            reward_scaling: Reward scaling factor
        """
        super().__init__(encoder_input, decoder_input, latent_dim, hidden_dim,
                        annotation_size, size_segment, kl_weight, learned_prior,
                        flow_prior, annealer, reward_scaling)

    def encode_pairwise(self, s_a, s_b, pref):
        """
        Encode pairwise observations and preferences.

        Args:
            s_a: (Batch, set_len, T, d) - First episode in each pair
            s_b: (Batch, set_len, T, d) - Second episode in each pair
            pref: (Batch, set_len, 1) - Preference labels

        Returns:
            mean, log_var: Latent distribution parameters
        """
        # Flatten episodes
        s_a_flat = s_a.reshape(s_a.shape[0], s_a.shape[1], -1)  # (B, set_len, T*d)
        s_b_flat = s_b.reshape(s_b.shape[0], s_b.shape[1], -1)  # (B, set_len, T*d)
        pref = pref.reshape(s_a.shape[0], s_a.shape[1], -1)  # (B, set_len, 1)

        # Concatenate: [obs_a, obs_b, preference]
        encoder_input = torch.cat([s_a_flat, s_b_flat, pref], dim=-1).reshape(
            s_a.shape[0], -1
        )  # (B, set_len * (2*T*d + 1))

        mean, log_var = self.Encoder(encoder_input)
        return mean, log_var

    def forward_pairwise(self, s_a, s_b, pref):
        """
        Forward pass for pairwise training.

        Args:
            s_a, s_b: (Batch, set_len, T, d)
            pref: (Batch, set_len, 1)

        Returns:
            loss, metrics
        """
        # Encode
        mean, log_var = self.encode_pairwise(s_a, s_b, pref)

        # Reparameterization
        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
            log_det = None

        # Expand z for decoding
        z_expanded = z.repeat((1, self.annotation_size * self.size_segment)).reshape(
            -1, self.annotation_size, self.size_segment, z.shape[1]
        )  # (B, set_len, T, latent_dim)

        # Decode both episodes
        r_a = self.decode(s_a, z_expanded)  # (B, set_len, T, 1)
        r_b = self.decode(s_b, z_expanded)  # (B, set_len, T, 1)

        # Sum rewards over timesteps
        r_a_sum = r_a.sum(dim=2) / self.scaling  # (B, set_len, 1)
        r_b_sum = r_b.sum(dim=2) / self.scaling  # (B, set_len, 1)

        # Bradley-Terry-Luce probability: P(A > B) = sigmoid(r_a - r_b)
        logits = (r_a_sum - r_b_sum).reshape(-1, 1)
        labels = pref.reshape(-1, 1)

        # Binary cross-entropy loss
        reconstruction_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='sum'
        )

        # Compute accuracy
        p_hat = torch.sigmoid(logits)
        accuracy = self.accuracy(labels, p_hat)

        # KL divergence
        latent_loss = self.latent_loss(mean, log_var)

        # Total loss
        kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss

        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        metrics = {
            'loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kld_loss': latent_loss.item(),
            'accuracy': accuracy.item(),
            'kl_weight': kl_weight,
        }

        return loss, metrics

    def forward(self, *args, **kwargs):
        """
        Dispatch to appropriate forward function based on input format.

        For pairwise training: forward(s_a, s_b, pref)
        For binary inference: forward(s, y) - inherited from parent VAEModel
        """
        if len(args) == 3:
            # Pairwise mode
            return self.forward_pairwise(*args, **kwargs)
        elif len(args) == 2:
            # Binary mode (inference)
            return super().forward(*args, **kwargs)
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")
