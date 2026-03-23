import torch
import torch.nn as nn
from .flow import Flow


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
        )
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h_ = self.model(x)
        return self.FC_mean(h_), self.FC_var(h_)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class VAEModel(nn.Module):
    def __init__(self, encoder_input, decoder_input, latent_dim, hidden_dim,
                 annotation_size, size_segment, kl_weight=1.0,
                 learned_prior=False, flow_prior=False, annealer=None, reward_scaling=1.0):
        super().__init__()
        self.Encoder = Encoder(encoder_input, hidden_dim, latent_dim)
        self.Decoder = Decoder(decoder_input, hidden_dim, 1)
        self.latent_dim = latent_dim
        self.mean = nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.log_var = nn.Parameter(torch.zeros(latent_dim), requires_grad=learned_prior)
        self.annotation_size = annotation_size
        self.size_segment = size_segment
        self.learned_prior = learned_prior
        self.flow_prior = flow_prior
        if flow_prior:
            self.flow = Flow(latent_dim, "radial", 4)
        self.kl_weight = kl_weight
        self.annealer = annealer
        self.scaling = reward_scaling

    def reparameterization(self, mean, var):
        return mean + var * torch.randn_like(var).to(mean.device)

    def encode(self, s, y):
        s_ = s.view(s.shape[0], s.shape[1], -1)
        y = y.reshape(s.shape[0], s.shape[1], -1)
        encoder_input = torch.cat([s_, y], dim=-1).view(s.shape[0], -1)
        return self.Encoder(encoder_input)

    def decode(self, obs, z):
        return self.Decoder(torch.cat([obs, z], dim=-1))

    def transform(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        z = torch.randn_like(std).mul(std).add_(mean)
        return self.flow(z)

    def reconstruction_loss(self, x, x_hat):
        return nn.functional.binary_cross_entropy(x_hat, x, reduction="mean")

    def accuracy(self, x, x_hat):
        return torch.mean(((x_hat > 0.5).float() == x).float())

    def latent_loss(self, mean, log_var):
        if self.learned_prior:
            kl = -0.5 * (
                1 + (log_var - self.log_var) - (log_var - self.log_var).exp()
                - (mean.pow(2) - self.mean.pow(2)) / (self.log_var.exp())
            ).sum(dim=1).mean()
        else:
            kl = -0.5 * (1.0 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()
        return kl

    def forward(self, s, y):
        mean, log_var = self.encode(s, y)
        if self.flow_prior:
            z, log_det = self.transform(mean, log_var)
        else:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
            log_det = None
        z = z.repeat((1, self.annotation_size * self.size_segment)).view(
            -1, self.annotation_size, self.size_segment, z.shape[1])

        r = self.decode(s, z)
        r_hat = r.sum(axis=2) / self.scaling
        p_hat = torch.sigmoid(r_hat).view(-1, 1)
        labels = y.view(-1, 1)

        reconstruction_loss = self.reconstruction_loss(labels, p_hat)
        accuracy = self.accuracy(labels, p_hat)
        latent_loss = self.latent_loss(mean, log_var)
        kl_weight = self.annealer.slope() * self.kl_weight if self.annealer else self.kl_weight
        loss = reconstruction_loss + kl_weight * latent_loss
        if self.flow_prior:
            loss = loss - torch.sum(log_det)

        return loss, {
            "loss": loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kld_loss": latent_loss.item(),
            "accuracy": accuracy.item(),
            "kl_weight": kl_weight,
        }

    def sample_prior(self, size):
        z = torch.randn(size, self.latent_dim).cuda()
        if self.learned_prior:
            z = z * torch.exp(0.5 * self.log_var) + self.mean
        elif self.flow_prior:
            z, _ = self.flow(z)
        return z

    def sample_posterior(self, s, y):
        mean, log_var = self.encode(s, y)
        return mean, log_var, self.reparameterization(mean, torch.exp(0.5 * log_var))
