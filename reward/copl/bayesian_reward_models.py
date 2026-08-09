import torch
import torch.nn as nn
import torch.nn.functional as F
import hamiltorch


def unflatten(model, flat):
    out, i = {}, 0
    for name, p in model.named_parameters():
        out[name] = flat[i:i + p.numel()].view(p.shape)
        i += p.numel()
    return out


class BayesianRM(nn.Module):
    """sample_probs()가 (S, B) 확률을 내면 forward는 평균 확률의 logit을 반환.
    기존 torch.sigmoid(rm(user_emb, obs)) 평가 코드가 posterior predictive mean을 그대로 얻는다."""
    uses_user_embedding = True

    def forward(self, user_emb, obs):
        return torch.logit(self.sample_probs(user_emb, obs).mean(0), eps=1e-6)

    def uncertainty(self, user_emb, obs):
        """(mean, epistemic, aleatoric). total H(p̄) = epistemic(MI) + aleatoric(E[H(p_s)]), 단위 nat."""
        ent = lambda q: torch.special.entr(q) + torch.special.entr(1 - q)
        p = self.sample_probs(user_emb, obs)
        p_mean = p.mean(0)
        aleatoric = ent(p).mean(0)
        return p_mean, ent(p_mean) - aleatoric, aleatoric


class EnsembleRM(BayesianRM):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def sample_probs(self, user_emb, obs):
        return torch.stack([torch.sigmoid(m(user_emb, obs)) for m in self.members])


class MCDropoutRM(BayesianRM):
    def __init__(self, model, n_samples):
        super().__init__()
        self.model = model
        self.n_samples = n_samples

    def sample_probs(self, user_emb, obs):
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        return torch.stack([torch.sigmoid(self.model(user_emb, obs)) for _ in range(self.n_samples)])


class HMCPosteriorRM(BayesianRM):
    def __init__(self, model, samples):
        super().__init__()
        self.model = model
        self.register_buffer("samples", samples)

    def sample_probs(self, user_emb, obs):
        return torch.stack([
            torch.sigmoid(torch.func.functional_call(self.model, unflatten(self.model, s), (user_emb, obs)))
            for s in self.samples])


def hmc_sample(model, user_emb, obs, y, cfg, pos_weight=None):
    """학습된 model에서 warm start해 posterior weight 샘플 (S, P)를 반환. full-batch likelihood."""
    def log_prob(flat):
        logits = torch.func.functional_call(model, unflatten(model, flat), (user_emb, obs))
        ll = -F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="sum")
        return ll - flat.pow(2).sum() / (2 * cfg.rm_prior_std ** 2)

    init = torch.cat([p.detach().flatten() for p in model.parameters()])
    samples = hamiltorch.sample(log_prob_func=log_prob, params_init=init,
                                num_samples=cfg.hmc_burn + cfg.hmc_num_samples + 1, burn=cfg.hmc_burn,
                                step_size=cfg.hmc_step_size, num_steps_per_sample=cfg.hmc_leapfrog)
    return torch.stack([s.detach() for s in samples[1:]])
