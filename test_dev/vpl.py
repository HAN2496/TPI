# -*- coding: utf-8 -*-
"""
Minimal end-to-end implementation of:
- VPL reward model: q_psi(z|context), r_phi(s,z), conditional BTL, ELBO training
- VPL-SPO scaled rewards: r_tilde(s,z)=E_{s'}[p(y=1|s,s',z)] in [0,1]
- Active query selection via mutual information (BALD sampling approximation)
- Conditional policy learning pi(a|s,z) with a small discrete-action IQL variant (offline RL)
- Test-time: active querying -> infer z -> deploy pi(.|z)

This is a faithful "toy" pipeline; not a robotics-scale reproduction.
"""

import math
import random
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Repro
# -----------------------------
torch.manual_seed(0)
random.seed(0)

# -----------------------------
# 1) Small GridWorld
# -----------------------------
class GridWorld:
    """
    N x N grid. State is an integer idx = r*N + c
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    def __init__(self, N=5, max_steps=30):
        self.N = N
        self.num_states = N * N
        self.num_actions = 4
        self.max_steps = max_steps
        self.reset()

    def reset(self, start_state=None):
        if start_state is None:
            self.s = random.randrange(self.num_states)
        else:
            self.s = start_state
        self.t = 0
        return self.s

    def step(self, a: int):
        r, c = divmod(self.s, self.N)
        if a == 0:  # up
            r = max(0, r - 1)
        elif a == 1:  # down
            r = min(self.N - 1, r + 1)
        elif a == 2:  # left
            c = max(0, c - 1)
        elif a == 3:  # right
            c = min(self.N - 1, c + 1)

        s_next = r * self.N + c
        self.s = s_next
        self.t += 1
        done = (self.t >= self.max_steps)
        return s_next, done

def manhattan(N, s, g):
    rs, cs = divmod(s, N)
    rg, cg = divmod(g, N)
    return abs(rs - rg) + abs(cs - cg)

# -----------------------------
# 2) Diverse users / oracle preference labeling
# -----------------------------
class UserOracle:
    """
    Each user h has a goal state g_h.
    Preference: among (sA, sB), prefers the one closer to goal (smaller manhattan distance).
    Adds mild noise optionally.
    """
    def __init__(self, env: GridWorld, goal_state: int, noise_p: float = 0.05):
        self.env = env
        self.goal = goal_state
        self.noise_p = noise_p

    def prefers(self, sA: int, sB: int) -> int:
        dA = manhattan(self.env.N, sA, self.goal)
        dB = manhattan(self.env.N, sB, self.goal)
        y = 1 if dA < dB else 0 if dA > dB else random.choice([0,1])
        # noise flip
        if random.random() < self.noise_p:
            y = 1 - y
        return y

# -----------------------------
# 3) Preference dataset generation: (sA, sB, y, user_id)
# -----------------------------
def sample_state_pairs(num_states: int, num_pairs: int):
    pairs = []
    for _ in range(num_pairs):
        a = random.randrange(num_states)
        b = random.randrange(num_states)
        while b == a:
            b = random.randrange(num_states)
        pairs.append((a, b))
    return pairs

def build_preference_dataset(users: Dict[int, UserOracle],
                             num_pairs_per_user: int,
                             num_states: int) -> Dict[int, List[Tuple[int,int,int]]]:
    """
    Returns per-user context pool: user_id -> list of (sA, sB, y).
    """
    data = {}
    for uid, oracle in users.items():
        pairs = sample_state_pairs(num_states, num_pairs_per_user)
        triples = []
        for sA, sB in pairs:
            y = oracle.prefers(sA, sB)
            triples.append((sA, sB, y))
        data[uid] = triples
    return data

# -----------------------------
# 4) Offline transition dataset for RL
# -----------------------------
@dataclass
class Transition:
    s: int
    a: int
    s_next: int
    done: int

def collect_offline_dataset(env: GridWorld, num_traj: int = 200, behavior_eps: float = 0.3):
    """
    Collect transitions using an epsilon-random behavior policy.
    """
    D: List[Transition] = []
    for _ in range(num_traj):
        s = env.reset()
        done = False
        while not done:
            if random.random() < behavior_eps:
                a = random.randrange(env.num_actions)
            else:
                # mild heuristic: random anyway (kept simple)
                a = random.randrange(env.num_actions)
            s_next, done = env.step(a)
            D.append(Transition(s=s, a=a, s_next=s_next, done=int(done)))
            s = s_next
    return D

# -----------------------------
# 5) VPL: q_psi(z|context) and reward model r_phi(s,z)
# -----------------------------
def one_hot(n, idx):
    x = torch.zeros(n)
    x[idx] = 1.0
    return x

@dataclass
class Config:
    # VPL
    z_dim: int = 4
    state_embed_dim: int = 32
    enc_hidden: int = 128
    rew_hidden: int = 128
    beta_kl: float = 0.05
    lr_vpl: float = 2e-3
    # SPO
    spo_num_comparators: int = 32  # sample sB for expectation
    # Active query
    mi_z_samples: int = 32
    # IQL
    gamma: float = 0.97
    iql_lr: float = 2e-3
    iql_expectile: float = 0.7
    adv_beta: float = 3.0
    adv_clip: float = 20.0
    batch_size: int = 256
    iql_steps: int = 4000
    # Policy net sizes
    q_hidden: int = 256
    v_hidden: int = 256
    pi_hidden: int = 256
    device: str = "cpu"

class StateEmbed(nn.Module):
    def __init__(self, num_states: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states, d),
            nn.Tanh(),
        )
    def forward(self, s_onehot: torch.Tensor) -> torch.Tensor:
        return self.net(s_onehot)

class EncoderQ(nn.Module):
    """
    q_psi(z | context) = N(mu, diag(sigma^2))
    context: list of (sA, sB, y)
    """
    def __init__(self, num_states: int, embed: StateEmbed, cfg: Config):
        super().__init__()
        self.num_states = num_states
        self.embed = embed
        d = cfg.state_embed_dim
        self.pair = nn.Sequential(
            nn.Linear(2*d + 1, cfg.enc_hidden),
            nn.ReLU(),
            nn.Linear(cfg.enc_hidden, cfg.enc_hidden),
            nn.ReLU(),
        )
        self.to_mu = nn.Linear(cfg.enc_hidden, cfg.z_dim)
        self.to_logvar = nn.Linear(cfg.enc_hidden, cfg.z_dim)

    def forward(self, context: List[Tuple[int,int,int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # If empty context: output near-zero posterior (acts like prior)
        if len(context) == 0:
            mu = torch.zeros(1, self.to_mu.out_features)
            logvar = torch.zeros_like(mu)
            return mu, logvar

        feats = []
        for sA, sB, y in context:
            eA = self.embed(one_hot(self.num_states, sA)).unsqueeze(0)
            eB = self.embed(one_hot(self.num_states, sB)).unsqueeze(0)
            y_t = torch.tensor([[float(y)]])
            x = torch.cat([eA, eB, y_t], dim=1)
            feats.append(self.pair(x))
        H = torch.cat(feats, dim=0)              # [N, enc_hidden]
        h = H.mean(dim=0, keepdim=True)          # [1, enc_hidden]
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps

def kl_to_standard_normal(mu, logvar):
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)

class RewardModel(nn.Module):
    """
    Outputs scalar r(s,z). Used inside conditional BTL classifier.
    """
    def __init__(self, num_states: int, embed: StateEmbed, cfg: Config):
        super().__init__()
        self.num_states = num_states
        self.embed = embed
        d = cfg.state_embed_dim
        self.net = nn.Sequential(
            nn.Linear(d + cfg.z_dim, cfg.rew_hidden),
            nn.ReLU(),
            nn.Linear(cfg.rew_hidden, cfg.rew_hidden),
            nn.ReLU(),
            nn.Linear(cfg.rew_hidden, 1),
        )

    def forward(self, s: int, z: torch.Tensor) -> torch.Tensor:
        e = self.embed(one_hot(self.num_states, s)).unsqueeze(0)
        x = torch.cat([e, z], dim=1)
        return self.net(x).squeeze(1)  # [1]

def btl_logit(rew: RewardModel, sA: int, sB: int, z: torch.Tensor):
    return rew(sA, z) - rew(sB, z)

# -----------------------------
# 6) VPL training (jointly phi, psi)
# -----------------------------
def train_vpl(preference_data: Dict[int, List[Tuple[int,int,int]]],
              num_states: int,
              cfg: Config,
              steps: int = 3000,
              context_size: int = 8):
    embed = StateEmbed(num_states, cfg.state_embed_dim).to(cfg.device)
    enc = EncoderQ(num_states, embed, cfg).to(cfg.device)
    rew = RewardModel(num_states, embed, cfg).to(cfg.device)

    opt = torch.optim.Adam(list(embed.parameters()) + list(enc.parameters()) + list(rew.parameters()),
                           lr=cfg.lr_vpl)

    user_ids = list(preference_data.keys())

    for step in range(1, steps + 1):
        uid = random.choice(user_ids)
        pool = preference_data[uid]
        # sample a context batch from same user
        ctx = random.sample(pool, k=min(context_size, len(pool)))
        mu, logvar = enc(ctx)
        z = reparameterize(mu, logvar)

        # recon loss (BTL likelihood as BCE on logits)
        losses = []
        for sA, sB, y in ctx:
            logits = btl_logit(rew, sA, sB, z)
            y_t = torch.tensor([float(y)], device=logits.device)
            losses.append(F.binary_cross_entropy_with_logits(logits, y_t))
        recon = torch.stack(losses).mean()

        kl = kl_to_standard_normal(mu, logvar)
        loss = recon + cfg.beta_kl * kl

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            # sanity: infer z for each user with full context and show one preference prob
            with torch.no_grad():
                def user_mu(u):
                    mu_u, _ = enc(preference_data[u])
                    return mu_u
                z0 = user_mu(user_ids[0])
                z1 = user_mu(user_ids[1]) if len(user_ids) > 1 else z0
                # compare 0 vs last state
                p0 = torch.sigmoid(btl_logit(rew, 0, num_states-1, z0)).item()
                p1 = torch.sigmoid(btl_logit(rew, 0, num_states-1, z1)).item()
            print(f"[VPL step {step:4d}] loss={loss.item():.3f} recon={recon.item():.3f} kl={kl.item():.3f} "
                  f"| P_u0(s0>s_last)={p0:.2f}, P_u1(s0>s_last)={p1:.2f}")

    return embed, enc, rew

# -----------------------------
# 7) VPL-SPO scaled reward: r_tilde(s,z)=E_{sB}[p(y=1|s,sB,z)]
# -----------------------------
@torch.no_grad()
def spo_scaled_reward(rew: RewardModel,
                      s: int,
                      z: torch.Tensor,
                      num_states: int,
                      num_samples: int = 32):
    # sample comparator states sB
    ps = []
    for _ in range(num_samples):
        sB = random.randrange(num_states)
        if sB == s:
            continue
        p = torch.sigmoid(btl_logit(rew, s, sB, z))
        ps.append(p)
    if len(ps) == 0:
        return 0.5
    return torch.stack(ps).mean().item()

# vectorized (for batches) helper
@torch.no_grad()
def batch_spo_reward(rew: RewardModel,
                     s_batch: torch.Tensor,
                     z_batch: torch.Tensor,
                     num_states: int,
                     num_samples: int):
    """
    s_batch: [B] ints
    z_batch: [B, z_dim]
    returns r: [B] float tensor
    """
    B = s_batch.shape[0]
    out = torch.zeros(B)
    for i in range(B):
        out[i] = spo_scaled_reward(rew, int(s_batch[i].item()), z_batch[i:i+1], num_states, num_samples)
    return out

# -----------------------------
# 8) Active query selection (mutual information via BALD sampling)
# -----------------------------
def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-8):
    p = torch.clamp(p, eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

@torch.no_grad()
def mi_score_for_pair(enc: EncoderQ, rew: RewardModel, context: List[Tuple[int,int,int]],
                      pair: Tuple[int,int], cfg: Config):
    sA, sB = pair
    mu, logvar = enc(context)
    ps = []
    for _ in range(cfg.mi_z_samples):
        z = reparameterize(mu, logvar)
        p = torch.sigmoid(btl_logit(rew, sA, sB, z))
        ps.append(p)
    p = torch.stack(ps).squeeze()   # [K]
    p_bar = p.mean()
    # BALD: H(E[p]) - E[H(p)]
    return (bernoulli_entropy(p_bar) - bernoulli_entropy(p).mean()).item()

@torch.no_grad()
def select_active_queries(enc: EncoderQ, rew: RewardModel, context: List[Tuple[int,int,int]],
                          num_states: int, cfg: Config, k: int = 5):
    candidates = [(a,b) for a,b in itertools.permutations(range(num_states), 2)]
    scored = []
    for pair in candidates:
        scored.append((mi_score_for_pair(enc, rew, context, pair, cfg), pair))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]  # top-k

# -----------------------------
# 9) Conditional IQL (simple discrete version)
# -----------------------------
class QNet(nn.Module):
    def __init__(self, num_states: int, num_actions: int, z_dim: int, hidden: int):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(num_states + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
    def forward(self, s_onehot: torch.Tensor, z: torch.Tensor):
        x = torch.cat([s_onehot, z], dim=1)
        return self.net(x)  # [B, A]

class VNet(nn.Module):
    def __init__(self, num_states: int, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, s_onehot: torch.Tensor, z: torch.Tensor):
        x = torch.cat([s_onehot, z], dim=1)
        return self.net(x).squeeze(1)  # [B]

class PiNet(nn.Module):
    def __init__(self, num_states: int, num_actions: int, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_states + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
    def forward(self, s_onehot: torch.Tensor, z: torch.Tensor):
        x = torch.cat([s_onehot, z], dim=1)
        return self.net(x)  # logits

def expectile_loss(diff: torch.Tensor, expectile: float):
    # diff = q - v
    weight = torch.where(diff >= 0, expectile, 1 - expectile)
    return (weight * diff.pow(2)).mean()

def one_hot_batch(num_states: int, idx: torch.Tensor):
    # idx: [B]
    x = torch.zeros(idx.shape[0], num_states)
    x[torch.arange(idx.shape[0]), idx] = 1.0
    return x

def train_iql_conditional(env: GridWorld,
                          offline_D: List[Transition],
                          rew_model: RewardModel,
                          cfg: Config,
                          num_states: int,
                          num_actions: int):
    """
    Train pi(a|s,z) using offline IQL where reward is VPL-SPO scaled reward r_tilde(s,z).
    During training we sample z ~ N(0,I) (prior) per batch, like paper's multi-task training idea.
    """
    Q = QNet(num_states, num_actions, cfg.z_dim, cfg.q_hidden).to(cfg.device)
    V = VNet(num_states, cfg.z_dim, cfg.v_hidden).to(cfg.device)
    Pi = PiNet(num_states, num_actions, cfg.z_dim, cfg.pi_hidden).to(cfg.device)

    optQ = torch.optim.Adam(Q.parameters(), lr=cfg.iql_lr)
    optV = torch.optim.Adam(V.parameters(), lr=cfg.iql_lr)
    optPi = torch.optim.Adam(Pi.parameters(), lr=cfg.iql_lr)

    # convert dataset to tensors (CPU)
    S = torch.tensor([tr.s for tr in offline_D], dtype=torch.long)
    A = torch.tensor([tr.a for tr in offline_D], dtype=torch.long)
    SN = torch.tensor([tr.s_next for tr in offline_D], dtype=torch.long)
    Dn = torch.tensor([tr.done for tr in offline_D], dtype=torch.float32)

    n = S.shape[0]

    for step in range(1, cfg.iql_steps + 1):
        idx = torch.randint(0, n, (cfg.batch_size,))
        s = S[idx]
        a = A[idx]
        sn = SN[idx]
        done = Dn[idx]

        # sample z ~ N(0, I) per sample
        z = torch.randn(cfg.batch_size, cfg.z_dim)

        s_oh = one_hot_batch(num_states, s)
        sn_oh = one_hot_batch(num_states, sn)

        # SPO scaled reward on next state (typical r(s',z) labeling)
        r = batch_spo_reward(rew_model, sn, z, num_states, cfg.spo_num_comparators)

        # 1) Update V via expectile regression to Q
        with torch.no_grad():
            q_all = Q(s_oh, z)                    # [B, A]
            q_sa = q_all.gather(1, a.view(-1,1)).squeeze(1)  # [B]
        v_s = V(s_oh, z)                          # [B]
        v_loss = expectile_loss(q_sa - v_s, cfg.iql_expectile)

        optV.zero_grad()
        v_loss.backward()
        optV.step()

        # 2) Update Q via TD target: r + gamma * V(s')
        with torch.no_grad():
            v_sn = V(sn_oh, z)
            target = r + cfg.gamma * (1 - done) * v_sn
        q_all = Q(s_oh, z)
        q_sa = q_all.gather(1, a.view(-1,1)).squeeze(1)
        q_loss = F.mse_loss(q_sa, target)

        optQ.zero_grad()
        q_loss.backward()
        optQ.step()

        # 3) Update policy via advantage-weighted regression (discrete)
        with torch.no_grad():
            q_all = Q(s_oh, z)
            q_sa = q_all.gather(1, a.view(-1,1)).squeeze(1)
            v_s = V(s_oh, z)
            adv = q_sa - v_s
            w = torch.exp(torch.clamp(cfg.adv_beta * adv, max=cfg.adv_clip))  # [B]
        logits = Pi(s_oh, z)
        logp = F.log_softmax(logits, dim=1).gather(1, a.view(-1,1)).squeeze(1)
        pi_loss = -(w * logp).mean()

        optPi.zero_grad()
        pi_loss.backward()
        optPi.step()

        if step % 800 == 0:
            print(f"[IQL step {step:4d}] v_loss={v_loss.item():.3f} q_loss={q_loss.item():.3f} pi_loss={pi_loss.item():.3f}")

    return Pi

@torch.no_grad()
def rollout_policy(env: GridWorld, Pi: PiNet, z: torch.Tensor, start=None, steps=25):
    s = env.reset(start_state=start)
    total = 0
    for _ in range(steps):
        s_oh = one_hot_batch(env.num_states, torch.tensor([s]))
        logits = Pi(s_oh, z)
        a = torch.argmax(logits, dim=1).item()
        s, done = env.step(a)
        total += 1
        if done:
            break
    return s, total

# -----------------------------
# 10) End-to-end demo
# -----------------------------
def main():
    cfg = Config()

    env = GridWorld(N=5, max_steps=25)
    num_states, num_actions = env.num_states, env.num_actions

    # Define two diverse users with different goals (pluralistic preferences)
    users = {
        0: UserOracle(env, goal_state=0, noise_p=0.02),                 # top-left
        1: UserOracle(env, goal_state=num_states - 1, noise_p=0.02),    # bottom-right
    }

    # Build preference dataset per user
    pref_data = build_preference_dataset(users, num_pairs_per_user=250, num_states=num_states)

    # Train VPL reward model (phi, psi jointly)
    embed, enc, rew = train_vpl(pref_data, num_states, cfg, steps=3000, context_size=12)

    # Collect reward-free offline transitions
    offline_D = collect_offline_dataset(env, num_traj=300, behavior_eps=0.8)

    # Train conditional policy pi(a|s,z) with IQL using VPL-SPO rewards
    Pi = train_iql_conditional(env, offline_D, rew, cfg, num_states, num_actions)

    # ---- Test-time "deployment": active queries -> infer z -> run policy
    def infer_z_from_context(context):
        mu, logvar = enc(context)
        return mu  # use mean as point estimate

    for uid in users.keys():
        print("\n" + "="*70)
        print(f"TEST USER {uid} (goal={users[uid].goal})")

        # Start with empty context; actively ask a few queries
        context = []
        for t in range(6):
            topk = select_active_queries(enc, rew, context, num_states, cfg, k=3)
            score, (sA, sB) = topk[0]
            y = users[uid].prefers(sA, sB)
            context.append((sA, sB, y))
            z_hat = infer_z_from_context(context)
            print(f"  query{t+1}: (s{sA} vs s{sB}) y={y} MI≈{score:.4f} | ||z_hat||={z_hat.norm().item():.2f}")

        z_hat = infer_z_from_context(context)
        # Evaluate: roll out from random start and see where it ends
        end_s, steps = rollout_policy(env, Pi, z_hat, start=None, steps=25)
        dist_to_goal = manhattan(env.N, end_s, users[uid].goal)
        print(f"  rollout end_state={end_s}, steps={steps}, dist_to_goal={dist_to_goal}")

if __name__ == "__main__":
    main()
