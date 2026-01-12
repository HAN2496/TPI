# VPL Pairwise: Variational Preference Learning with Pairwise Comparisons

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Dataset Creation](#dataset-creation)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Binary Feedback Inference](#binary-feedback-inference)
7. [Implementation Details](#implementation-details)

---

## Overview

### What is VPL?

**VPL (Variational Preference Learning)** is a framework for learning personalized reward functions from human feedback. The key idea is to use a **Variational Autoencoder (VAE)** to:

1. **Encode** user preferences into a latent representation $z$ (driver style)
2. **Decode** observations with latent $z$ to predict rewards $r(s, z)$

### Binary VPL vs Pairwise VPL

| Aspect | Binary VPL | Pairwise VPL |
|--------|-----------|--------------|
| **Input** | Episodes with binary labels (True/False) | Episode pairs with preferences (A > B, A = B) |
| **Training Signal** | Absolute classification | Relative comparison |
| **Loss Function** | Binary Cross-Entropy on P(label \| episode) | Bradley-Terry-Luce on P(A > B) |
| **Encoder Input** | $K$ episodes + labels: $(s_1, y_1), ..., (s_K, y_K)$ | $K$ pairs: $(s_a^1, s_b^1, p_1), ..., (s_a^K, s_b^K, p_K)$ |
| **Decoder** | $r(s, z) \rightarrow$ reward | Same: $r(s, z) \rightarrow$ reward |

**Key Advantage of Pairwise:**
- Richer training signal from comparisons
- More robust to labeling noise
- Better generalization with tie pairs (A = B)

---

## Theoretical Foundation

### Problem Formulation

Given:
- Driver episodes: $\mathcal{D} = \{(s_i, y_i)\}_{i=1}^N$
  - $s_i \in \mathbb{R}^{T \times d}$: Trajectory with $T$ timesteps, $d$ features
  - $y_i \in \{0, 1\}$: Binary label (False=0, True=1)

Goal:
- Learn a **driver-specific latent variable** $z \in \mathbb{R}^{D}$ (latent dimension $D$)
- Learn a **reward function** $r_\theta(s, z): \mathbb{R}^{T \times d} \times \mathbb{R}^{D} \rightarrow \mathbb{R}^T$

### Variational Framework

**Encoder (Inference Network):**
$$q_\phi(z | \mathcal{X}) = \mathcal{N}(z; \mu_\phi(\mathcal{X}), \sigma^2_\phi(\mathcal{X}))$$

where $\mathcal{X}$ is a set of $K$ annotated queries (pairwise comparisons).

**Decoder (Reward Model):**
$$r_\theta(s_t, z) = \text{MLP}_\theta([s_t, z])$$

Each timestep $t$ gets a reward independently.

**Prior:**
$$p(z) = \mathcal{N}(z; 0, I)$$

Standard Gaussian prior.

---

## Dataset Creation

### Step 1: Binary Labels → Pairwise Comparisons

Given binary labeled episodes:
- True episodes: $\mathcal{E}_{\text{True}} = \{s_i | y_i = 1\}$, $n_{\text{True}}$ episodes
- False episodes: $\mathcal{E}_{\text{False}} = \{s_j | y_j = 0\}$, $n_{\text{False}}$ episodes

#### Phase 1: True vs False Pairs (Primary Signal)

For each True episode $s_i^{\text{True}}$, sample $M$ False episodes:

$$\text{Pairs}_{\text{primary}} = \{(s_i^{\text{True}}, s_j^{\text{False}}, p=1.0) | i \in [1, n_{\text{True}}], j \sim \text{sample}(n_{\text{False}}, M)\}$$

- Preference $p = 1.0$ means "True > False"
- Total primary pairs: $n_{\text{True}} \times M$

**Implementation:**
```python
# src/utils/vpl_pairwise_dataset.py, lines 76-89
for i in range(n_true):
    n_samples = min(n_false, max_pairs_per_true)  # M
    false_indices = np.random.choice(n_false, size=n_samples, replace=False)

    for j in false_indices:
        all_pairs.append({
            'obs_a': true_episodes[i],
            'obs_b': false_episodes[j],
            'preference': 1.0  # True > False
        })
```

#### Phase 2: Tie Pairs (Augmentation)

Add $N_{\text{tie}} = \alpha \times |\text{Pairs}_{\text{primary}}|$ tie pairs (default $\alpha = 0.25$ = 20%):

**True vs True:**
$$\text{Pairs}_{\text{tie}}^{\text{TT}} = \{(s_i^{\text{True}}, s_j^{\text{True}}, p=0.5) | i \neq j\}$$

**False vs False:**
$$\text{Pairs}_{\text{tie}}^{\text{FF}} = \{(s_i^{\text{False}}, s_j^{\text{False}}, p=0.5) | i \neq j\}$$

- Preference $p = 0.5$ means "tie" (no preference)
- Helps model learn invariance to spurious differences

**Implementation:**
```python
# src/utils/vpl_pairwise_dataset.py, lines 93-118
num_tie_pairs = int(driver_primary_pairs * tie_ratio)

# True vs True
for _ in range(num_tie_pairs // 2):
    idx_a, idx_b = np.random.choice(n_true, size=2, replace=False)
    all_pairs.append({
        'obs_a': true_episodes[idx_a],
        'obs_b': true_episodes[idx_b],
        'preference': 0.5  # Tie
    })

# False vs False (similar)
```

#### Phase 3: Group into Queries

Shuffle all pairs and group into queries of size $K$ (annotation size):

$$\mathcal{Q} = \{\{(s_a^1, s_b^1, p_1), ..., (s_a^K, s_b^K, p_K)\}_1, \{(s_a^1, s_b^1, p_1), ..., (s_a^K, s_b^K, p_K)\}_2, ...\}$$

Each query contains $K$ pairs (e.g., $K=10$).

**Final Dataset Shape:**
```
obs_a:       (num_queries, K, T, d)
obs_b:       (num_queries, K, T, d)
preferences: (num_queries, K, 1)
driver_ids:  (num_queries,)
```

**Example:**
- 50 True, 50 False episodes
- Primary pairs: $50 \times 10 = 500$
- Tie pairs: $500 \times 0.25 = 125$
- Total: $625$ pairs → $62$ queries (K=10)

---

## Model Architecture

### Encoder: Pairwise Observations → Latent z

**Input:**
- $s_a \in \mathbb{R}^{B \times K \times T \times d}$: First episodes in pairs
- $s_b \in \mathbb{R}^{B \times K \times T \times d}$: Second episodes in pairs
- $p \in \mathbb{R}^{B \times K \times 1}$: Preferences

**Step 1: Flatten episodes**
$$s_a^{\text{flat}} = \text{flatten}(s_a) \in \mathbb{R}^{B \times K \times (T \cdot d)}$$
$$s_b^{\text{flat}} = \text{flatten}(s_b) \in \mathbb{R}^{B \times K \times (T \cdot d)}$$

**Step 2: Concatenate pairwise information**
$$x_{\text{pair}} = [s_a^{\text{flat}}, s_b^{\text{flat}}, p] \in \mathbb{R}^{B \times K \times (2Td + 1)}$$

**Step 3: Flatten all pairs**
$$x_{\text{encoder}} = \text{flatten}(x_{\text{pair}}) \in \mathbb{R}^{B \times K(2Td + 1)}$$

With $K=10, T=40, d=4$:
$$\text{encoder\_input\_dim} = 10 \times (2 \times 40 \times 4 + 1) = 10 \times 321 = 3210$$

**Step 4: MLP encoder**
$$h = \text{ReLU}(\text{Linear}_{3210 \rightarrow 256}(x_{\text{encoder}}))$$
$$\mu, \log\sigma^2 = \text{Linear}_{256 \rightarrow 2D}(h)$$

Split output into mean and log-variance.

**Output:**
$$z \sim q_\phi(z | s_a, s_b, p) = \mathcal{N}(\mu_\phi, \sigma^2_\phi)$$

**Implementation:**
```python
# src/model/vpl/vae_pairwise.py, lines 36-59
def encode_pairwise(self, s_a, s_b, pref):
    # Flatten episodes
    s_a_flat = s_a.reshape(s_a.shape[0], s_a.shape[1], -1)  # (B, K, T*d)
    s_b_flat = s_b.reshape(s_b.shape[0], s_b.shape[1], -1)
    pref = pref.reshape(s_a.shape[0], s_a.shape[1], -1)     # (B, K, 1)

    # Concatenate: [obs_a, obs_b, preference]
    encoder_input = torch.cat([s_a_flat, s_b_flat, pref], dim=-1).reshape(
        s_a.shape[0], -1
    )  # (B, K * (2*T*d + 1))

    mean, log_var = self.Encoder(encoder_input)
    return mean, log_var
```

### Decoder: Observation + Latent z → Reward

**Input:**
- $s \in \mathbb{R}^{B \times K \times T \times d}$: Observations
- $z \in \mathbb{R}^{B \times D}$: Latent variable

**Step 1: Expand z to match timesteps**
$$z_{\text{expanded}} = \text{repeat}(z) \in \mathbb{R}^{B \times K \times T \times D}$$

**Step 2: Concatenate with observations**
$$x_{\text{decoder}} = [s, z_{\text{expanded}}] \in \mathbb{R}^{B \times K \times T \times (d + D)}$$

With $d=4, D=32$:
$$\text{decoder\_input\_dim} = 4 + 32 = 36$$

**Step 3: MLP decoder (per timestep)**
$$r_t = \text{MLP}_\theta(x_{\text{decoder}, t}) \in \mathbb{R}$$

Each timestep independently:
$$h_t = \text{ReLU}(\text{Linear}_{36 \rightarrow 256}([s_t, z]))$$
$$r_t = \text{Linear}_{256 \rightarrow 1}(h_t)$$

**Output:**
$$r(s, z) = [r_1, r_2, ..., r_T] \in \mathbb{R}^{B \times K \times T \times 1}$$

**Implementation:**
```python
# Inherited from src/model/vpl/vae.py
def decode(self, x, z):
    obs_z = torch.cat([x, z], dim=-1)  # (B, K, T, d+D)
    rewards = self.Decoder(obs_z)       # (B, K, T, 1)
    return rewards
```

---

## Training Process

### Bradley-Terry-Luce (BTL) Model

The **Bradley-Terry-Luce model** for pairwise comparisons:

$$P(A \succ B | s_a, s_b, z) = \sigma\left(\frac{R(s_a, z) - R(s_b, z)}{\tau}\right)$$

where:
- $R(s, z) = \sum_{t=1}^T r_t(s, z)$: Total reward over trajectory
- $\tau$: Temperature (reward scaling), typically $\tau = T$
- $\sigma(x) = \frac{1}{1 + e^{-x}}$: Sigmoid function

### Loss Function

**Total Loss:**
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

#### 1. Reconstruction Loss (Bradley-Terry-Luce)

For each pair $(s_a, s_b, p)$ in a query:

**Step 1: Compute total rewards**
$$R_a = \sum_{t=1}^T r_t(s_a, z) / \tau$$
$$R_b = \sum_{t=1}^T r_t(s_b, z) / \tau$$

**Step 2: Compute logits**
$$\text{logit} = R_a - R_b$$

**Step 3: Binary cross-entropy**
$$\mathcal{L}_{\text{recon}} = -\sum_{i=1}^{B \times K} \left[p_i \log \sigma(\text{logit}_i) + (1-p_i) \log(1 - \sigma(\text{logit}_i))\right]$$

Using `binary_cross_entropy_with_logits` for numerical stability:
$$\mathcal{L}_{\text{recon}} = \text{BCE}(\text{logit}, p)$$

**Note on preference values:**
- $p = 1.0$: Episode A preferred over B (True > False)
- $p = 0.5$: Tie (True = True or False = False)
- $p = 0.0$: Episode B preferred over A (not used in our dataset)

**Implementation:**
```python
# src/model/vpl/vae_pairwise.py, lines 87-102
# Decode both episodes
r_a = self.decode(s_a, z_expanded)  # (B, K, T, 1)
r_b = self.decode(s_b, z_expanded)  # (B, K, T, 1)

# Sum over timesteps
r_a_sum = r_a.sum(dim=2) / self.scaling  # (B, K, 1), scaling = T
r_b_sum = r_b.sum(dim=2) / self.scaling  # (B, K, 1)

# Bradley-Terry-Luce probability
logits = (r_a_sum - r_b_sum).reshape(-1, 1)
labels = pref.reshape(-1, 1)

# Binary cross-entropy
reconstruction_loss = nn.functional.binary_cross_entropy_with_logits(
    logits, labels, reduction='sum'
)
```

#### 2. KL Divergence Loss

Regularize latent distribution to match prior:

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q_\phi(z | \mathcal{X}) \| p(z))$$

For Gaussian distributions:
$$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{D} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

**KL Annealing:**
To prevent posterior collapse, use cyclical annealing:

$$\beta(t) = \beta_{\max} \cdot \text{schedule}(t)$$

Cosine schedule (default):
$$\text{schedule}(t) = \frac{1 - \cos(\pi t / T_{\text{cycle}})}{2}$$

- Starts at 0, increases to 1 over $T_{\text{cycle}}$ steps
- Repeats for multiple cycles

**Implementation:**
```python
# src/model/vpl/vae_pairwise.py, lines 109-113
latent_loss = self.latent_loss(mean, log_var)

# Annealing
kl_weight = self.annealer.slope() if self.annealer else self.kl_weight
loss = reconstruction_loss + kl_weight * latent_loss
```

### Training Accuracy

**Metric:**
$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\sigma(\text{logit}_i) > 0.5 \text{ and } p_i = 1] + \mathbb{1}[\sigma(\text{logit}_i) < 0.5 \text{ and } p_i = 0]$$

For tie pairs ($p=0.5$), model is correct if $|\sigma(\text{logit}) - 0.5| < \epsilon$.

**Expected Performance:**
- Random: 50%
- Good model: 70-85%
- Tie pairs lower accuracy ceiling

---

## Binary Feedback Inference

### Problem: Pairwise-Trained Model with Binary Feedback

**Challenge:**
- Model trained with pairwise encoder: $q_\phi(z | s_a, s_b, p)$
- At inference, we only have binary labeled episodes: $(s_i, y_i)$
- Encoder expects pairwise format (3210 dims), but binary format is different

**Solution: Two-Stage Inference**

#### Stage 1: Estimate Latent z from Binary Feedback

**Given:**
- Binary episodes: $\{(s_i, y_i)\}_{i=1}^{N}$
- Split: $N_z$ episodes for z estimation

**Step 1: On-the-fly Pairwise Conversion**

Temporarily convert to pairwise format:

**True vs False pairs:**
$$\mathcal{P}_{\text{TF}} = \{(s_i^{\text{True}}, s_j^{\text{False}}, 1.0) | i \in [1, n_{\text{True}}], j \sim \text{sample}(n_{\text{False}}, M)\}$$

**Tie pairs:**
$$\mathcal{P}_{\text{TT}} = \{(s_i^{\text{True}}, s_j^{\text{True}}, 0.5) | i \neq j\}$$
$$\mathcal{P}_{\text{FF}} = \{(s_i^{\text{False}}, s_j^{\text{False}}, 0.5) | i \neq j\}$$

**Step 2: Encode with Pairwise Encoder**

Group into queries of size $K$:
$$z_1 \sim q_\phi(z | \mathcal{Q}_1), \quad z_2 \sim q_\phi(z | \mathcal{Q}_2), \quad ...$$

**Step 3: Average Latent Means**

$$\bar{z} = \frac{1}{M} \sum_{i=1}^{M} \mu_i$$

This gives a single driver-specific latent vector.

**Implementation:**
```python
# src/model/vpl/inference.py, lines 195-290
def estimate_z_from_binary_feedback(model, episodes, labels, set_len=10, device='cpu'):
    # Separate True and False
    true_episodes = episodes[labels == 1]
    false_episodes = episodes[labels == 0]

    # Create True vs False pairs
    for i in range(n_true):
        false_indices = np.random.choice(n_false, size=min(n_false, 10), replace=False)
        for j in false_indices:
            all_pairs.append({
                'obs_a': true_episodes[i],
                'obs_b': false_episodes[j],
                'preference': 1.0
            })

    # Create tie pairs (20%)
    num_tie = int(len(all_pairs) * 0.25)
    # True vs True and False vs False...

    # Group into queries and encode
    all_means = []
    for i in range(0, len(all_pairs), set_len):
        query_pairs = all_pairs[i:i+set_len]
        # Stack and encode
        mean, logvar = model.encode_pairwise(obs_a, obs_b, prefs)
        all_means.append(mean.cpu().numpy())

    # Average all query means
    z_mean = np.concatenate(all_means, axis=0).mean(axis=0)
    return z_mean
```

#### Stage 2: Predict Rewards with Decoder

**Given:**
- Estimated latent: $\bar{z}$
- Remaining episodes: $\{s_i\}_{i=N_z+1}^{N}$

**Decoder as Reward Model:**

For each episode $s_i$:
$$r_i = r_\theta(s_i, \bar{z}) = [r_1, r_2, ..., r_T]$$

**Classification Score:**
$$\text{score}_i = \frac{1}{T} \sum_{t=1}^{T} r_t$$

Higher mean reward → predict True, Lower → predict False.

**Implementation:**
```python
# src/model/vpl/inference.py, lines 293-317
def predict_rewards_with_z(model, episodes, z_latent, device='cpu'):
    all_rewards = []

    for episode in episodes:
        # Use existing compute_reward_for_episode function
        rewards = compute_reward_for_episode(model, episode, z_latent, device)
        all_rewards.append(rewards)

    return np.array(all_rewards)  # (N, T)

# Per episode reward:
def compute_reward_for_episode(model, episode_data, z_latent, device='cpu'):
    obs = torch.tensor(episode_data).float().to(device)  # (T, d)
    z = torch.tensor(z_latent).float().to(device)        # (D,)

    # Expand z to match timesteps
    z_expanded = z.unsqueeze(0).repeat(obs.shape[0], 1)  # (T, D)
    obs_z = torch.cat([obs, z_expanded], dim=-1)         # (T, d+D)

    # Decode
    rewards = model.Decoder(obs_z).squeeze(-1).cpu().numpy()  # (T,)
    return rewards
```

### Complete Inference Pipeline

```python
# src/model/vpl/inference.py, lines 321-373
def infer_and_predict_from_binary_feedback(
    model, episodes, labels,
    z_estimation_ratio=0.5,
    set_len=10,
    device='cpu'
):
    # Split episodes
    n_total = len(episodes)
    n_z_estimation = max(2, int(n_total * z_estimation_ratio))

    indices = np.random.permutation(n_total)
    z_indices = indices[:n_z_estimation]
    pred_indices = indices[n_z_estimation:]

    z_episodes = episodes[z_indices]
    z_labels = labels[z_indices]
    pred_episodes = episodes[pred_indices]

    # Stage 1: Estimate z
    z_estimated = estimate_z_from_binary_feedback(
        model, z_episodes, z_labels, set_len, device
    )

    # Stage 2: Predict rewards
    predictions = predict_rewards_with_z(
        model, pred_episodes, z_estimated, device
    )

    return z_estimated, predictions, pred_indices
```

### Evaluation Metrics

**1. AUROC (Area Under ROC Curve)**

Threshold-free metric using mean rewards as scores:

$$\text{AUROC} = P(\text{score}(s^+) > \text{score}(s^-))$$

where $s^+$ is a True episode, $s^-$ is a False episode.

**2. Threshold-Based Metrics**

Using median reward as threshold $\tau$:

$$\hat{y}_i = \begin{cases} 1 & \text{if } \text{mean}(r_i) > \tau \\ 0 & \text{otherwise} \end{cases}$$

- **Accuracy:** $\frac{1}{N} \sum_i \mathbb{1}[\hat{y}_i = y_i]$
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall:** $\frac{TP}{TP + FN}$
- **F1 Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

**Implementation:**
```python
# scripts/train_model_vpl_pairwise.py, lines 356-405
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, ...

# AUROC
auroc = roc_auc_score(actual_labels, mean_rewards)

# ROC curve
fpr, tpr, thresholds = roc_curve(actual_labels, mean_rewards)

# Threshold-based
threshold = np.median(mean_rewards)
pred_labels = (mean_rewards > threshold).astype(int)

accuracy = accuracy_score(actual_labels, pred_labels)
precision = precision_score(actual_labels, pred_labels)
recall = recall_score(actual_labels, pred_labels)
f1 = f1_score(actual_labels, pred_labels)
```

---

## Implementation Details

### File Structure

```
TPI/
├── src/
│   ├── model/vpl/
│   │   ├── vae.py                  # Base VAE model (Binary VPL)
│   │   ├── vae_pairwise.py         # Pairwise VAE extension
│   │   ├── trainer.py              # Base trainer + Annealer
│   │   ├── trainer_pairwise.py     # Pairwise trainer
│   │   └── inference.py            # Inference functions
│   ├── utils/
│   │   ├── vpl_dataset.py          # Binary VPL dataset
│   │   ├── vpl_pairwise_dataset.py # Pairwise dataset creation
│   │   └── visualization.py        # Plotting functions
├── scripts/
│   ├── train_model_vpl.py          # Train Binary VPL
│   └── train_model_vpl_pairwise.py # Train Pairwise VPL
└── docs/
    └── vpl_pairwise_explanation.md # This file
```

### Training Command

```bash
python scripts/train_model_vpl_pairwise.py \
    --drivers all \
    --test-drivers 조현석 \
    --time_range "[5,7]" \
    --downsample 5 \
    --device cuda \
    --set_len 10 \
    --max_pairs_per_true 10 \
    --tie_ratio 0.25 \
    --latent_dim 32 \
    --hidden_dim 256 \
    --epochs 500 \
    --batch_size 64 \
    --lr 1e-3 \
    --use_annealing \
    --annealer_type cosine \
    --annealer_cycles 4 \
    --early_stop \
    --patience 10 \
    --test_binary_inference \
    --z_estimation_ratio 0.5
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `set_len` | 10 | Number of pairs per query (annotation size) |
| `max_pairs_per_true` | 10 | Max False episodes paired with each True |
| `tie_ratio` | 0.25 | Ratio of tie pairs (20% of primary) |
| `latent_dim` | 32 | Latent dimension $D$ |
| `hidden_dim` | 256 | Hidden layer size in encoder/decoder |
| `kl_weight` | 1.0 | $\beta$ for KL term (without annealing) |
| `reward_scaling` | T | $\tau$ for BTL model (default: trajectory length) |
| `lr` | 1e-3 | Learning rate |
| `batch_size` | 64 | Batch size |
| `use_annealing` | False | Enable KL annealing |
| `annealer_type` | cosine | Annealing schedule (linear/cosine/logistic) |
| `annealer_cycles` | 4 | Number of annealing cycles |
| `z_estimation_ratio` | 0.5 | Fraction for z estimation in binary inference |

### Expected Dataset Sizes

**Example: 7 drivers, 100 episodes each (50 True, 50 False)**

Per driver:
- Primary pairs: $50 \times 10 = 500$
- Tie pairs: $500 \times 0.25 = 125$
- Total pairs: $625$
- Queries (K=10): $62$

Total dataset:
- Queries: $7 \times 62 = 434$
- Train/val split (90/10): $390$ train, $44$ val
- Batch size 64: $\approx 6$ batches per epoch

### Model Size

**Encoder:**
- Input: 3210 → Hidden: 256 → Output: 64 (mean + logvar)
- Parameters: $3210 \times 256 + 256 \times 64 \approx 840K$

**Decoder:**
- Input: 36 → Hidden: 256 → Output: 1
- Parameters: $36 \times 256 + 256 \times 1 \approx 9K$

**Total:** ~850K parameters

### Training Performance

**Typical metrics after 500 epochs:**
- Train Loss: ~100-150
- Val Loss: ~120-180
- Val Accuracy: 70-85%
- AUROC (binary inference): 0.75-0.90

**Training time (GPU):**
- ~30-60 seconds per epoch
- Total: ~15-30 minutes for 500 epochs

---

## Mathematical Derivations

### KL Divergence for Gaussians

Given:
- $q(z) = \mathcal{N}(\mu_q, \sigma_q^2)$
- $p(z) = \mathcal{N}(0, 1)$

The KL divergence:

$$D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} dz$$

For Gaussians:

$$D_{\text{KL}}(q \| p) = \frac{1}{2} \left[\log\frac{\sigma_p^2}{\sigma_q^2} - 1 + \frac{\sigma_q^2}{\sigma_p^2} + \frac{(\mu_q - \mu_p)^2}{\sigma_p^2}\right]$$

With $\mu_p = 0, \sigma_p = 1$:

$$D_{\text{KL}}(q \| p) = \frac{1}{2} \left[-\log\sigma_q^2 - 1 + \sigma_q^2 + \mu_q^2\right]$$

$$= -\frac{1}{2} \left[1 + \log\sigma_q^2 - \mu_q^2 - \sigma_q^2\right]$$

For $D$ dimensions:

$$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{D} \left[1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right]$$

### Reparameterization Trick

To sample $z \sim \mathcal{N}(\mu, \sigma^2)$ in a differentiable way:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This allows gradients to flow through $\mu$ and $\sigma$.

**Implementation:**
```python
def reparameterization(self, mean, std):
    epsilon = torch.randn_like(std)
    z = mean + std * epsilon
    return z
```

### Bradley-Terry-Luce Gradient

Loss:
$$\mathcal{L}_{\text{BTL}} = -\left[p \log \sigma(\Delta R) + (1-p) \log(1 - \sigma(\Delta R))\right]$$

where $\Delta R = R_a - R_b$.

Gradient w.r.t. $\Delta R$:
$$\frac{\partial \mathcal{L}_{\text{BTL}}}{\partial \Delta R} = \sigma(\Delta R) - p$$

- If $p = 1$ (A > B) and $\Delta R < 0$ (model predicts B > A): gradient pushes $\Delta R$ up
- If $p = 0.5$ (tie) and $|\Delta R|$ is large: gradient pushes $\Delta R$ toward 0

---

## Visualization Outputs

### 1. Training Curves
- Loss, Reconstruction Loss, KL Loss over epochs
- Train vs Val accuracy
- KL weight schedule (if annealing)

### 2. Latent Space
- PCA projection of driver latents (if multiple test drivers)
- Or first 2 latent dimensions (if single driver)

### 3. ROC Curve (Binary Inference)
- True Positive Rate vs False Positive Rate
- AUROC value
- Diagonal reference (random classifier)

### 4. Reward Distribution (Binary Inference)
- Histogram: True vs False episodes
- Box plot: Label comparison
- Classification threshold (median)

---

## References

1. **Bradley-Terry-Luce Model:** Bradley, R. A., & Terry, M. E. (1952). "Rank analysis of incomplete block designs: I. The method of paired comparisons."
2. **Variational Autoencoders:** Kingma, D. P., & Welling, M. (2013). "Auto-encoding variational bayes."
3. **KL Annealing:** Bowman, S. R., et al. (2015). "Generating sentences from a continuous space."
4. **Preference Learning:** Christiano, P. F., et al. (2017). "Deep reinforcement learning from human preferences."

---

## Appendix: Dimension Tracking

**Training Forward Pass:**

```
Input batch:
  obs_a:       (64, 10, 40, 4)   [B, K, T, d]
  obs_b:       (64, 10, 40, 4)
  preferences: (64, 10, 1)

Encoder:
  obs_a_flat:  (64, 10, 160)     [B, K, T*d] = [64, 10, 40*4]
  obs_b_flat:  (64, 10, 160)
  pref:        (64, 10, 1)
  concat:      (64, 10, 321)     [B, K, 2*T*d+1] = [64, 10, 160+160+1]
  flatten:     (64, 3210)        [B, K*(2*T*d+1)]

  mean:        (64, 32)          [B, D]
  logvar:      (64, 32)
  z:           (64, 32)

Decoder:
  z_expanded:  (64, 10, 40, 32)  [B, K, T, D]
  obs_a:       (64, 10, 40, 4)
  concat:      (64, 10, 40, 36)  [B, K, T, d+D]

  r_a:         (64, 10, 40, 1)   [B, K, T, 1]
  r_b:         (64, 10, 40, 1)

  r_a_sum:     (64, 10, 1)       [B, K, 1] (sum over T, divide by T)
  r_b_sum:     (64, 10, 1)

  logits:      (640, 1)          [B*K, 1]
  labels:      (640, 1)

Loss:
  recon_loss:  scalar (sum over all 640 pairs)
  kl_loss:     scalar (sum over 64 queries * 32 dims)
  total_loss:  scalar
```

**Binary Inference:**

```
Input:
  episodes:    (200, 40, 4)      [N, T, d]
  labels:      (200,)            [N]

Split (50/50):
  z_episodes:  (100, 40, 4)
  z_labels:    (100,)
  pred_episodes: (100, 40, 4)

Stage 1: Estimate z
  Create ~1000 pairs from 100 episodes
  Group into 100 queries (K=10)
  Encode each query → z_i (32,)
  Average → z_mean (32,)

Stage 2: Predict
  For each episode in pred_episodes (100,):
    rewards: (40,)
    mean_reward: scalar

  Output:
    predictions: (100, 40)
    mean_rewards: (100,)

Metrics:
  AUROC using mean_rewards vs actual_labels
```
