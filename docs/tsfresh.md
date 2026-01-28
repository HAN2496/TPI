# Explicit Feature-Based Latent Space and Preference Learning

This document details the mathematical framework and pipeline implemented in `src/model/tsfresh/` and `scripts/train_tsfresh.py`. The objective is to construct an interpretable, non-deep learning model for learning user preferences from time-series driving data.

The pipeline consists of three main stages:
1.  **Explicit Feature Extraction** (via `tsfresh`)
2.  **Common Latent Space Construction** (via PCA)
3.  **User-Specific Adaptation** (via Logistic Regression)

---

## 1. Mathematical Formulation

### 1.1. Input Data
Let $\mathcal{D}$ be a dataset of driving episodes. Each episode $i$ is a multivariate time-series matrix $X_i$:

$$ X_i \in \mathbb{R}^{T \times D} $$

where:
*   $T$: Number of time steps (e.g., 40 steps after downsampling).
*   $D$: Number of sensor channels (e.g., `IMU_VerAccelVal`, `Pitch_rate`, etc.).
*   $X_{i}^{(d)} \in \mathbb{R}^T$: The univariate time-series for the $d$-th sensor of episode $i$.

### 1.2. Explicit Feature Extraction ($\Phi$)
Unlike deep learning encoders that learn implicit features, we apply a predefined set of statistical mappings, denoted as $\Phi$, to transform the raw time-series into a high-dimensional explicit feature vector.

$$ \mathbf{f}_i = \Phi(X_i) = \left[ \phi_1(X_i^{(1)}), \dots, \phi_K(X_i^{(D)}) \right]^\top \in \mathbb{R}^M $$

where $M$ is the total number of extracted features (often $M \approx 10^3$).

Examples of mapping functions $\phi$ provided by `tsfresh` include:

*   **Moments**: Mean $\mu = \frac{1}{T}\sum x_t$, Variance $\sigma^2 = \frac{1}{T}\sum (x_t - \mu)^2$.
*   **FFT Coefficients**: The magnitude of the $k$-th frequency component:
    $$ |\mathcal{F}(k)| = \left| \sum_{t=0}^{T-1} x_t e^{-j 2\pi k t / T} \right| $$
*   **Autocorrelation**: Correlation of the series with itself at lag $\tau$:
    $$ R(\tau) = \frac{\mathbb{E}[(x_t - \mu)(x_{t+\tau} - \mu)]}{\sigma^2} $$
*   **Approximate Entropy**: A measure of regularity and unpredictability in the series.

**Preprocessing:**
Since $\mathbf{f}_i$ may contain NaNs (due to short sequence length) or constant values, we apply:
1.  **Imputation**: Replace NaNs with column medians.
2.  **Variance Thresholding**: Remove features $\phi_j$ where $\text{Var}(\phi_j) = 0$.
3.  **Standardization**: Normalize features to zero mean and unit variance.

$$ \tilde{\mathbf{f}}_i = \frac{\mathbf{f}_i - \boldsymbol{\mu}_f}{\boldsymbol{\sigma}_f} $$

### 1.3. Common Latent Space Construction (PCA)
The "Latent Space" is defined as the subspace spanned by the principal components of the population's feature vectors. This captures the **common modes of variation** across all drivers.

We compute the Principal Component Analysis (PCA) on the aggregated training set $\mathcal{X}_{train} = \{ \tilde{\mathbf{f}}_j \}_{j=1}^{N}$.

We seek a projection matrix $W_{pca} \in \mathbb{R}^{M \times K}$ that maximizes the variance of the projected data, where $K \ll M$.

$$ W_{pca} = \underset{W}{\text{argmax}} \ \text{Tr}(W^\top \Sigma W) \quad \text{s.t.} \quad W^\top W = I $$

where $\Sigma$ is the covariance matrix of $\tilde{\mathbf{f}}$.
In our pipeline, $K$ is chosen such that the explained variance ratio $\ge 0.95$.

The **Latent Vector** $\mathbf{z}_i$ for episode $i$ is:

$$ \mathbf{z}_i = W_{pca}^\top \tilde{\mathbf{f}}_i \in \mathbb{R}^K $$

Here, each dimension of $\mathbf{z}_i$ represents an orthogonal, interpretable axis of driving behavior (e.g., "Intensity of Stop-and-Go", "High-frequency Vibration").

### 1.4. User Adaptation (Preference Learning)
We model a specific user $u$'s preference (Good vs. Bad) as a linear decision boundary within this latent space.

For a test user $u$, we have a labeled dataset $\mathcal{D}_u = \{ (\mathbf{z}_j, y_j) \}$, where $y_j \in \{0, 1\}$.
We assume the probability of the user labeling an episode as "Good" ($y=1$) is given by a **Logistic Regression** model:

$$ P(y_j=1 | \mathbf{z}_j; \mathbf{w}_u, b_u) = \sigma( \underbrace{\mathbf{w}_u^\top \mathbf{z}_j + b_u}_{\text{Reward } r_j} ) $$

*   $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.
*   **$\mathbf{w}_u \in \mathbb{R}^K$ (User Weights)**: Represents the user's "taste". A positive weight on the $k$-th component means the user prefers behaviors exhibiting that specific latent feature.
*   **$r_j = \mathbf{w}_u^\top \mathbf{z}_j + b_u$**: This scalar value is interpreted as the **estimated reward** of the episode.

The weights are learned by minimizing the binary cross-entropy loss:

$$ \mathcal{L}(\mathbf{w}_u, b_u) = - \sum_{j} \left[ y_j \log \hat{y}_j + (1-y_j) \log (1-\hat{y}_j) \right] $$

---

## 2. Visualization and Interpretation

### 2.1. t-SNE vs. PCA for Visualization
While the model ($\mathbf{w}_u$) is learned in the PCA space ($\mathbb{R}^K$), visualizing $K$-dimensional data ($K \approx 30 \sim 50$) is difficult. We use **t-SNE (t-Distributed Stochastic Neighbor Embedding)** to project $\mathbf{z}_i$ into $\mathbb{R}^2$ strictly for visualization.

$$ \mathbf{z}_i \in \mathbb{R}^K \xrightarrow{\text{t-SNE}} \mathbf{v}_i \in \mathbb{R}^2 $$

*   **Cluster Structure**: Points close in the t-SNE plot share similar high-dimensional statistics (similar driving contexts).
*   **Separability**: If "Good" (Red) and "Bad" (Blue) points form distinct clusters or gradients in the t-SNE plot, it indicates that the explicit features ($\mathbf{f}$) and the latent representation ($\mathbf{z}$) successfully capture the factors influencing user satisfaction.

### 2.2. Reward Distribution
We visualize the distribution of the estimated rewards $r_j = \mathbf{w}_u^\top \mathbf{z}_j + b_u$.

*   **Ideal Case**: Two disjoint distributions (e.g., Bad centered at -2, Good centered at +2).
*   **Overlap**: Indicates ambiguity in the preference or limitations in the feature set's ability to describe the user's criteria.

---

## 3. Pipeline Summary

1.  **Extract**: Raw Series $(N, T, D) \xrightarrow{\text{tsfresh}} \text{Features } (N, M)$.
2.  **Refine**: Impute NaNs $\rightarrow$ Remove Constant Features $\rightarrow$ Standard Scale.
3.  **Compress**: PCA $(N, M) \rightarrow \text{Latents } (N, K)$ (preserving 95% variance).
4.  **Adapt**: Train Logistic Regression on $(\mathbf{z}, y)$ to find $\mathbf{w}_u$.
5.  **Visualize**: Use t-SNE on $\mathbf{z}$ to inspect data structure and preference separability.
