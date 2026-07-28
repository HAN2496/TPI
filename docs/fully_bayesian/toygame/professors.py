import numpy as np
from scipy.special import expit  # sigmoid function
from polyagamma import random_polyagamma
import matplotlib.pyplot as plt

def pg_logistic_gibbs(F, y, mu0, Sigma0, n_samples=3000, burnin=1000, seed=0):
    """Gibbs sampler for y_i ~ Bernoulli(sigmoid(f_i^T phi))."""
    rng = np.random.default_rng(seed)
    
    N, d = F.shape

    # Prior precision and logistic-regression sufficient term
    Lambda0 = np.linalg.inv(Sigma0)
    kappa = y - 0.5

    # Initialize phi; zeros, MAP, or logistic regression all work.
    phi = np.zeros(d)
    samples = []

    for m in range(n_samples):
        # Current logit psi_i = f_i^T phi
        psi = F @ phi

        # Sample omega_i | phi ~ PG(1, psi_i)
        omega = random_polyagamma(h=1.0, z=psi)

        # Equivalent mathematical object:
        # Omega = diag(omega), but we do not build it explicitly.
        # Efficient: F.T @ Omega @ F = F.T @ (omega[:, None] * F)

        # Posterior precision: A = F^T Omega F + Lambda0
        A = F.T @ (omega[:, None] * F) + Lambda0

        # Posterior natural parameter: b = F^T kappa + Lambda0 mu0
        b = F.T @ kappa + Lambda0 @ mu0

        # Cholesky factorization of precision A
        L = np.linalg.cholesky(A)

        # mean = A^{-1} b
        mean = np.linalg.solve(L.T, np.linalg.solve(L, b))

        # Draw phi ~ N(mean, A^{-1}) without forming A^{-1}
        z = rng.standard_normal(d)
        phi = mean + np.linalg.solve(L.T, z)

        if m >= burnin:
            samples.append(phi.copy())

    return np.asarray(samples)


def predict_reward_probability(F_new, phi_samples):
    """
    Posterior predictive reward probability and uncertainty.
    F_new:       N_new x d feature matrix
    phi_samples: M x d posterior samples
    """
    # probs[j,m] = sigmoid(f_new_j^T phi_m)
    probs = expit(F_new @ phi_samples.T)

    mean_prob = probs.mean(axis=1)
    std_prob = probs.std(axis=1, ddof=1)
    q05, q95 = np.quantile(probs, [0.05, 0.95], axis=1)

    # Law of total variance for future binary feedback y_*
    aleatoric_var = (probs * (1.0 - probs)).mean(axis=1)
    epistemic_var = probs.var(axis=1, ddof=1)
    total_var = aleatoric_var + epistemic_var

    return mean_prob, std_prob, q05, q95, aleatoric_var, epistemic_var, total_var

# Example data: F is N x d, y is binary {0,1}
N, d = 200, 4
rng = np.random.default_rng(1)
F = rng.normal(size=(N, d))
true_phi = np.array([1.0, -0.7, 0.5, 0.2])
y = rng.binomial(1, expit(F @ true_phi))

# Weakly informative Gaussian prior
mu0 = np.zeros(d)
Sigma0 = 10.0 * np.eye(d)

phi_samples = pg_logistic_gibbs(
    F, y, mu0, Sigma0,
    n_samples=4000, burnin=1000, seed=3
)

F_new = rng.normal(size=(5, d))
out = predict_reward_probability(F_new, phi_samples)
mean_p, std_p, q05, q95, v_a, v_e, v_total = out

print(mean_p)   # posterior mean reward probabilities
print(std_p)    # variability of probability due to phi uncertainty
print(v_a)      # aleatoric variance E[p(1-p)]
print(v_e)      # epistemic variance Var[p]
print(v_total)  # total predictive variance for binary feedback y_*