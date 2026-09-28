# Offline fixed-gain comparison

This experiment compares alternative offline optimizers while holding the
existing hierarchical Bayesian preference estimates fixed. It leaves the
monthly meeting presentation and the default preference-loop pipeline intact.

Run from the repository root with the existing virtual environment:

```powershell
.venv\Scripts\python.exe -m unittest lab.offline_control.test_offline_models -v
.venv\Scripts\python.exe -u -m lab.offline_control.run_benchmark
```

The default source is `outputs/preference_loop/20260814_223618`. It contains
simulator episodes and synthetic user feedback, not a real-vehicle experiment.
The experiment checks that three saved episodes can be reproduced by the
current simulator before running the comparison.

## What is compared

| Method | Model | Objective used to select a gain |
|---|---|---|
| poly | Full quadratic response surface estimated by ordinary least squares | Predicted episode features, averaged over the fitting log's scenarios, projected onto the user's posterior mean |
| gp | Matérn 5/2 Gaussian process with per-input length scales and normalized outputs | Same posterior-mean reward objective |
| mlp | Separate scalar reward network per user, trained together in GPU batches | Predicted reward averaged over the same scenarios |
| coms | Same network and initialization, plus a conservative objective penalty | Same fixed-gain objective using the conservative reward model |

The GP has normalized-input bounds, a constant kernel amplitude, learned white
noise, and numerical diagonal regularization of 0.000001. Kernel parameters
are fit using marginal likelihood, without evaluating candidate gains in the
simulator. Its predictive mean is used here; no confidence-bound penalty is
applied.

The network has two 64-unit tanh hidden layers. Training uses Adam with learning
rate 0.001 and weight decay 0.00001, minibatches of 256 episodes, and a default
maximum of 5,000 updates. A validation split inside the fitting log selects the
training duration, followed by refitting on the full fitting log. MLP and COMs
use the same seeds, architecture, minibatch sequence, and training budget.

COMs implements the Eq. 3 idea from Trabucco et al. (2021): squared reward error
plus alpha times the difference between adversarial-input and logged-input
predictions. Alpha is fixed in advance at 0.1 after per-user reward
standardization. Ten ascent steps, with step size 0.02 in normalized gain units,
generate adversarial gains. Gains remain inside the controller bounds;
scenario covariates and user preference weights remain fixed. The adversarial
inputs are detached during the model update.

This is a contextual adaptation of the conservative training objective, not a
reproduction of the paper's entire benchmark or its trust-region design
optimizer. All four methods select from the same 271 gains, spaced one unit
apart from 30 to 300, so optimizer differences do not obscure model differences.
Each user receives one fixed gain for the whole episode.

## Separation of data and evaluation

- 100 existing training users supply 1,000 logged episodes.
- Each of three seeded splits assigns 80 users (800 episodes) to surrogate
  fitting and 20 users (200 episodes) to offline value evaluation. Validation
  users are selected inside the fitting partition.
- The 50 existing test users supply the same saved posterior means to every
  method. Their feedback trajectories never enter surrogate fitting.
- A fresh simulator bank evaluates all 271 gains on 128 common scenarios.
  These outcomes are used only after all models have selected their gains.
- True preference weights are used only for final performance measurement.
  The additive preference bias is excluded from episode rewards.

The saved hierarchical population posterior was originally fit on all 100
training users. Consequently the audit partition is excluded from surrogate
training, but was not excluded from that earlier population inference. This is
a comparison with fixed preference estimates, not an end-to-end independent
inference experiment or a formal confidence-interval study.

The three partitions overlap and reuse the same 50 users. Standard deviations
over partitions describe split sensitivity; they are not standard errors over
150 independent users or three independently collected populations.

## Continuous-action evaluation

Stage three independently evaluates the already selected gains. It does not
learn a Q function, run IQL, or add another policy-training stage.

The logger samples gain uniformly on [30,300], independently of the scenario,
so its density is known. Implemented estimators are kernel IPW and DM plus
kernel-weighted prediction residuals (DR). The residual models are fitted only
to the 800 fitting episodes. DM predictions are averaged over the 200 audit
contexts, and residual correction uses those same audit episodes.

The Gaussian kernel is analytically truncated and normalized at the two gain
bounds. Bandwidths of 10, 20, and 40 gain units are reported; 20 is the
predeclared primary comparison. At finite bandwidth, DR retains smoothing
bias. It does not provide an exact unbiased estimate of a deterministic
continuous action. Effective sample size is saved as an overlap diagnostic.
No audit result is used to select gains or tune models.

The audit compares reward estimates against simulator rewards using the same
posterior mean. This separates value-estimation error from preference-inference
error. Its errors still include finite-sample differences between audit
contexts and the 128 simulator scenarios.

## Outputs and interpretation

`summary.csv` reports actual simulator reward, regret, gain error, curve error,
and training time. Regret uses the best of the same 271 gains under the true
weights on the finite evaluation bank. This is an empirical reference, not a
proof of the globally optimal continuous gain. The simulator/posterior
reference uses the same bank and is an optimistic diagnostic, not an
independently evaluated online baseline.

`posterior_objective_regret` instead uses the estimated user weights for both
the selected policy and the simulator reference. It isolates gain selection
error from errors in those preference weights. `curve_rmse` compares the full
predicted reward curve against the simulator with estimated weights. Its
absolute level includes scenario-distribution sampling differences.

`continuous_summary.csv` reports MAE, RMSE, bias, and effective sample size for
the offline evaluators, averaged over gains proposed by the four methods.
The detailed user results, split assignments, model checkpoints, source-data
hashes, code hashes, and simulator scenario seeds are saved alongside it.

To reuse an already computed evaluation bank while changing training settings:

```powershell
.venv\Scripts\python.exe -u -m lab.offline_control.run_benchmark --output outputs/offline_control/another_run --simulator-bank outputs/offline_control/20260915_comparison/simulator_bank.npz
```

Both the gain grid and scenario seeds must match exactly. Settings should be
chosen using training/validation behavior, not the final simulator outcomes.

## References

- [Conservative Objective Models for Effective Offline Model-Based Optimization](https://proceedings.mlr.press/v139/trabucco21a.html), Trabucco et al., ICML 2021.
- [Policy Evaluation and Optimization with Continuous Treatments](https://proceedings.mlr.press/v84/kallus18a.html), Kallus and Zhou, AISTATS 2018.
- [Gaussian-process implementation](https://scikit-learn.org/stable/modules/gaussian_process.html), scikit-learn documentation.
