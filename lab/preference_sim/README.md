# Walker2d preference simulation

## Linear gait experiment (current)

`run_linear_gaits.py` is the new experiment. `main.py` and its configs remain
the earlier offline experiment and are not the entry point for this version.

```powershell
.venv\Scripts\python.exe lab/preference_sim/run_linear_gaits.py `
  --config lab/preference_sim/configs/walker2d_role_exchange.yaml `
  --run-id linear_gaits_role_exchange
.venv\Scripts\python.exe lab/preference_sim/run_linear_gaits.py users validate `
  --run-id linear_gaits_role_exchange
.venv\Scripts\python.exe lab/preference_sim/test_linear_gaits.py
```

The five behaviors are upright walking, running, crouched walking, walking with
extended knees, and intentionally one-legged walking. Walker2d is planar:
"upright" means torso posture, not steering/yaw. The one-legged policy is an
intentional edge case, not a failure accepted into the ordinary bilateral set.

Fine-tuning and user utility use exactly the same 13 trajectory features:
`survival`, `progress`, `walk_speed`, `run_speed`, `upright`, `crouch`,
`straight_legs`, `smooth`, `flight`, `alternation`, `role_exchange`,
`stance_balance`, and `push_balance`. Repeated touchdowns by the same foot and
simultaneous landings do not count as alternation. `role_exchange` separately
requires the colored feet to exchange their world-x front/back ordering and
balances how long each leg leads. `stance_balance` compares exclusive left/right
stance time, and `push_balance` compares positive forward ground impulse.
Running, upright, and straight checkpoints use a deterministic periodic mirror
transform selected on validation seeds. It swaps both leg observations and leg
actions; reported features come from the resulting MuJoCo motion.

PPO receives the prefix difference `reward_t = weights @ (phi_prefix_t -
phi_prefix_previous)`. Its undiscounted episode sum is therefore exactly
`weights @ phi_episode`, which is also the synthetic-user utility. No original
Walker2d reward, healthy bonus, terminal penalty, or other hidden reward is
added. Survival and progress have explicit user coefficients rather than a
separate fixed base reward. Additive features use the fixed 1,000-step horizon;
early termination cannot receive full survival or progress. Style credit is
gated by forward progress, so standing still cannot collect it. The running
feature peaks near 2.1 m/s instead of rewarding unlimited speed.

Basic competence is an admissibility criterion for candidate policies: at
least 90% completion, 95% mean horizon fraction, and 0.7 m/s mean speed over
1,000 steps (8 seconds). The four ordinary gait policies must additionally
pass minimum alternation, role-exchange, stance-balance, and push-balance gates.
The one-leg policy instead has maximum stance/push gates so it remains
intentionally asymmetric.
Checkpoint screening and selection use separate seeds.
Final test reports include every attempted rollout, including falls, and test
seeds are never used for checkpoint selection. `gait_quality.json` reports
competence and quantitative style effects separately; a completed command
does not imply the gait quality gate passed.

The default population has 5,000 training and 1,000 test users, sampled from
one Gaussian in feature weights. Survival and progress weights have positive
population means and vary between users. Style weights vary broadly and may
be negative. Means are calibrated without looking at test trajectories.
Upright, running, and crouched occupy broad oracle regions; the current
straight policy overlaps them heavily and may receive few Gaussian users. Its
exact count is reported instead of being hidden by an archetype mixture. This
retains the hierarchical Gaussian modeling assumption and does not add a
common reward outside `w`.
A named `one_leg_fan` probe has strongly negative alternation/stance/push
weights. It is stored with split `probe` and is deliberately excluded from the
Gaussian population fit. A separate threshold
governs binary feedback: `p(good) = sigmoid(beta * (w @ phi - threshold))`.
Calibration uses its own trajectory seeds. A linear reference and Gaussian
threshold noise preserve joint Gaussianity of the effective coefficient vector.
The bias is part of that vector, with `Z @ theta` equal to the exact logit.
Both raw weights and effective coefficients are exported to avoid mixing scales.

`linear_preferences.npz` contains `Z`, raw `phi`, effective `theta_true`, raw
`weights_true`, thresholds, calibration transforms, binary labels, per-user
context indices, episode splits, and policy eligibility. Calibration labels
are unavailable (`-1`). Each user receives 120 independently selected context
trajectories. The 6,000 users share a trajectory bank; user count is not a claim
of 6,000 independently simulated trajectories per policy. Diagnostic zero-action
and noisy-policy episodes help identify competence weights but are explicitly
ineligible for policy selection. They are split by seed just like normal data.

The initial fully Bayesian check uses 128 train and 64 test Gaussian users from
the full export. It reports overall and competent-only AUROC, personalized
policy recovery, and regret. It also adapts the named probe separately and
reports whether inference selects the one-leg policy. This is an integration
check, not an MCMC convergence claim or a full-population fit.

Artifacts are in `data/runs/<run-id>/`: checkpoints, selected-policy metadata,
`rollouts/bank.npz` (raw signals, per-step features and valid masks), compact
preference exports, CSV tables, measured gait plots, and videos replaying the
first held-out seed without selecting attractive episodes.

Environment/API references: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/)
and [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

## Earlier experiment

This experiment builds an offline trajectory bank, simulates personalized binary
feedback, and exports the result for the repository's fully Bayesian model.

The important separation is:

1. policy reward weights are used only to create diverse trajectories;
2. synthetic-user weights independently determine trajectory utility;
3. the inference model receives trajectory signals and binary labels, not the
   synthetic ground-truth weights.

## Layout

```text
main.py                  single entry point (stage selection via arguments)
pipeline/                implementation modules, one per stage
configs/                 experiment configurations
data/runs/<run-id>/      generated runs (ignored by Git)
```

## Usage

### Competence-first experiment

The reliable workflow separates locomotion from preference optimization:

1. train the base policy with the original Walker2d reward;
2. stop only when deterministic competence evaluation passes;
3. fine-tune one preference at a time while retaining the original reward;
4. select checkpoints on separate validation seeds;
5. collect the final trajectory bank on previously unseen test seeds.

The first validated preference is smooth walking. The training run creates
candidate checkpoints, and the second command creates a self-contained final
run from the selected policies:

```powershell
uv run python lab/preference_sim/main.py train `
  --config lab/preference_sim/configs/walker2d_smooth_dense.yaml

uv run python lab/preference_sim/main.py select collect videos `
  --source-run competence_smooth_dense `
  --config lab/preference_sim/configs/walker2d_smooth_dense.yaml `
  --run-id competence_smooth_final
```

The competence gate checks completion rate, mean episode length, and forward
motion. The rollout gate additionally checks every profile rather than only
the pooled average. `checkpoint_selection.validation_seed` and
`collection.seed_offset` must remain different to preserve the validation/test
split.

Running `main.py` without stage arguments executes the full pipeline
(`train collect users reports export validate videos`):

```powershell
uv run python lab/preference_sim/main.py --config lab/preference_sim/configs/walker2d_style_smoke.yaml
```

Run directories are named after the config `name` (numeric suffix when taken);
no timestamps are recorded. Pass stage names to run a subset on an existing
run — the run's own `config.yaml` is used unless `--config` is given:

```powershell
uv run python lab/preference_sim/main.py export validate --run-id walker2d_smoke
uv run python lab/preference_sim/main.py videos --run-id walker2d_smoke --video-episodes 4
```

A trained policy bank can be reused without further RL training, for example to
collect two-second gait segments:

```powershell
uv run python lab/preference_sim/main.py --source-run walker2d_pilot `
  --config lab/preference_sim/configs/walker2d_segments.yaml
```

This separation allows horizon, rollout noise, synthetic-user count, and
feedback noise to change without retraining policies.

The `videos` stage replays the exact rollout seeds, so each mp4 shows an
episode that is literally in the dataset (frame count = episode length + 1).
Videos land in `<run>/reports/videos/`, including a labeled
`profiles_side_by_side.mp4` comparing the final checkpoint of every profile.

## Run contents

```text
<run>/
├─ config.yaml
├─ manifest.json
├─ policies/                 PPO checkpoints and scalarization metadata
├─ rollouts/                 compressed trajectory shards
├─ tables/
│  ├─ episodes.csv           trajectory metadata and episode features
│  ├─ users.csv              user split and ground-truth parameters
│  └─ feedback.csv           binary labels and query order
├─ exports/
│  └─ fully_bayesian_input.npz
└─ reports/                  rollout, feedback, inference checks, videos
```

`reports/trajectory_tradeoffs.png` visualizes policy-profile coverage, while
`reports/feature_correlation.csv` makes strongly redundant reward components
easy to identify before running feature-selection experiments.

`fully_bayesian_input.npz` contains trajectory signals, a user-by-episode binary
label matrix, query order, context masks, standardized episode features, and
ground-truth user weights for evaluation only.

The smoke configuration first learns one shared locomotion policy and then
fine-tunes four preference styles, saving two checkpoints per style. The full
configuration is intentionally much more expensive; use it
only after inspecting the smoke run's feature ranges and fall rate.

All reward components are signed so that larger is better: speed, efficiency,
stability, smoothness, low impact, and survival. Dense components and terminal
behavior are combined at trajectory level. The paper claim is therefore weight
recovery over an interpretable feature basis, not unique recovery of an
unrestricted reward function.
