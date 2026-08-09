# Walker2d preference simulation

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
