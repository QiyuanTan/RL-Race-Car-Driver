# RL-Race-Car-Driver

We built and tested race car controllers on the Austria track in [racecar_gym](https://github.com/axelbr/racecar_gym). The project includes a rule-based Follow-the-Gap baseline, DQN with several observation modes, and PPO and SAC implementations.

## Branches

We submit the `final` branch. The table shows where each part of the project was developed.

| Branch | Adds | Built on |
| --- | --- | --- |
| `main`, `SAC` | _TODO (Qiyuan)_ | - |
| `DQN` | _TODO (Qiyuan)_ | `main` |
| `baseline-Elaine` | Follow-the-Gap driver, `evaluate_baseline.py`, unit tests | `SAC` |
| `jx_smoothness_metrics` | Added motor and steering smoothness metrics to measure changes in driving actions | `baseline-Elaine` |
| `dqn-lidar` | DQN + LiDAR, action repeat, eval / resume checkpoints, `evaluate_sb3.py`, random starts, FTG demos | `jx_smoothness_metrics`, `DQN` |
| `dqn-track` | Track-relative observations (`--obs-mode track`, `track-lookahead`) | `dqn-lidar` |
| `dqn-lidar-hairpin` | Hairpin fine-tuning options and analysis tools | `dqn-lidar` |
| `jx_dqn_improvement` | Added signed progress reward, FTG demonstrations for track-lookahead DQN, Behavior Cloning with motor and steering losses, and BC-regularized DQN training | `dqn-track` |
| `comparison-matrix` | Performance comparison matrix, unified evaluation metrics, lap completion rate, lap time, collision rate, and control smoothness comparison | `dqn-lidar` |
| `comparison-interface-track` | Comparison interface for track-relative DQN agents, integration with the evaluation framework, and standardized performance comparison across agents | `jx_dqn_improvement` |
| `final` | `comparison-interface-track` with `dqn-lidar-hairpin` merged in | both |

The final branch merges the hairpin work into `comparison-interface-track`, which already has the newer comparison document.

## Setup

Use a Python 3.9 environment (the DQN experiments used Python 3.9). The upstream `racecar_gym` repository may have compatibility issues with newer Python versions.

Run the following commands, which will set up racecar_gym and install the required dependencies:

```bash
git clone https://github.com/axelbr/racecar_gym.git
pip install -e racecar_gym
pip install -r requirements.txt
```

Run the unit tests:

```bash
python -m pytest tests -q
```

## Non-RL baseline: Follow-the-Gap

Follow-the-Gap uses LiDAR and vehicle speed to find an open gap and steer towards it. Our turn-in rule helps the car start turning before it reaches the wall at the hairpin. FTG needs no training. These commands test it with and without the rule:

```bash
python evaluate_baseline.py --episodes 10 --seed 100 --workers 4
python evaluate_baseline.py --episodes 10 --seed 100 --workers 4 --param turn_in_distance=0  # plain FTG
```

FTG keeps driving after wall contact, so we count contacts over the full run. Results go to `results/baseline_ftg/`. See [BASELINE.md](BASELINE.md) for the baseline results.

## Model Definition

Models are implemented with Stable-Baselines3 and wrapped by `SB3Driver` in
`Agents/SB3Drivers.py`. The driver creates the model, applies environment wrappers,
and exposes `train`, `save`, `load`, and `act` methods.

`SB3ObservationWrapper` converts scalar `Box` observations into one-element
vectors to satisfy SB3's requirements.

The action wrappers are defined in `Agents/wrappers.py`. The continuous wrapper
preserves the original action bounds and converts predictions back to the
environment's action dictionary. The discrete wrapper combines motor levels
`(-0.5, 0.0, 0.5)` with steering levels `(-1.0, -0.5, 0.0, 0.5, 1.0)`.
Steering varies fastest, so an action ID is decoded as
`motor_index * 5 + steering_index`.

| Algorithm | Action wrapper | Action representation |
| --- | --- | --- |
| PPO | `FlattenActionWrapper` | Continuous motor and steering commands in a flat `Box` |
| SAC | `FlattenActionWrapper` | Continuous motor and steering commands in a flat `Box` |
| DQN | `DiscreteActionWrapper` | 15 discrete combinations of motor and steering commands |



## Training

Run commands from the repository root after installing the dependencies:

```bash
python train.py
```

`train.py` provides the following helpers. 

| Helper | Default save path | TensorBoard log directory |
| --- | --- | --- |
| `train_ppo` | `ppo_model.zip` | `logs/ppo/` |
| `train_dqn` | `dqn_model.zip` | `logs/dqn/` |
| `train_sac` | `sac_model.zip` | `logs/sac/` |

To select another algorithm, change the call inside the `if __name__ == '__main__':`
block in `train.py`, creating a fresh environment for each training run:

```python
env = gym_api.SingleAgentRaceEnv(
    scenario=scenario, render_mode='rgb_array_birds_eye'
)
train_ppo(env, total_timesteps=1_000_000, save_path='models/ppo_model')
# Alternatively, use train_sac(env, ..., save_path='models/sac_model').
```

PPO can also train with multiple environment processes. Replace the active
environment creation and training call in the main block with:

```python
env_factories = [
    partial(
        gym_api.SingleAgentRaceEnv,
        scenario=scenario,
        render_mode='rgb_array_birds_eye',
    )
    for _ in range(4)
]
train_ppo(env_factories, total_timesteps=1_000_000, save_path='models/ppo_model')
```

`SB3Driver` wraps these factories with `SubprocVecEnv` using the `spawn` start
method. Keep this code under the main guard for multiprocessing on Windows.
With vectorized environments, `total_timesteps` counts transitions across all
environments, rather than separately for each process. The DQN helper uses a
single environment.

DQN's replay buffer defaults to `10_000` transitions, independently of the
training duration. Adjust it with `train_dqn(..., buffer_size=...)` when needed.
For 128 × 128 RGB `uint8` images, current and next images alone occupy about
0.92 GiB at this buffer size; other observations and training require additional
memory.

### DQN with LiDAR observations

Running `python train.py` without flags trains base DQN for 1M steps and saves `models/dqn_model.zip`. Add `--lidar` to use `scenarios/austria_lidar.yml` and give DQN the LiDAR distances scaled to `[0, 1]`. The two commands below use the same actions, reward, episode rules, and budget.

```bash
python train.py --timesteps 200000 --seed 0 --save-path models/dqn_model
python train.py --lidar --timesteps 200000 --seed 0 --save-path models/dqn_lidar_model
```

Checkpoints and run settings go in `models/checkpoints/<save-path name>/`.

- `--eval-every N` saves a model checkpoint every N decisions for evaluation.
- `--resume-every N` saves the latest model and replay buffer. To resume, add `--resume` to the same command and keep `--timesteps` the same.
- `config.json` records the training settings.

```bash
python train.py --timesteps 200000 --seed 0 --save-path models/dqn_model --eval-every 20000 --resume-every 100000
python train.py --timesteps 200000 --seed 0 --save-path models/dqn_model --eval-every 20000 --resume-every 100000 --resume
```

- `--action-repeat N` holds each action for N simulator steps. The training budget, replay buffer size, and target-network update interval count decisions. Use the same action repeat when evaluating a model.
- `--train-start random` gives the car practice from different points on the track. Evaluation starts from the grid.
- `--demo-episodes K` fills part of the LiDAR replay buffer with FTG runs using the same actions and action repeat as DQN. `--demo-laps K` keeps complete laps only.

These commands evaluate the base and LiDAR models on the same test seeds used for FTG:

```bash
python evaluate_sb3.py --model models/dqn_model.zip --episodes 10 --seed 100 --workers 4
python evaluate_sb3.py --model models/dqn_lidar_model.zip --lidar --episodes 10 --seed 100 --workers 4
```

DQN episodes stop at the first wall contact. Progress shows how far the car got before the episode ended.

### DQN with track-relative / look-ahead observations

`--obs-mode track` adds the car's offset, heading, and velocity relative to the centre line. `--obs-mode track-lookahead` also gives it centre line points 1–8 m ahead. Both use `scenarios/austria_centerline.npz`. `--lidar` is equivalent to `--obs-mode lidar`. See [DQN_TRACK.md](DQN_TRACK.md) for the setup and commands.

We also tested several methods to improve track-lookahead DQN. First, we added a signed progress reward to reward forward movement instead of backward movement. We then used FTG driving demonstrations to train the model with Behavior Cloning (BC). We added motor and steering losses to help the model learn driving actions. We also tested BC-regularized DQN, which keeps an imitation loss during DQN training. The training scripts are train_ftg_bc.py and train_ftg_bc_rl.py. We used evaluate_ftg_bc.py to evaluate these models.

### DQN hairpin fine-tuning

The LiDAR DQN reached the first hairpin at about 35% progress but could not get through it. We tried these options to train it on the turn:

- `--init-from <model.zip>` loads a trained model and starts with 5% exploration. Match the observation mode and action repeat used in training.
- `--train-start hairpin` starts some training episodes just before the turn. `--hairpin-share` defaults to 0.5.
- `--crash-penalty P` subtracts P from the reward on a wall hit that ends the episode.
- `--takeover-demos K` collects K successful FTG takeovers from the trained LiDAR model. It uses `--takeover-points` and samples a share of each batch from the demos, set by `--demo-fraction`.
- `--margin-loss W` adds a DQfD loss on demo samples. `--demo-pretrain-steps N` runs N updates on demos before regular training.

The settings are saved in `config.json`. Evaluation still starts from the grid.

```bash
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_hairpin --train-start hairpin --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
```

See [DQN_LIDAR.md](DQN_LIDAR.md) for the hairpin experiments, results, and analysis commands.

To view training metrics, run:

```bash
tensorboard --logdir logs
```

For inference, `run_sim.py` provides a simple interface to load a trained model and run it in the simulator.


## Track-Lookahead DQN: Yutao Lin's Fine-Tuning Experiment (Fall 2026)

This section documents **Yutao Lin's individual track-lookahead DQN experiment**,
not a final training run or model for the entire team. It builds on the
track-relative feature representation developed by another team member.
Both the Stage 1 model and this subsequent fine-tuning experiment use the same
15-action discrete motor/steering space, `track-lookahead` observations, and
`action_repeat=4` (four simulator steps per DQN decision). The fine-tuning
experiment initializes from Yutao's earlier Stage 1 checkpoint.

### Models from this experiment

| Model | Purpose | Grid evaluation (10 episodes, seeds 100–109) |
| --- | --- | --- |
| `models/dqn_centerline_stage1_300k.zip` | Original Stage 1 baseline, trained for 300k steps | Historical mean progress ~35.1%, best progress ~35.4% |
| `models/dqn_final_best_75k.zip` | Selected candidate among 20 checkpoints from this fine-tuning experiment | Mean maximum progress **35.45%**, maximum **35.5%**, minimum **35.3%** |

This experiment fine-tuned Stage 1 for **500,000 additional decisions** with
reward shaping and mixed starting positions. The 75k checkpoint (75k new steps,
not 75k total training steps) had the highest mean Grid progress among the
20 evaluated checkpoints. Its performance improvement over the historical Stage 1
result is small; **the original baseline was not successfully re-evaluated in the
same checkpoint batch because of a saved training-configuration metadata mismatch**.
Therefore, these figures are descriptive rather than a verified head-to-head
improvement. Neither model has demonstrated a completed lap.

### Fine-tuning changes

The standalone fine-tuning script is `train_dqn_final_stall_v2_fixed.py`. It builds on
Stage 1 with:

- **Progress-oriented reward shaping:** forward progress remains the main
  incentive, while lateral/heading misalignment, excessive cornering speed,
  steering/motor oscillations, and wall collisions receive penalties.
- **Stall detection:** sustained low speed and insufficient net track progress
  are penalized. Prolonged stuck episodes can be truncated, instead of treating
  momentary braking or coasting as a stall.
- **Mixed-start curriculum:** 40% Grid, 35% hairpin-area randomized starts
  (`0.26–0.31` environment progress), and 25% other random starts. This was
  intended to preserve early-track performance while exposing the policy to
  the challenging turn.
- **Checkpointing:** evaluation model snapshots every 25k decisions and
  resumable state snapshots every 10k decisions in the archived run.

The fine-tuning settings were: learning rate `3e-5`, replay buffer `20,000`,
batch size `64`, target update interval `5,000`, discount factor `0.99`, and
exploration rate linearly decreased from `0.15` to `0.03` over the first 35%
of the 500k-step experiment. A smoke test was run before training.

To reproduce the **training configuration** from a retained Stage 1 model,
run from the repository root (the script saves to a new prefix and should
not overwrite existing models):

```powershell
python train_dqn_final_stall_v2_fixed.py `
  --source-model models/dqn_centerline_stage1_300k.zip `
  --save-path models/dqn_final_stall_reproduction `
  --timesteps 500000 `
  --seed 0 `
  --action-repeat 4 `
  --buffer-size 20000 `
  --batch-size 64 `
  --learning-rate 0.00003 `
  --target-update-interval 5000 `
  --initial-eps 0.15 `
  --final-eps 0.03 `
  --exploration-fraction 0.35 `
  --grid-prob 0.40 `
  --hairpin-prob 0.35 `
  --start-min 0.26 `
  --start-max 0.31 `
  --target-progress 0.42 `
  --max-reset-attempts 600 `
  --lateral-weight 0.12 `
  --heading-weight 0.12 `
  --speed-weight 0.018 `
  --steer-smooth-weight 0.004 `
  --motor-smooth-weight 0.002 `
  --curve-speed 1.0 `
  --collision-penalty 1.0 `
  --stall-window-steps 60 `
  --stall-speed-threshold 0.2 `
  --stall-progress-threshold 0.0001 `
  --stall-penalty 0.03 `
  --stall-timeout-steps 200 `
  --oscillation-timeout-steps 250 `
  --curve-grace-steps 40 `
  --eval-every 25000 `
  --resume-every 10000
```

> **Note:** This standalone script creates checkpoints during training, but a
> `--resume` command for this standalone training script has not been verified.
> The `train.py --resume` workflow described above is separate.

### Checkpoint evaluation and limitations

For this experiment, we evaluated **20 fine-tuning checkpoints** at 25k intervals, each with 10 Grid
starts (seeds 100–109), for **200 Grid evaluation episodes**. The best checkpoint
was the 75k fine-tuning snapshot:

| Fine-tuning checkpoint | Mean maximum Grid progress | Maximum Grid progress |
| --- | ---: | ---: |
| 75k (selected candidate) | **35.45%** | **35.5%** |
| 175k | 35.40% | 35.5% |
| 275k | 35.34% | 35.5% |
| 300k | 35.17% | 35.2% |
| 500k | 35.24% | 35.4% |

No evaluated checkpoint completed a lap. The dominant failure remains a sharp
hairpin around 36% environment progress, and this fine-tuning did not establish a
reliable solution to this turn. Local geometry analysis estimated a minimum
centre-line radius of about 1.23 m; separate short-turn tests measured transient
turn radii as low as ~0.98 m. These tests do **not** prove that a complete,
collision-free path through the hairpin is physically achievable. Speed-only
interventions and heuristic Pure Pursuit control also failed to pass the turn
in the exploratory tests. A hairpin-only 100k fine-tuning experiment exhibited
substantial Grid-performance regression, motivating the mixed-start design.

The checkpoint comparison summary is retained at:

`results/final_checkpoint_grid_eval/checkpoint_comparison.csv`

The full exploratory logs and intermediate checkpoints may be kept in a separate
local archive to avoid bloating the shared Git repository. If checkpoint
artifacts have been archived, `evaluate_final_checkpoints.py` requires the
checkpoint directory to be restored before a full 20-checkpoint evaluation.

### Evaluation compatibility

`evaluate_sb3.py` compares the requested environment against training metadata.
The Stage 1 model previously produced a mismatch involving `features.obs_mode`
(`base` versus `track-lookahead`) and missing saved feature keys. This is a
**configuration-metadata issue that must be investigated**, not a reason to
silently bypass feature checks. Before evaluating or comparing a model, confirm
that observation dimensions **and semantics**, action mapping, action repeat,
centerline data, and preprocessing match those used in training. The original
Stage 1 evaluation numbers above come from earlier tests, not from this
20-checkpoint batch.
## Shared evaluation interface

`evaluate_shared.py` evaluates FTG, DQN (including `base`, `lidar`, `track`, and `track-lookahead` modes), and PPO/SAC with the existing `evaluate_baseline.run_episode` metrics. For the behavior-cloning geometry-only DQN, use `--obs-mode track-lookahead --geometry-only`. Models must be evaluated with their training-time observation mode, action repeat and feature settings. Saved-model observation and action spaces are checked before evaluation; geometry-only models have no complete configuration check, so the user must supply matching feature settings.

Examples (install the project dependencies and racecar_gym first):

```bash
python evaluate_shared.py --algorithm ftg --episodes 10 --seed 100
python evaluate_shared.py --algorithm dqn --model path/to/dqn.zip --obs-mode lidar --action-repeat 4
python evaluate_shared.py --algorithm dqn --model path/to/student.zip --obs-mode track-lookahead --geometry-only --action-repeat 4
python evaluate_shared.py --algorithm ppo --model models/ppo_model_w_collision.zip
python evaluate_shared.py --algorithm sac --model path/to/sac.zip
```

Each episode uses a fresh process and JSON records the settings, summary and per-episode metrics. These are a common interface, not yet a fully aligned experiment: FTG uses `SingleAgentAustria-v0` and continues after wall contact; the SB3 scenarios terminate at the first contact. Smoothness per simulator step is affected by action repeat. Keep these differences in mind when comparing results.

### Testing a trained model

Get the model `.zip` file and, if available, its `config.json` from the person
who trained it. Make sure you know the model's observation mode and action repeat.

For DQN checkpoints, keep the files in this structure so the evaluator can
find the configuration:

```text
models/checkpoints/run_name/
├── config.json
└── eval/
    └── model.zip
```

Run one episode first to make sure the model loads. For example, to test a
LiDAR DQN trained with action repeat 4:

```bash
python evaluate_shared.py --algorithm dqn \
  --model models/checkpoints/run_name/eval/model.zip \
  --obs-mode lidar --action-repeat 4 --episodes 1 --seed 100
```

Change the model path, observation mode, and action repeat to match your run.
The DQN observation modes are `base`, `lidar`, `track`, and `track-lookahead`.
For a geometry-only model, use `--obs-mode track-lookahead --geometry-only`.

Once it works, use `--episodes 10 --seed 100` to test seeds 100–109. Results
are saved in `results/shared/` as JSON.

If the evaluator can't find `config.json`, check the training settings with
your teammate. If the model uses new inputs, actions, or custom code, we may
need to update the evaluator first.

FTG and RL currently use different collision rules, and action repeat affects
smoothness. Keep that in mind when comparing results.

## Results

_TODO: final comparison table (see COMPARISON_MATRIX.md)._

The initial FTG/DQN comparisons used grid-start evaluation seeds 100–109, with DQN checkpoints selected on validation seeds 0–9. The separately developed track-lookahead DQN experiments used a different evaluation procedure: the Stage 1 result is historical, while the fine-tuned 75k checkpoint was **selected using seeds 100–109**. Its reported progress is therefore a checkpoint-selection result, **not an independent held-out test**. Progress values below are mean lap progress (or mean maximum lap progress for the track-lookahead rows).

| Method | Decision every | Checkpoint | Progress (min-max) | Laps | Collision-free laps |
| --- | --- | --- | --- | --- | --- |
| FTG (turn-in rule) | 1 step | - | 100% | 10/10, 81.77 ± 0.77 s | 10/10 |
| FTG (plain) | 1 step | - | 100% | 10/10, 68.56 ± 0.35 s | 2/10 |
| Base DQN | 1 step | 20k | 12.6% (12.2-12.8) | 0/10 | 0/10 |
| DQN + LiDAR | 1 step | 200k | 8.7% (7.4-9.4) | 0/10 | 0/10 |
| Base DQN | 4 steps | 200k | 21.9% (9.2-35.5) | 0/10 | 0/10 |
| DQN + LiDAR | 4 steps | 140k | 35.3% (34.9-35.4) | 0/10 | 0/10 |
| DQN + track | 4 steps | 140k | 34.4% | 0/10 | 0/10 |
| LiDAR, hairpin fine-tuning (grid starts / hairpin starts / crash penalty) | 4 steps | validation | 35.4% | 0/10 | 0/10 |
| LiDAR, takeover demos + margin loss | 4 steps | validation | 17.7% | 0/10 | 0/10 |
| Track-lookahead DQN — Stage 1 (Yutao) | 4 steps | 300k | ~35.1% (historical; range not available) | 0/10 | 0/10 |
| Track-lookahead DQN — mixed-start fine-tuning (Yutao) | 4 steps | 75k additional (selected from 500k run) | **35.45%** (35.3–35.5) | 0/10 | 0/10 |

In these runs, no DQN variant completed a lap; the first hairpin remained a major failure point. FTG could navigate it, motivating additional DQN analysis. Yutao's track-lookahead fine-tuning selected the 75k checkpoint from 20 candidates, but did not establish a reliable improvement over the historical Stage 1 baseline; see the individual experiment section above for details.

## Use of AI tools

We used AI tools to draft unit tests and organise the Markdown documentation. We reviewed the tests and ran the experiments ourselves. The numbers above come from the files in `results/`.

- TODO: any other AI use for their own part

## References

- racecar_gym: https://github.com/axelbr/racecar_gym
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- T. Hester et al. Deep Q-learning from Demonstrations. AAAI 2018 (margin loss).
- _TODO: other references._
