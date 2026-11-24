# RL-Race-Car-Driver

## Setup

Use a Python 3.9 environment (the DQN experiments used Python 3.9). The upstream `racecar_gym` repository may have compatibility issues with newer Python versions.

Run the following commands, which will set up racecar_gym and install the required dependencies:

```bash
git clone https://github.com/axelbr/racecar_gym.git
pip install -e racecar_gym
pip install -r requirements.txt
```

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

`train.py` takes command-line options. Without options it still trains base DQN for 1M steps
and saves `models/dqn_model.zip`. `--lidar` uses `scenarios/austria_lidar.yml` (same as
`austria.yml` plus the `lidar` sensor) and adds `LidarObservationWrapper`, which scales the
1080 distances (metres) to `[0, 1]` float32. Actions, reward, termination and budget are
unchanged, so the two variants differ only in the LiDAR input.

```bash
python train.py --timesteps 200000 --seed 0 --save-path models/dqn_model
python train.py --lidar --timesteps 200000 --seed 0 --save-path models/dqn_lidar_model
```

Optional checkpoints (off by default), stored in `models/checkpoints/<save-path name>/`:

- `--eval-every N`: every N steps save `eval/<name>_<step>_steps.zip` (model only, about 40 MB),
  which `evaluate_sb3.py --model ...` can evaluate.
- `--resume-every N`: every N steps save `resume/` (model + replay buffer, about 1 GB),
  keeping only the latest copy. After an interruption, run the same command with `--resume`
  (same `--timesteps`): the step count and exploration schedule continue from the saved step.
  Steps after the last resumable checkpoint are redone.
- `config.json` records the run settings (variant, scenario, timesteps, seed, buffer size).

```bash
python train.py --timesteps 200000 --seed 0 --save-path models/dqn_model --eval-every 20000 --resume-every 100000
python train.py --timesteps 200000 --seed 0 --save-path models/dqn_model --eval-every 20000 --resume-every 100000 --resume
```

`--action-repeat N` holds each DQN decision for N simulator steps (N=4 -> 25 Hz, the LiDAR
update rate). SB3 then counts decisions, so `--timesteps`, `--buffer-size` and
`--target-update-interval` are in decisions; `config.json` also records `simulated_steps`.
Evaluate such a model with the same `--action-repeat` (checked against `config.json`).
Smoothness is measured per simulator step, so it is only comparable between runs with the
same action repeat.

`--train-start random` starts each training episode at a random point of the track
(racecar_gym's `random` reset mode, facing forward), so later sections such as the hairpin are
practised too. Evaluation always starts from the grid. Recorded as `train_start` in `config.json`.

`--demo-episodes K` (LiDAR only) prefills the replay buffer with K episodes driven by FTG
restricted to DQN's discrete actions and the same action repeat (seeds 1000, 1001, ...). The
demo transitions occupy the first slots of the buffer for the whole run; DQN's own transitions
use the remaining `buffer_size - demo decisions` slots. `config.json` records the demo episodes,
decisions and simulated steps next to `online_decisions` (DQN's own training decisions).
`--demo-laps K` instead keeps only demo episodes that complete a full lap (failed tries are
discarded, at most 10 tries); the discarded seeds are listed in `config.json`.

Evaluate with the same episode loop as the FTG baseline (`evaluate_baseline.run_episode`);
seeds 100-109 match the FTG test seeds:

```bash
python evaluate_sb3.py --model models/dqn_model.zip --episodes 10 --seed 100 --workers 4
python evaluate_sb3.py --model models/dqn_lidar_model.zip --lidar --episodes 10 --seed 100 --workers 4
```

The DQN scenarios end an episode at the first wall contact, so `collisions` is 0 or 1 and
`progress` shows how far the car got.

### DQN with track-relative / look-ahead observations

`--obs-mode track` adds 5 numbers describing the car relative to the track centre line (offset,
heading error, velocity along/across the track); `--obs-mode track-lookahead` also adds the
centre line points 1-8 m ahead in the car's frame. Both use a centre line extracted from
racecar_gym's map (`scenarios/austria_centerline.npz`), so these are map-based policies. The old
`--lidar` flag still works (`--obs-mode lidar`). Details, checks and commands: [DQN_TRACK.md](DQN_TRACK.md).

To view training metrics, run:

```bash
tensorboard --logdir logs
```

For inference, `run_sim.py` provides a simple interface to load a trained model and run it in the simulator.


## DQN Final Experiment (Fall 2026)

This repository includes a DQN racing baseline and a final reward-shaped training run
on the Austria track. The final experiments retain the same 15-action discrete
motor/steering space, `track-lookahead` observations, and `action_repeat=4`
(four simulator steps per DQN decision). The final training run uses the earlier
Stage 1 model as its initialization.

### Models to retain

| Model | Purpose | Grid evaluation (10 episodes, seeds 100–109) |
| --- | --- | --- |
| `models/dqn_centerline_stage1_300k.zip` | Original Stage 1 baseline, trained for 300k steps | Historical mean progress ~35.1%, best progress ~35.4% |
| `models/dqn_final_best_75k.zip` | Best candidate among the 20 checkpoints from the final training run | Mean maximum progress **35.45%**, maximum **35.5%**, minimum **35.3%** |

The final run fine-tuned Stage 1 for **500,000 additional decisions** with
reward shaping and mixed starting positions. The 75k checkpoint (75k new steps,
not 75k total training steps) had the highest mean Grid progress among the
20 evaluated checkpoints. Its performance improvement over the historical Stage 1
result is small; **the original baseline was not successfully re-evaluated in the
same checkpoint batch because of a saved training-configuration metadata mismatch**.
Therefore, these figures are descriptive rather than a verified head-to-head
improvement. Neither model has demonstrated a completed lap.

### Final DQN training changes

The final training script is `train_dqn_final_stall_v2_fixed.py`. It builds on
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
of the 500k-step run. A smoke test was run before full training.

To reproduce the **training configuration** from a retained Stage 1 model,
run from the repository root (the final script saves to a new prefix and should
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

> **Note:** The final script creates checkpoints during training, but a
> `--resume` command for this standalone training script has not been verified.
> The `train.py --resume` workflow described above is separate.

### Final evaluation and limitations

We evaluated **20 final-run checkpoints** at 25k intervals, each with 10 Grid
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
hairpin around 36% environment progress, and final training did not establish a
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
Stage 1 evaluation numbers above come from earlier tests, not from the final
20-checkpoint batch.
