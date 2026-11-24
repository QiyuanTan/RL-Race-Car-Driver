# RL-Race-Car-Driver

We built and tested race car controllers on the Austria track in [racecar_gym](https://github.com/axelbr/racecar_gym). The project includes a rule-based Follow-the-Gap baseline, DQN with several observation modes, and a PPO implementation. The analysis and discussion of all results are in our report; this README explains how to run the code.

## Branches

We submit the `final` branch. The table shows where each part of the project was developed.

| Branch | Adds | Built on |
| --- | --- | --- |
| `main`, `SAC` | Project setup: `SB3Driver`, observation and action wrappers, PPO training helper, `run_sim.py` | - |
| `DQN` | Base DQN with 15 discrete motor/steering actions on camera and vehicle-state observations | `main` |
| `baseline-Elaine` | Follow-the-Gap driver, `evaluate_baseline.py`, unit tests | `SAC` |
| `jx_smoothness_metrics` | Added motor and steering smoothness metrics to measure changes in driving actions | `baseline-Elaine` |
| `dqn-lidar` | DQN + LiDAR, action repeat, eval / resume checkpoints, `evaluate_sb3.py`, random starts, FTG demos | `jx_smoothness_metrics`, `DQN` |
| `dqn-track` | Track-relative observations (`--obs-mode track`, `track-lookahead`); later also the track-lookahead Stage 1 model and mixed-start fine-tuning | `dqn-lidar` |
| `dqn-lidar-hairpin` | Hairpin fine-tuning options and analysis tools | `dqn-lidar` |
| `jx_dqn_improvement` | Added signed progress reward, FTG demonstrations for track-lookahead DQN, Behavior Cloning with motor and steering losses, and BC-regularized DQN training | `dqn-track` |
| `comparison-matrix` | Performance comparison matrix, unified evaluation metrics, lap completion rate, lap time, collision rate, and control smoothness comparison | `dqn-lidar` |
| `comparison-interface-track` | Comparison interface for track-relative DQN agents, integration with the evaluation framework, and standardized performance comparison across agents | `jx_dqn_improvement` |
| `birdview_features` | Bird's-eye track view and drivable-area observations for PPO and DQN (not included in the report) | `dqn-lidar` |
| `final` | `comparison-interface-track` with `dqn-lidar-hairpin` merged in, plus the later bird's-eye view, BC models and track-lookahead fine-tuning | all of the above |

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

To select another algorithm, change the call inside the `if __name__ == '__main__':`
block in `train.py`, creating a fresh environment for each training run:

```python
env = gym_api.SingleAgentRaceEnv(
    scenario=scenario, render_mode='rgb_array_birds_eye'
)
train_ppo(env, total_timesteps=1_000_000, save_path='models/ppo_model')
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

The signed progress reward (`--reward-mode signed-progress`), FTG Behavior Cloning, and BC-regularized DQN for track-lookahead DQN are trained with `train_ftg_bc.py` and `train_ftg_bc_rl.py` and evaluated with `evaluate_ftg_bc.py`.

### DQN hairpin fine-tuning

These options fine-tune a trained DQN on the first hairpin:

- `--init-from <model.zip>` loads a trained model and starts with 5% exploration. Match the observation mode and action repeat used in training.
- `--train-start hairpin` starts some training episodes just before the turn. `--hairpin-share` defaults to 0.5.
- `--crash-penalty P` subtracts P from the reward on a wall hit that ends the episode.
- `--takeover-demos K` collects K successful FTG takeovers from the trained LiDAR model. It uses `--takeover-points` and samples a share of each batch from the demos, set by `--demo-fraction`.
- `--margin-loss W` adds a DQfD loss on demo samples. `--demo-pretrain-steps N` runs N updates on demos before regular training.

The settings are saved in `config.json`. Evaluation still starts from the grid.

```bash
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_hairpin --train-start hairpin --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
```

See [DQN_LIDAR.md](DQN_LIDAR.md) for the commands of every run and the analysis tools.

To view training metrics, run:

```bash
tensorboard --logdir logs
```

For inference, `run_sim.py` provides a simple interface to load a trained model and run it in the simulator.


## Track-Lookahead DQN: mixed-start fine-tuning (Yutao Lin)

This experiment fine-tunes a track-lookahead DQN (Stage 1, 300k decisions) with
reward shaping, stall detection, and mixed starting positions (40% grid, 35% near
the hairpin, 25% random). Both models use the 15 discrete actions and
`action_repeat=4`.

| Model | Purpose |
| --- | --- |
| `models/dqn_centerline_stage1_300k.zip` | Stage 1 model, trained for 300k decisions |
| `models/dqn_final_best_75k.zip` | Selected checkpoint from the 500k-decision fine-tuning run (75k additional decisions) |

The fine-tuning script is `train_dqn_final_stall_v2_fixed.py` (learning rate `3e-5`,
replay buffer `20,000`, batch size `64`, target update interval `5,000`, exploration
from `0.15` to `0.03` over the first 35% of training). To reproduce the training
configuration from the Stage 1 model (PowerShell line continuations):

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

`evaluate_final_checkpoints.py` evaluates all saved checkpoints of this run on the
same grid seeds; the summary is in
`results/final_checkpoint_grid_eval/checkpoint_comparison.csv`.

Known limitations of this part:

- The script reads the Stage 1 `config.json` from
  `models/checkpoints/dqn_centerline_stage1_300k/`, which is not in the repository.
- `--resume` for this standalone script has not been verified (the `train.py --resume`
  workflow is separate).
- Both models were saved with NumPy 1.26.4. With the NumPy 1.22.3 pinned in
  `requirements.txt`, Stable-Baselines3 cannot load their saved observation space,
  so they need an environment with NumPy ≥ 1.25.

## Shared evaluation interface

`evaluate_shared.py` evaluates FTG, DQN (including `base`, `lidar`, `track`, and `track-lookahead` modes), and PPO with the existing `evaluate_baseline.run_episode` metrics. For the behavior-cloning geometry-only DQN, use `--obs-mode track-lookahead --geometry-only`. Models must be evaluated with their training-time observation mode, action repeat and feature settings. Saved-model observation and action spaces are checked before evaluation; geometry-only models have no complete configuration check, so the user must supply matching feature settings.

Examples (install the project dependencies and racecar_gym first):

```bash
python evaluate_shared.py --algorithm ftg --episodes 10 --seed 100
python evaluate_shared.py --algorithm dqn --model path/to/dqn.zip --obs-mode lidar --action-repeat 4
python evaluate_shared.py --algorithm dqn --model path/to/student.zip --obs-mode track-lookahead --geometry-only --action-repeat 4
python evaluate_shared.py --algorithm ppo --model models/ppo_model_w_collision.zip
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

Austria track, grid starts. FTG and the first DQN rows use test seeds 100–109,
with DQN checkpoints picked on validation seeds 0–9. The track-lookahead
fine-tuning checkpoint was selected on seeds 100–109, so its number is a
selection result, not a held-out test. Track-lookahead rows report mean maximum
progress. See [COMPARISON_MATRIX.md](COMPARISON_MATRIX.md) for smoothness and
collision metrics, and the report for the discussion.

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
| Track-lookahead DQN — Stage 1 | 4 steps | 300k | ~35.1% | 0/10 | 0/10 |
| Track-lookahead DQN — mixed-start fine-tuning | 4 steps | 75k additional | 35.45% (35.3–35.5) | 0/10 | 0/10 |

