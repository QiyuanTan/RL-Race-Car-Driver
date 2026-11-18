# RL-Race-Car-Driver

## Setup

Have virtual env setup with Python <= 3.9. racecar_gym was last updated in 3 years ago so you might encounter 
compatibility issues with later Python versions.

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

Evaluate with the same episode loop as the FTG baseline (`evaluate_baseline.run_episode`);
seeds 100-109 match the FTG test seeds:

```bash
python evaluate_sb3.py --model models/dqn_model.zip --episodes 10 --seed 100 --workers 4
python evaluate_sb3.py --model models/dqn_lidar_model.zip --lidar --episodes 10 --seed 100 --workers 4
```

The DQN scenarios end an episode at the first wall contact, so `collisions` is 0 or 1 and
`progress` shows how far the car got.

To view training metrics, run:

```bash
tensorboard --logdir logs
```

For inference, `run_sim.py` provides a simple interface to load a trained model and run it in the simulator.

