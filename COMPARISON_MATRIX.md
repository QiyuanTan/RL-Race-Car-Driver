# Driver interface and evaluation comparison

This document maps the drivers to a shared evaluation interface and records the repository's existing Austria results. It is a starting point for a common comparison, not a new experiment.

## Common evaluation interface

The evaluator resets the driver at the beginning of an episode and calls `act(obs)` for each simulator step. An evaluation-ready driver should implement:

```python
driver.reset()
action = driver.act(observation)  # {'motor': array([value]), 'steering': array([value])}
```

The action values must use the environment's expected bounds, shapes, and steering convention. `evaluate_baseline.run_episode` consumes this interface and returns episode metrics. `SB3EvalDriver` adapts a DQN model to it, including holding an action across repeated simulator steps. A raw `SB3Driver.act()` returns the model's action representation, so continuous PPO/SAC models need an appropriate adapter and matching observation/action wrappers before use with this evaluator.

| Implementation | Driver / integration | Observations | Action representation | Status |
| --- | --- | --- | --- | --- |
| Follow-the-Gap (FTG) | `FollowTheGapDriver` implements `reset()` and `act(obs)` | LiDAR and velocity | Motor and steering commands | Baseline evaluated |
| Base DQN | `SB3Driver` with `SB3EvalDriver` adapter | Camera and vehicle state | 15 discrete motor/steering combinations, decoded to commands | Evaluated |
| LiDAR DQN | Same DQN adapter, with LiDAR observation wrapper | Camera, vehicle state and normalized LiDAR | Same 15 combinations | Evaluated |
| PPO | `SB3Driver`; evaluator adapter needed | Scenario-dependent | Continuous motor and steering via flattened action wrapper | Models present; no comparable evaluation established here |
| SAC | `SB3Driver`; evaluator adapter needed | Scenario-dependent | Continuous motor and steering via flattened action wrapper | Training support exists; no comparable evaluation established here |

## Recorded results

Austria track, grid starts, test seeds 100–109, ten episodes per configuration, as reported in `BASELINE.md` and `DQN_LIDAR.md`. The two DQN rows use their documented validation-selected checkpoints. These are existing results, not measurements made for this document.

| Metric | Better | FTG default | Base DQN, repeat 4 | LiDAR DQN, repeat 4 |
| --- | --- | ---: | ---: | ---: |
| Laps completed | Higher | 10/10 | 0/10 | 0/10 |
| Collision-free laps | Higher | 10/10 | 0/10 | 0/10 |
| Mean progress (fraction of a lap) | Higher | Not summarized | 0.219 | 0.353 |
| Mean lap time, completed laps (s) | Lower | 81.77 ± 0.77 | Not available | Not available |
| Mean wall contacts per episode | Lower | 0.0 | 0.8 | 1.0 |
| Mean motor command change per simulator step | Lower | 0.045106 | 0.010489 | 0.026083 |
| Mean steering command change per simulator step | Lower | 0.040999 | 0.011823 | 0.068175 |

Sources: `results/baseline_ftg/ftg_20261102_190856.json`, `results/test_ar4/dqn_ar4_20261117_060209.json`, and `results/test_ar4/dqn_lidar_ar4_20261116_200843.json`. A dash or unavailable result should never be treated as zero.

## Interpreting the comparison

- The FTG scenario continues after wall contact, while the DQN scenarios terminate on the first contact. Collision counts and episode duration therefore do not have identical conditions. The base DQN result has 0.8 mean contacts; consult individual episodes rather than assuming every episode ended on contact.
- DQN repeats each action for four simulator steps, while FTG chooses an action every step. Per-step smoothness is affected by action repeat and should not be ranked across these settings. Episode lengths and completed track portions also differ.
- Lap time is averaged only over completed laps; progress provides a useful measure when no lap is completed. Progress and lap completion are distinct fields.
- The reports describe runs on one M1 Mac, and only one training seed per DQN setup. Checkpoint selection used validation seeds 0–9. Treat the results as preliminary and compare methods on the same machine and settings.

## Plan for a common comparison

1. Add evaluation adapters for any driver that cannot yet return the common action dictionary, preserving its training-time preprocessing and action decoding.
2. Agree on a shared scenario, track, start mode, episode limit, collision termination rule, seed list, and machine. Record action repeat and report its effect on smoothness.
3. Run each driver under those conditions. Record model/checkpoint, algorithm, observation configuration, training seed and budget, evaluation seeds, simulator version and run settings.
4. Save per-episode results plus summaries: completion rate, collision-free completion rate, progress, lap time for completed laps, contacts, and motor/steering command changes. Clearly mark unavailable results.
5. Keep the current historical results separate from any new aligned runs. Do not infer PPO or SAC performance until evaluated.

## Newer implementations and shared runner

The `jx_dqn_improvement` branch extends DQN with `track` and `track-lookahead` observations and behavior-cloned DQN models, including a geometry-only option. `evaluate_shared.py` connects these variants, FTG, PPO and SAC to the existing episode metrics loop. For commands and limitations, see the shared evaluation section in `README.md`. The historical results above do not include these newer variants.
