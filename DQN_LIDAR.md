# DQN + LiDAR

I added LiDAR observations to Qiyuan's DQN branch. The original setup uses SB3 DQN with 15 discrete actions (3 motor levels × 5 steering levels), a progress reward, and episodes that end when the car touches a wall.

The base DQN gets the RGB camera image plus pose, velocity, acceleration, and time. Its scenario file, `scenarios/austria.yml`, doesn't include LiDAR. For this version, I added `scenarios/austria_lidar.yml` with the same settings and a `lidar` sensor. `LidarObservationWrapper` normalizes the 1080 distance readings to [0, 1].

The actions, reward, and termination conditions are unchanged, so the base and LiDAR versions only differ in their observations.

## Running it

These commands train the base and LiDAR versions with the same budget and seed:

```bash
python train.py --action-repeat 4 --target-update-interval 2500 --timesteps 200000 --seed 0 --save-path models/dqn_ar4_long --eval-every 10000 --resume-every 50000
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 200000 --seed 0 --save-path models/dqn_lidar_ar4_long --eval-every 10000 --resume-every 50000
```

Running `python train.py` without arguments still uses the original settings: base DQN, 1M steps.

A few options I added:

- `--action-repeat N`: repeats each action for N simulation steps. SB3 counts each decision as one step, so the training budget, replay buffer, and target update interval are measured in decisions. For these experiments, I set `--target-update-interval` to cover 10k simulation steps: 2500 for N=4 and 5000 for N=2.
- `--eval-every N`: saves a model every N decisions under `models/checkpoints/<name>/eval/`.
- `--resume-every N`: saves the model and replay buffer so training can continue if interrupted. It keeps only the latest copy, which takes about 1–2 GB. To resume, use the same training command with `--resume` added.
- `--train-start random`: starts training episodes at random positions on the track. Evaluation still starts on the grid.
- `--demo-episodes K` / `--demo-laps K`: adds FTG demonstrations to the replay buffer before training (LiDAR only). FTG uses the same discrete actions and action repeat as DQN. The demos stay in the buffer throughout training. `--demo-laps` only keeps demonstrations that complete a lap.

Each run saves its settings to `models/checkpoints/<name>/config.json`, including the demo counts when demos are used.

For evaluation, I used the same episode loop as the FTG baseline, `evaluate_baseline.run_episode`:

```bash
python evaluate_sb3.py --model models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --lidar --action-repeat 4 --episodes 10 --seed 100 --workers 4 --out results/test_ar4
```

The `--lidar` and `--action-repeat` settings need to match training. The script checks `config.json` and stops if they don't match. Each episode runs in a separate process because `racecar_gym` doesn't fully reset between episodes.

Checkpoint performance varied a lot: one could reach 35% progress and the next only 2%. I used validation seeds 0–9 to choose checkpoints, then evaluated the selected models on test seeds 100–109, the same test seeds used for FTG. The two 1-step runs were trained before I started picking on validation, so I went back afterward and picked their checkpoints the same way (they only saved one every 20k decisions).

To check the results:

- `python tools/dqn_diagnose.py evals <results folder>` prints the evaluation table.
- `python tools/dqn_diagnose.py curves` prints training curves from the TensorBoard logs.

Tests can be run with `python -m pytest tests/` (28 tests, no simulator needed).

## Results

These results are from the Austria track on an M1 Mac, using test seeds 100–109. Episodes end at the first wall contact. Progress is the fraction of a lap completed.

| Setup | Decision interval | Budget (decisions / sim steps) | Checkpoint | Test progress (min–max) | Laps |
|---|---|---|---|---|---|
| Base | 1 step | 200k / 200k | 20k | 12.6% (12.2–12.8) | 0/10 |
| LiDAR | 1 step | 200k / 200k | 200k | 8.7% (7.4–9.4) | 0/10 |
| Base | 4 steps | 200k / 800k | 200k | 21.9% (9.2–35.5) | 0/10 |
| **LiDAR** | **4 steps** | **200k / 800k** | **140k** | **35.3% (34.9–35.4)** | **0/10** |
| LiDAR + random start | 4 steps | 200k / 800k | 120k | 35.0% (34.2–35.5) | 0/10 |
| LiDAR + 2 FTG demo episodes | 4 steps | 200k / 800k | 120k | 35.3% (35.2–35.4) | 0/10 |
| LiDAR | 2 steps | 200k / 400k | 130k | 12.3% (11.8–12.8) | 0/10 |
| LiDAR + 2 FTG demo laps | 2 steps | 200k / 400k | 150k | 16.0% (12.5–29.2) | 0/10 |
| FTG baseline (reference) | — | — | — | 10/10 laps, 81.77 s | 10/10 |

For the demo run with 4-step decisions, FTG completed 0/2 demo laps on the Mac. Those two episodes provided 1746 decisions / 6985 simulation steps. Each run's demo counts are saved in its `config.json`.

## What I found

- **LiDAR helped with 4-step decisions.** With 4-step decisions, average test progress went from 21.9% to 35.3%. The LiDAR results also varied much less across test seeds.
- **Action repeat made a big difference to training.** Without it, training reward was highest during random exploration and dropped afterward. Repeating each action for 4 steps brought the decision rate to 25 Hz, matching the LiDAR update rate. This was the setting where the training curve started improving.
- **The best DQN runs still stopped around 35% progress**, near the Austria hairpin. Plain FTG also scraped the wall there, and an uploaded PPO model stopped at 35.5% in a quick one-episode test. The car's minimum turning radius is about 1.5 m, so it needs to start turning early.
- **I checked whether FTG could complete a lap using DQN's actions.** With `tools/ftg_action_check.py` on the Mac, using seeds 1000–1009, FTG finished 3/10 laps with 4-step decisions, 6/10 with 2-step decisions, and 10/10 with 1-step decisions. Adding finer steering levels didn't help much. This suggests the decision interval is part of the problem. But DQN learned less with 2-step decisions than with 4-step decisions, so I haven't found a setting that both learns well and handles the hairpin reliably.
- **Random starts and FTG demos didn't improve the best result within this budget.**

## Fine-tuning the hairpin

The best LiDAR checkpoint (AR4, 140k) reaches the first hairpin apex (progress 0.351) every time and crashes there.

In a separate run, we checked where FTG crashed using `tools/ftg_hairpin_check.py` and `tools/ftg_action_check.py`.

| FTG restricted to DQN's 15 actions, deciding every 4 steps | passed 1st hairpin | full laps |
|---|---|---|
| starting from the grid (seeds 1000-1009) | 10/10 | 4/10 (the other 6 crashed at the 2nd hairpin, progress ~0.53) |
| starting 4.5-11 m before the hairpin (seeds 1000-1019) | 18/20 | - |

The full-lap count makes the first hairpin look harder than it is. FTG got through the first one from the grid with the same actions and timing as AR4, then lost 6 runs at the second hairpin. That gave us a reason to try more practice on the first one.

We started from LiDAR AR4 at 140k and trained for 100k more decisions on the M1 Mac. One run started on the grid as usual. The other used `--train-start hairpin`, so half the episodes (set by `--hairpin-share`) started 4.5-11 m before the turn (progress 0.22-0.30) and the rest started on the grid. Closer random starts point 40-110 degrees away from the track direction. Eval still runs the whole track from the grid.

`--init-from <model.zip>` loads the trained model. Match `--lidar` and `--action-repeat`. The replay buffer starts empty and exploration stays at 5% from the start so the car mostly uses what it already learned. We used the same validation seeds 0–9 and tested each selected checkpoint once on 100–109.

| run | checkpoint | test progress | laps | notes |
|---|---|---|---|---|
| LiDAR AR4 (start point) | 140k | 35.3% | 0/10 | crashes at the apex after ~11 s |
| + hairpin starts | 80k | 35.4% | 0/10 | slows down before the hairpin (3 episodes lasted 40+ s), still crashes |
| + grid starts (control) | 90k | 35.4% | 0/10 | crashes at the apex after ~10 s |

Both fine-tuned models passed the hairpin in 0/10 test episodes. Even on validation, the farthest either got was 0.355. One guess is that the reward makes it easy to learn the wrong line. Driving straight at the apex makes progress right up to the wall (test runs ended at 0.354-0.355), and crashing costs nothing. Going wide and turning in early barely gains any progress at first.

```bash
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_hairpin --train-start hairpin --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_grid --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
```

We tried the same hairpin run with `--crash-penalty 10`. The penalty only applies to training and comes off the reward when a wall hit ends the episode. A full lap is worth 100.

```bash
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_hairpin_cp10 --train-start hairpin --crash-penalty 10 --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
```

On the M1 Mac, the best validation checkpoint was 70k. It got 35.4% progress, 0/10 hairpin passes, and 0/10 laps on test seeds 100–109. All 10 runs hit the wall at the apex after ~10.6 s. Across all checkpoints, no validation episode went past 0.355.

At 80k-100k, 2-5 of the 10 validation episodes ended without a crash. Average episode time went from ~11 s to 60-74 s. With zero laps, those crash-free runs appear to have hit the 120 s time limit by stopping or creeping. At 5% exploration, our guess is that the model rarely tries the full sequence of going wide and turning in early, so it doesn't get to see it pay off.

I'd try FTG demos starting right before the hairpin next. The notes for that idea give 16/20 passes (~80%) with DQN's actions, and the table above records 18/20. Both counts are in the original notes. I'd report the demo run separately from pure RL and include the demo counts. If we get through the first hairpin, the next one is around progress 0.53.

## Notes and limitations

- I only trained with one seed (0) per setup, so I wouldn't draw conclusions from differences of a few percentage points.
- The main results and fine-tuning runs are from the M1 Mac with conda-forge PyBullet. Compare runs from the same machine.
- Smoothness metrics are calculated per simulation step and should only be compared between runs with the same action repeat.
- Runs with FTG demos should be reported separately from pure RL runs, along with their demo counts.
- The `time` observation exceeds its declared [0, 1] range because `racecar_gym` returns time in seconds. I left this unchanged for all runs.
- Training took about 1.5–2 hours per 200k decisions on the Mac. One evaluation hung after a worker silently exited during a PyBullet LiDAR raycast. It worked again after rebooting.
