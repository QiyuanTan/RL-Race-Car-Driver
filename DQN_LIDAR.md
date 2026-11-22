# DQN + LiDAR

I added LiDAR to Qiyuan's DQN branch to see if knowing how far the walls are helps the car drive farther. The baseline uses the RGB camera and car state. With `--lidar`, it also gets 1080 distance readings scaled to [0, 1]. Both use the same 15 actions, get rewarded for progress, and stop at the first wall hit.

I also added action repeat, checkpoints, and a few ways to fine-tune around the first hairpin. Action repeat holds each action for several simulator steps. The checkpoint options let me evaluate models during training or resume a run with its replay buffer.

## Results on Austria

I picked checkpoints on seeds 0–9 and tested on 100–109, the same test seeds as FTG. All runs were on an M1 Mac. Each setup used one training seed.

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

LiDAR helped with 4-step decisions. Average progress went from 21.9% to 35.3%, and the results varied less across seeds. With 1-step decisions, the LiDAR run did worse than base DQN. The 4-step LiDAR run was the best one, but it still crashed at the first hairpin. Random starts and the first FTG demos didn't get it any farther.

## What I tried at the hairpin

I first wanted to check if the car could make the turn with DQN's actions. I gave FTG the same 15 actions and let it pick one every 4 simulator steps. It passed the first hairpin in 8/10 runs from the grid and 16/20 runs starting near the turn. Only 3/10 of the grid runs finished a lap because most of the crashes were at the second hairpin. That was enough to try teaching DQN the turn with the same actions.

I took the 140k LiDAR checkpoint and trained it for another 100k decisions. One run kept the grid start so I could see what extra training alone did. The second started half its training episodes before the hairpin. The third used the same starts and subtracted 10 from the reward for a crash. All three reached 35.4% on test and finished 0/10 laps. The crash penalty made some validation runs last longer, but they timed out without getting past the turn.

### How they enter the turn

I used `tools/hairpin_entry_compare.py` to compare DQN and FTG from the grid with the same actions. DQN approached at about 3.3 m/s, close to the inside wall, and crashed in all 10 runs. FTG slowed to 2 m/s or less, stayed closer to the middle, and started holding a right turn around progress 0.32. It passed all 10 runs.

That made me wonder if FTG could still get through when DQN had already set up the turn badly. With `tools/takeover_check.py`, I let DQN drive first and switched to FTG before the turn. Using DQN's actions, FTG passed 9/10 at each takeover point from 0.24 to 0.28. At 0.30 it passed 5/10. That gave me a way to collect demos from the approach DQN actually takes.

### Takeover demos

I kept 25 successful takeovers from 34 attempts. In both runs, 20% of each training batch came from those demos. In the second run I also added a DQfD margin loss to push the demo action above the others and did 5000 updates on demos before the car started driving.

Neither run passed the hairpin on validation. The demos made the car slower and less consistent. With margin loss it often stopped and timed out, and the best validation checkpoint reached only 17.7% on test. My guess is that the demos covered only the approach and the turn, so the model picked up the slowing down part elsewhere on the track too.

## Running it

Base and LiDAR training:

```bash
python train.py --action-repeat 4 --target-update-interval 2500 --timesteps 200000 --seed 0 --save-path models/dqn_ar4_long --eval-every 10000 --resume-every 50000
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 200000 --seed 0 --save-path models/dqn_lidar_ar4_long --eval-every 10000 --resume-every 50000
```

Eval picks up the saved checkpoint. Match the LiDAR and action repeat settings from training.

```bash
python evaluate_sb3.py --model models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --lidar --action-repeat 4 --episodes 10 --seed 100 --workers 4 --out results/test_ar4
```

Fine-tuning with hairpin starts, grid starts, and a crash penalty:

```bash
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_hairpin --train-start hairpin --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_grid --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 100000 --seed 0 --save-path models/dqn_lidar_ar4_ft_hairpin_cp10 --train-start hairpin --crash-penalty 10 --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --eval-every 10000 --resume-every 50000
```

Takeover demos, with and without the margin loss:

```bash
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 50000 --seed 0 --save-path models/dqn_lidar_ar4_ft_takeover --train-start hairpin --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --takeover-demos 25 --demo-fraction 0.2 --eval-every 10000 --resume-every 25000
python train.py --lidar --action-repeat 4 --target-update-interval 2500 --timesteps 50000 --seed 0 --save-path models/dqn_lidar_ar4_ft_margin --train-start hairpin --init-from models/checkpoints/dqn_lidar_ar4_long/eval/dqn_lidar_ar4_long_140000_steps.zip --takeover-demos 25 --demo-fraction 0.2 --margin-loss 1.0 --demo-pretrain-steps 5000 --eval-every 10000 --resume-every 25000
```
