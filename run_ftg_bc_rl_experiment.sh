#!/bin/sh
# Run only from the root of RL-Race-Car-Driver; no files are overwritten.
set -eu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m pytest -q tests/test_ftg_bc_regularized_dqn.py
python train_ftg_bc_rl.py \
  --base-model models/jx_dqn_ftg_steeringaware_2500_bc_only.zip \
  --demo-buffer models/checkpoints/jx_dqn_ftg_track_10k/resume/replay_buffer.pkl \
  --rl-timesteps 2000 --checkpoint-every 500 --learning-rate 0.00001 \
  --bc-weight-start 0.15 --bc-weight-end 0.10 \
  --eval-seeds 3 --eval-start-seed 100 --eval-max-sim-time 15 \
  --output-prefix models/jx_dqn_ftg_bc_regularized_2k
