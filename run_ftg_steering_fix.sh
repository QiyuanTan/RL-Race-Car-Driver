#!/bin/sh
# Run from the RL-Race-Car-Driver project root.
set -eu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m pytest -q tests/test_ftg_steering_loss.py tests/test_ftg_motor_loss.py tests/test_ftg_behavior_cloning.py

# One small supervised experiment; no RL and no overwriting prior ZIPs.
python train_ftg_bc.py \
  --features geometry \
  --bc-steps 2500 --bc-batch-size 64 \
  --motor-loss-weight 1.5 --motor-balance-power 0.65 \
  --steering-loss-weight 2.0 --steering-balance-power 0.6 \
  --start-boost-fraction 0.20 --start-prefix 80 \
  --rl-timesteps 0 --seed 0 \
  --output-prefix models/jx_dqn_ftg_steeringaware_2500

python evaluate_ftg_bc.py \
  --model models/jx_dqn_ftg_steeringaware_2500_bc_only.zip \
  --features geometry --episodes 3 --seed 100 --action-repeat 4 \
  --max-sim-time 15 --out /tmp/jx_ftg_steeringaware_2500_eval

python diagnose_ftg_bc_actions.py --models \
  models/jx_dqn_ftg_steeringaware_2500_bc_only.zip
