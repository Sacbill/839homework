# CS 839 Homework: Sustained Two-Leg MuJoCo Ant

This repository contains only the code needed to train, evaluate, and analyze
the custom `AntTwoLeg-v0` task with PPO and SAC. The Ant is rewarded for staying
upright while exactly two feet touch the ground.

## Files

- `cleanrl/two_leg_ant_env.py`: custom Gymnasium environment and reward
- `cleanrl/ppo_two_leg.py`: PPO trainer
- `cleanrl/sac_two_leg.py`: SAC trainer
- `cleanrl_utils/evals/two_leg_eval.py`: deterministic evaluation and metrics
- `cleanrl_utils/buffers.py`: replay buffer required by SAC
- `analysis/two_leg_five_seed/analyze_runs.py`: five-seed plots and summaries
- `tests/`: environment and evaluation tests

## Installation

Python 3.10 was used for the experiments.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Verify the installation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest \
  tests.test_two_leg_ant_env tests.test_two_leg_evaluation
```

## Full experiments

Run from the repository root. Each command writes TensorBoard data and models
under `runs/` and videos under `videos/`.

```bash
for seed in 1 2 3 4 5; do
  .venv/bin/python cleanrl/ppo_two_leg.py \
    --env-id AntTwoLeg-v0 \
    --exp-name ppo_sustained_hold_v4 \
    --seed "$seed" \
    --total-timesteps 900000 \
    --eval-episodes 10 \
    --capture-video \
    --save-model
done

for seed in 1 2 3 4 5; do
  .venv/bin/python cleanrl/sac_two_leg.py \
    --env-id AntTwoLeg-v0 \
    --exp-name sac_sustained_hold_v4 \
    --seed "$seed" \
    --total-timesteps 300000 \
    --learning-starts 5000 \
    --buffer-size 300000 \
    --eval-episodes 10 \
    --capture-video \
    --save-model
done
```

Generate the five-seed plots after all ten runs finish:

```bash
.venv/bin/python analysis/two_leg_five_seed/analyze_runs.py
```

View training logs with:

```bash
.venv/bin/python -m tensorboard.main --logdir runs
```

This implementation is based on the MIT-licensed
[CleanRL](https://github.com/vwxyzjn/cleanrl) project.
