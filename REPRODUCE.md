# Reproducing the CS 839 Two-Leg Ant Results

The submission files under `code/` are laid out relative to the CleanRL
repository root. Install CleanRL and its MuJoCo dependencies first:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements/requirements.txt
.venv/bin/pip install -r requirements/requirements-mujoco.txt
```

Run five paired seeds of PPO and SAC:

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

Regenerate the plots and tables after placing `analyze_runs.py` at
`analysis/two_leg_five_seed/analyze_runs.py`:

```bash
.venv/bin/python analysis/two_leg_five_seed/analyze_runs.py
```

The analysis applies a 20-episode rolling mean to every seed separately,
interpolates onto a 5,000-step common grid, and plots the mean with a 95%
Student-t interval using multiplier 2.776 for five training seeds.

The submitted best policy video is the deterministic evaluation of SAC seed 3.
