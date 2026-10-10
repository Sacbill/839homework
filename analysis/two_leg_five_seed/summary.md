# Five-seed final evaluation

Each entry is the mean of 10 deterministic evaluation episodes. The
reported 95% t interval uses the five independent training seeds as the
sample (df = 4). PPO used 899,072 steps and SAC used 300,000 steps.

| Algorithm | Return | Two-leg time (s / 50 s) | Two-leg fraction | Longest hold (s) | Switches |
|---|---:|---:|---:|---:|---:|
| PPO | 6011.03 $\pm$ 385.86 | 36.11 $\pm$ 3.58 | 0.722 $\pm$ 0.072 | 0.969 $\pm$ 0.539 | 0.00 $\pm$ 0.00 |
| SAC | 6777.35 $\pm$ 571.49 | 40.66 $\pm$ 3.09 | 0.813 $\pm$ 0.062 | 1.111 $\pm$ 0.287 | 0.00 $\pm$ 0.00 |

Paired SAC minus PPO differences:

- episodic_return: 766.3229 $\pm$ 688.8517 (95% CI)
- two_leg_seconds: 4.5480 $\pm$ 5.8919 (95% CI)
- two_leg_fraction: 0.0910 $\pm$ 0.1178 (95% CI)

All ten policies completed every 50-second evaluation episode without
a body contact. Every policy had zero valid support-pair switches. The
policies therefore learned long aggregate two-foot contact, but did not
learn the originally proposed alternating-leg behavior.
