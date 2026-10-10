import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_two_leg import Agent, make_env
from cleanrl.sac_two_leg import Actor
from cleanrl_utils.evals.two_leg_eval import (
    evaluate_policy, load_observation_stats, normalize_observation, save_observation_stats,
)


class EvaluationTests(unittest.TestCase):
    def test_saved_normalization_matches_training_and_is_frozen(self):
        env = make_env("AntTwoLeg-v0", 0, False, "test", 0.99)()
        self.addCleanup(env.close)
        expected, _ = env.reset(seed=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.npz"
            save_observation_stats(env, path)
            stats = load_observation_stats(path)
        raw = env.unwrapped._get_obs()
        np.testing.assert_allclose(normalize_observation(raw, stats), expected, atol=1e-6)
        before = {key: value.copy() for key, value in stats.items()}
        normalize_observation(raw + 10, stats)
        for key in stats:
            np.testing.assert_array_equal(before[key], stats[key])

    def test_deterministic_seeded_evaluation_and_weight_reload(self):
        # Short TimeLimits exercise truncation without making tests a training run.
        env_id = "AntTwoLegEvalTest-v0"
        if env_id not in gym.registry:
            gym.register(env_id, entry_point="cleanrl.two_leg_ant_env:TwoLegAntEnv", max_episode_steps=12)
        spaces = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)])
        self.addCleanup(spaces.close)
        torch.manual_seed(1)
        for algorithm, model_type in [("ppo", Agent), ("sac", Actor)]:
            with self.subTest(algorithm=algorithm):
                model = model_type(spaces)
                stats = None
                if algorithm == "ppo":
                    stats = {"mean": np.zeros(spaces.single_observation_space.shape),
                             "var": np.ones(spaces.single_observation_space.shape), "epsilon": 1e-8}
                first = evaluate_policy(model, algorithm, env_id, 10001, 2, observation_stats=stats)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt"
                    torch.save(model.state_dict(), path)
                    restored = model_type(spaces)
                    restored.load_state_dict(torch.load(path, weights_only=True))
                second = evaluate_policy(restored, algorithm, env_id, 10001, 2, observation_stats=stats)
                self.assertEqual(first, second)
                self.assertEqual([row["seed"] for row in first], [10001, 10002])
                self.assertTrue(all(row["truncated"] for row in first))


if __name__ == "__main__":
    unittest.main()
