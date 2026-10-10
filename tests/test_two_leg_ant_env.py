import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.ant_v4 import AntEnv

import cleanrl.two_leg_ant_env  # noqa: F401 - registers AntTwoLeg-v0


class TwoLegAntEnvTests(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("AntTwoLeg-v0")
        self.obs, _ = self.env.reset(seed=1)
        self.base = self.env.unwrapped
        self.addCleanup(self.env.close)

    def test_target_pair_is_part_of_observation(self):
        self.assertTrue(self.env.observation_space.contains(self.obs))
        np.testing.assert_array_equal(self.obs[-2:], [1.0, 0.0])

    def test_pairs_are_geometrically_diagonal(self):
        # Use actual model geometry, not the same name list as the environment.
        qpos = self.base.init_qpos.copy()
        qpos[3:7] = [1, 0, 0, 0]
        qpos[7:] = 0
        self.base.set_state(qpos, np.zeros_like(self.base.init_qvel))
        torso_xy = self.base.data.body("torso").xpos[:2]
        for pair in self.base.support_pairs:
            first, second = pair
            a = self.base.data.geom_xpos[first, :2] - torso_xy
            b = self.base.data.geom_xpos[second, :2] - torso_xy
            np.testing.assert_allclose(a, -b, atol=1e-8)

    def test_realistic_height_can_be_a_valid_pose(self):
        self.assertTrue(self.base.is_two_leg_pose(2, 0.45, 0.1, False))
        self.assertFalse(self.base.is_two_leg_pose(2, 0.44, 0.1, False))
        self.assertFalse(self.base.is_two_leg_pose(2, 0.45, 0.1, True))

    def test_stable_diagonal_pairs_advance_the_target(self):
        dwell_steps = self.base.dwell_steps
        first_pair, second_pair = self.base.support_pairs

        for _ in range(dwell_steps):
            self.base.update_task_metrics(first_pair, True)

        self.assertEqual(self.base.target_pair_index, 1)
        self.assertEqual(self.base.support_switches, 0)

        for _ in range(dwell_steps):
            self.base.update_task_metrics(second_pair, True)

        self.assertEqual(self.base.target_pair_index, 0)
        self.assertEqual(self.base.support_switches, 1)

    def test_long_failed_transfer_does_not_count_as_a_switch(self):
        dwell_steps = self.base.dwell_steps
        timeout_steps = int(np.ceil(self.base.max_transition_seconds / self.base.dt)) + dwell_steps + 1
        first_pair, second_pair = self.base.support_pairs

        for _ in range(dwell_steps):
            self.base.update_task_metrics(first_pair, True)
        for _ in range(timeout_steps):
            self.base.update_task_metrics(frozenset(), False)
        for _ in range(dwell_steps):
            self.base.update_task_metrics(second_pair, True)

        self.assertEqual(self.base.support_switches, 0)

    def test_partial_target_contacts_do_not_extend_timeout(self):
        first, second = self.base.support_pairs
        for _ in range(self.base.dwell_steps):
            self.base.update_task_metrics(first, True)
        for _ in range(20):
            self.base.update_task_metrics(second, True)
            self.base.update_task_metrics(frozenset(), False)
        for _ in range(self.base.dwell_steps):
            self.base.update_task_metrics(second, True)
        self.assertEqual(self.base.target_holds, 2)
        self.assertEqual(self.base.support_switches, 0)

    def test_body_contact_breaks_switch_sequence(self):
        first, second = self.base.support_pairs
        for _ in range(self.base.dwell_steps):
            self.base.update_task_metrics(first, True)
        self.base.update_task_metrics(frozenset(), False, failed=True)
        for _ in range(self.base.dwell_steps):
            self.base.update_task_metrics(second, True)
        self.assertEqual(self.base.support_switches, 0)

    def test_step_pays_for_completion_and_returns_updated_target(self):
        qpos = self.base.init_qpos.copy()
        qpos[2] = 0.5
        qpos[3:7] = [1, 0, 0, 0]
        self.base.set_state(qpos, np.zeros_like(self.base.init_qvel))
        first = self.base.support_pairs[0]
        with patch.object(AntEnv, "step", side_effect=lambda action: (None, 0, False, False, {})), \
             patch.object(self.base, "ground_contacts", return_value=(first, False)):
            previous_progress_reward = 0.0
            for step in range(self.base.dwell_steps):
                obs, reward, terminated, _, info = self.base.step(np.zeros(8))
                self.assertTrue(np.isfinite(reward))
                self.assertFalse(terminated)
                self.assertEqual(info["reward_two_leg"], 4.0)
                self.assertGreater(info["reward_hold_progress"], previous_progress_reward)
                previous_progress_reward = info["reward_hold_progress"]
                if step < self.base.dwell_steps - 1:
                    self.assertEqual(info["reward_target_pair"], 0)
                    self.assertAlmostEqual(obs[-3], (step + 1) / self.base.dwell_steps)
                else:
                    self.assertEqual(info["reward_target_pair"], 2.0)
                    np.testing.assert_array_equal(obs[-3:], [0, 0, 1])
                self.assertTrue(self.env.observation_space.contains(obs))

    def test_interrupted_hold_resets_progress(self):
        first = self.base.support_pairs[0]
        self.base.update_task_metrics(first, True)
        self.assertEqual(self.base.target_pair_steps, 1)
        self.base.update_task_metrics(frozenset(), False)
        self.assertEqual(self.base.target_pair_steps, 0)

    def test_reset_clears_history(self):
        for _ in range(self.base.dwell_steps):
            self.base.update_task_metrics(self.base.support_pairs[0], True)
        obs, _ = self.env.reset(seed=1)
        self.assertEqual(self.base.target_holds, 0)
        self.assertEqual(self.base.two_leg_steps, 0)
        self.assertIsNone(self.base.last_completed_pair)
        np.testing.assert_array_equal(obs[-3:], [0, 1, 0])

    def test_target_guidance_prefers_requested_contacts(self):
        target, other = self.base.support_pairs
        self.assertGreater(self.base.target_contact_score(target), self.base.target_contact_score(other))
        self.assertEqual(self.base.target_contact_score(target | other), 0)
        self.assertGreater(self.base.target_contact_score(frozenset([next(iter(target))])), 0)


if __name__ == "__main__":
    unittest.main()
