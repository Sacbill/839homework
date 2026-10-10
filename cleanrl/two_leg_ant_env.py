import numpy as np
from gymnasium.envs.registration import register
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.spaces import Box


class TwoLegAntEnv(AntEnv):
    reward_version = "sustained_two_leg_hold_v4"
    dwell_seconds = 1.0
    max_transition_seconds = 0.5

    def __init__(self, **kwargs):
        self.target_pair_index = 0
        self.target_pair_steps = 0
        super().__init__(
            terminate_when_unhealthy=False,
            use_contact_forces=True,
            **kwargs,
        )

        self.support_pairs = (
            frozenset(
                {
                    self.model.geom("left_ankle_geom").id,
                    self.model.geom("third_ankle_geom").id,
                }
            ),
            frozenset(
                {
                    self.model.geom("right_ankle_geom").id,
                    self.model.geom("fourth_ankle_geom").id,
                }
            ),
        )
        self.foot_ids = tuple(
            self.model.geom(name).id for name in (
                "left_ankle_geom", "right_ankle_geom",
                "third_ankle_geom", "fourth_ankle_geom",
            )
        )
        self.observation_space = Box(
            low=np.concatenate(
                [self.observation_space.low, np.zeros(7)]
            ).astype(np.float32),
            high=np.concatenate(
                [self.observation_space.high, np.ones(7)]
            ).astype(np.float32),
            dtype=np.float32,
        )
        self.reset_task_metrics()

    def _get_obs(self):
        target_pair = np.zeros(2, dtype=np.float32)
        target_pair[self.target_pair_index] = 1.0
        feet, _ = self.ground_contacts()
        contacts = [float(foot in feet) for foot in self.foot_ids]
        progress = self.target_pair_steps / self.dwell_steps
        return np.concatenate(
            [super()._get_obs(), contacts, [progress], target_pair]
        ).astype(np.float32)

    def reset_model(self):
        self.reset_task_metrics()
        return super().reset_model()

    def reset_task_metrics(self):
        """Reset all episode-level metrics for the two-leg task."""
        self.two_leg_steps = 0
        self.current_two_leg_streak = 0
        self.longest_two_leg_streak = 0
        self.support_switches = 0
        self.target_holds = 0
        self.pair_steps = [0, 0]
        self.episode_steps = 0
        self.body_contact_steps = 0
        self.target_pair_index = 0
        self.target_pair_steps = 0
        self.last_completed_pair = None
        self.transition_gap_steps = 0

    @property
    def dwell_steps(self):
        return max(1, int(np.ceil(self.dwell_seconds / self.dt)))

    @property
    def target_pair(self):
        """Return the diagonal pair the policy is currently asked to support."""
        return self.support_pairs[self.target_pair_index]

    def ground_contacts(self):
        """Return the supporting feet and whether another body geom touches the floor."""
        floor_id = self.model.geom("floor").id

        grounded_feet = set()
        body_contact = False

        for index in range(self.data.ncon):
            contact = self.data.contact[index]

            if contact.geom1 == floor_id:
                other_geom = contact.geom2
            elif contact.geom2 == floor_id:
                other_geom = contact.geom1
            else:
                continue

            if other_geom in self.foot_ids:
                grounded_feet.add(other_geom)
            else:
                body_contact = True

        return frozenset(grounded_feet), body_contact

    def is_two_leg_pose(self, grounded_feet, torso_height, torso_up_alignment, body_contact):
        """Define a valid two-leg pose independently from the shaped reward."""
        return (
            grounded_feet == 2
            and torso_height >= 0.45
            and torso_up_alignment > 0.0
            and not body_contact
        )

    def update_task_metrics(self, feet, valid_pose, failed=False):
        """Return whether a target hold completed; count only timely transfers."""
        self.episode_steps += 1
        if failed:
            self.last_completed_pair = None
            self.transition_gap_steps = 0
        if valid_pose:
            self.two_leg_steps += 1
            for index, pair in enumerate(self.support_pairs):
                self.pair_steps[index] += int(feet == pair)
            self.current_two_leg_streak += 1
            self.longest_two_leg_streak = max(
                self.longest_two_leg_streak,
                self.current_two_leg_streak,
            )
        else:
            self.current_two_leg_streak = 0

        # Count the whole transfer, including interrupted attempts on the new
        # pair. One brief target contact must not erase a long failed transfer.
        if self.last_completed_pair is not None:
            self.transition_gap_steps += 1
        on_target_pair = valid_pose and feet == self.target_pair
        if on_target_pair:
            self.target_pair_steps += 1
        else:
            self.target_pair_steps = 0

        # A transfer must finish promptly; otherwise the next acquired pair starts
        # a new sequence rather than counting as a switch.
        max_transition_steps = max(1, int(np.ceil(self.max_transition_seconds / self.dt)))
        if self.transition_gap_steps > max_transition_steps + self.dwell_steps:
            self.last_completed_pair = None
            self.transition_gap_steps = 0

        # Require a full dwell on the target pair so contact flicker cannot
        # count as a completed transfer.
        if self.target_pair_steps == self.dwell_steps:
            self.target_holds += 1
            if self.last_completed_pair is not None:
                self.support_switches += 1
            self.last_completed_pair = feet
            self.target_pair_index = 1 - self.target_pair_index
            self.target_pair_steps = 0
            self.transition_gap_steps = 0
            return True
        return False

    def target_contact_score(self, feet):
        """Give partial guidance for loading target feet and lifting the others."""
        return max(0.0, (len(feet & self.target_pair) - len(feet - self.target_pair)) / 2.0)

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)

        feet, body_contact = self.ground_contacts()
        grounded_feet = len(feet)
        torso_height = float(self.get_body_com("torso")[2])

        torso_up_alignment = float(
            self.data.body("torso").xmat[8]
        )

        upright_reward = float(
            np.clip(
                (torso_up_alignment - 0.1) / 0.7,
                0.0,
                1.0,
            )
        )

        upside_down = torso_up_alignment < -0.2

        state_invalid = not np.isfinite(self.state_vector()).all()

        terminated = bool(upside_down or state_invalid)

        termination_penalty = 20.0 if terminated else 0.0

        height_reward = float(
            np.clip(
                (torso_height - 0.35) / 0.15,
                0.0,
                1.0,
            )
        )

        contact_score = max(
            0.0,
            1.0 - 0.5 * abs(grounded_feet - 2),
        )

        stability_penalty = float(
            0.02 * np.sum(np.square(self.data.qvel[3:6]))
        )

        control_penalty = float(
            0.02 * np.sum(np.square(action))
        )

        in_two_leg_pose = not terminated and self.is_two_leg_pose(
            grounded_feet, torso_height, torso_up_alignment, body_contact
        )

        on_target_pair = in_two_leg_pose and feet == self.target_pair
        projected_hold_steps = self.target_pair_steps + 1 if on_target_pair else 0
        hold_progress = min(projected_hold_steps / self.dwell_steps, 1.0)

        # Make valid two-leg time the dominant dense objective. Progress grows
        # only while the same requested pair is held without interruption.
        two_leg_reward = 4.0 * float(in_two_leg_pose)
        target_contact_reward = (
            0.5
            * self.target_contact_score(feet)
            * height_reward
            * upright_reward
            if not body_contact and not terminated else 0.0
        )
        hold_progress_reward = 2.0 * hold_progress if on_target_pair else 0.0
        completed_hold = self.update_task_metrics(
            feet, in_two_leg_pose, failed=body_contact or terminated
        )
        target_pair_reward = 2.0 * float(completed_hold)
        body_contact_penalty = 4.0 if body_contact else 0.0
        self.body_contact_steps += int(body_contact)

        reward = (
            contact_score
            + height_reward
            + upright_reward
            + two_leg_reward
            + target_contact_reward
            + hold_progress_reward
            + target_pair_reward
            - body_contact_penalty
            - stability_penalty
            - control_penalty
            - termination_penalty
        )

        # update_task_metrics can change the target pair. Return an observation
        # that describes the state used for the next action.
        observation = self._get_obs()

        info["two_leg_steps"] = self.two_leg_steps
        info["two_leg_seconds"] = self.two_leg_steps * self.dt
        info["longest_two_leg_seconds"] = self.longest_two_leg_streak * self.dt
        info["support_switches"] = self.support_switches
        info["target_holds"] = self.target_holds
        info["pair_a_seconds"] = self.pair_steps[0] * self.dt
        info["pair_b_seconds"] = self.pair_steps[1] * self.dt
        info["two_leg_fraction"] = self.two_leg_steps / self.episode_steps
        info["body_contact_fraction"] = self.body_contact_steps / self.episode_steps
        info["reward_version"] = self.reward_version
        info["is_two_leg_pose"] = in_two_leg_pose
        info["grounded_feet"] = grounded_feet
        info["body_contact"] = body_contact
        info["torso_height"] = torso_height
        info["reward_height"] = height_reward
        info["reward_upright"] = upright_reward
        info["reward_two_leg"] = two_leg_reward
        info["reward_success"] = two_leg_reward
        info["reward_target_pair"] = target_pair_reward
        info["reward_target_contact"] = target_contact_reward
        info["reward_hold_progress"] = hold_progress_reward
        info["hold_progress"] = hold_progress
        info["contact_score"] = contact_score
        info["target_pair_index"] = self.target_pair_index
        info["target_pair_steps"] = self.target_pair_steps
        info["penalty_stability"] = stability_penalty
        info["penalty_control"] = control_penalty
        info["torso_up_alignment"] = torso_up_alignment
        info["upside_down"] = upside_down
        info["penalty_termination"] = termination_penalty
        info["state_invalid"] = state_invalid
        info["penalty_body_contact"] = body_contact_penalty

        return observation, reward, terminated, truncated, info


register(
    id="AntTwoLeg-v0",
    entry_point="cleanrl.two_leg_ant_env:TwoLegAntEnv",
    max_episode_steps=1000,
)
