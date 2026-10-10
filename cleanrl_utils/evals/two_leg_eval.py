"""Shared deterministic evaluation for the two-leg PPO and SAC experiments."""

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

import cleanrl.two_leg_ant_env


TASK_METRICS = (
    "two_leg_seconds", "longest_two_leg_seconds", "support_switches",
    "target_holds", "pair_a_seconds", "pair_b_seconds", "two_leg_fraction",
    "body_contact_fraction",
)


def save_run_context(run_dir, args, env, script_path):
    """Preserve the settings and source that generated each run's results."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "args": vars(args),
        "reward_version": env.unwrapped.reward_version,
        "observation_shape": list(env.observation_space.shape),
        "dt": env.unwrapped.dt,
        "dwell_seconds": env.unwrapped.dwell_seconds,
        "max_transition_seconds": env.unwrapped.max_transition_seconds,
    }
    (run_dir / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copyfile(cleanrl.two_leg_ant_env.__file__, run_dir / "environment_source.py")
    shutil.copyfile(script_path, run_dir / "training_source.py")


def save_observation_stats(env, path):
    """Save PPO's actual training normalization, including its epsilon."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, gym.wrappers.NormalizeObservation):
            np.savez(
                path, mean=env.obs_rms.mean, var=env.obs_rms.var,
                count=env.obs_rms.count, epsilon=env.epsilon,
            )
            return
        env = env.env
    raise ValueError("PPO training environment has no observation normalizer")


def load_observation_stats(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def normalize_observation(obs, stats):
    # Fixed arrays: evaluation must never update the training statistics.
    if stats is not None:
        obs = np.clip((obs - stats["mean"]) / np.sqrt(stats["var"] + stats["epsilon"]), -10, 10)
    return np.asarray(obs, dtype=np.float32)


def evaluate_policy(model, algorithm, env_id, seed, eval_episodes, video_dir=None, observation_stats=None):
    """Reset each episode with seed+i, so both algorithms see identical starts."""
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be positive")
    if algorithm not in ("ppo", "sac"):
        raise ValueError("algorithm must be ppo or sac")
    if algorithm == "ppo" and observation_stats is None:
        raise ValueError("PPO evaluation requires saved training observation statistics")
    env = gym.make(env_id, **({"render_mode": "rgb_array"} if video_dir else {}))
    if video_dir:
        env = gym.wrappers.RecordVideo(
            env, str(video_dir), episode_trigger=lambda episode_id: episode_id == 0,
            name_prefix=f"{algorithm}-eval",
        )
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    results = []
    try:
        for episode in range(eval_episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            length = 0
            while True:
                tensor = torch.as_tensor(
                    normalize_observation(obs, observation_stats), device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    action = model.actor_mean(tensor) if algorithm == "ppo" else model.get_deterministic_action(tensor)
                # Matches PPO's ClipAction wrapper; SAC's mean is already bounded.
                action = np.clip(action.cpu().numpy()[0], env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                if terminated or truncated:
                    result = {
                        "seed": seed + episode, "episodic_return": total_reward,
                        "episodic_length": length, "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        **{key: float(info[key]) for key in TASK_METRICS},
                    }
                    results.append(result)
                    print(f"eval_seed={seed + episode}, return={total_reward:.2f}, "
                          f"two_leg_seconds={info['two_leg_seconds']:.2f}, switches={info['support_switches']}")
                    break
    finally:
        env.close()
        model.train(was_training)
    return results


def record_evaluation(results, run_dir, writer, global_step):
    metric_names = ("episodic_return", "episodic_length", *TASK_METRICS)
    means = {key: float(np.mean([row[key] for row in results])) for key in metric_names}
    summary = {"environment_steps": global_step, "reward_version": cleanrl.two_leg_ant_env.TwoLegAntEnv.reward_version,
               "episodes": results, "means": means}
    (Path(run_dir) / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    if writer is not None:
        for index, row in enumerate(results):
            for key in metric_names:
                writer.add_scalar(f"eval/{key}", row[key], index)
        for key, value in means.items():
            writer.add_scalar(f"eval/mean_{key}", value, global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-id", default="AntTwoLeg-v0")
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gym.make(args.env_id) as env:
        spaces = SimpleNamespace(single_observation_space=env.observation_space, single_action_space=env.action_space)
        if args.algorithm == "ppo":
            from cleanrl.ppo_two_leg import Agent
            model = Agent(spaces)
            stats = load_observation_stats(args.model.parent / "observation_stats.npz")
        else:
            from cleanrl.sac_two_leg import Actor
            model = Actor(spaces)
            stats = None
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    results = evaluate_policy(model, args.algorithm, args.env_id, args.seed, args.eval_episodes,
                              args.output_dir / "video" if args.capture_video else None, stats)
    record_evaluation(results, args.output_dir, None, None)
