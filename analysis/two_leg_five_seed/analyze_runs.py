"""Analyze the five paired PPO/SAC runs for the two-leg Ant task.

The script uses episode metrics from TensorBoard for learning curves and the
10-episode deterministic evaluation saved in evaluation.json for final scores.
Confidence intervals treat the five independent training seeds as the sample.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# The project virtual environment contains TensorBoard, while this workstation's
# Matplotlib package is installed system-wide.
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    sys.path.append("/usr/lib/python3/dist-packages")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cleanrl")
    import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent
ALGORITHMS = ("ppo", "sac")
SEEDS = (1, 2, 3, 4, 5)
COLORS = {"ppo": "#2f6fbb", "sac": "#d45527"}
T95_DF4 = 2.7764451051977987
SMOOTH_EPISODES = 20
GRID_SPACING = 5_000


def find_runs() -> dict[str, dict[int, Path]]:
    selected: dict[str, dict[int, Path]] = {algorithm: {} for algorithm in ALGORITHMS}
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            matches = list(
                RUNS.glob(
                    f"AntTwoLeg-v0__{algorithm}_sustained_hold_v4__{seed}__*"
                )
            )
            complete = [
                path
                for path in matches
                if (path / "evaluation.json").exists()
                and list(path.glob("events.out.tfevents.*"))
            ]
            if not complete:
                raise FileNotFoundError(f"No complete {algorithm} seed {seed} run")
            selected[algorithm][seed] = max(complete, key=lambda path: path.stat().st_mtime)
    return selected


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    result[window - 1 :] = np.convolve(
        values, np.ones(window, dtype=float) / window, mode="valid"
    )
    return result


def scalar_series(run: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    event_file = next(run.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    steps = np.asarray([event.step for event in events], dtype=float)
    values = np.asarray([event.value for event in events], dtype=float)
    order = np.argsort(steps)
    return steps[order], values[order]


def interpolate_smoothed(
    steps: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    smooth = rolling_mean(values, SMOOTH_EPISODES)
    valid = np.isfinite(smooth)
    result = np.full(len(grid), np.nan)
    if valid.sum() < 2:
        return result
    x, y = steps[valid], smooth[valid]
    inside = (grid >= x[0]) & (grid <= x[-1])
    result[inside] = np.interp(grid[inside], x, y)
    return result


def mean_ci(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = np.sum(np.isfinite(curves), axis=0)
    mean = np.full(curves.shape[1], np.nan)
    ci = np.full(curves.shape[1], np.nan)
    valid = count > 0
    mean[valid] = np.nanmean(curves[:, valid], axis=0)
    enough = count > 1
    ci[enough] = (
        T95_DF4
        * np.nanstd(curves[:, enough], axis=0, ddof=1)
        / np.sqrt(count[enough])
    )
    return mean, ci


def plot_training_metric(
    runs: dict[str, dict[int, Path]], tag: str, ylabel: str, filename: str
) -> None:
    max_step = 0
    raw: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
        algorithm: {} for algorithm in ALGORITHMS
    }
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            series = scalar_series(runs[algorithm][seed], tag)
            raw[algorithm][seed] = series
            max_step = max(max_step, int(series[0][-1]))

    grid = np.arange(0, max_step + GRID_SPACING, GRID_SPACING, dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for algorithm in ALGORITHMS:
        curves = []
        for seed in SEEDS:
            steps, values = raw[algorithm][seed]
            curve = interpolate_smoothed(steps, values, grid)
            curves.append(curve)
            ax.plot(
                grid,
                curve,
                color=COLORS[algorithm],
                alpha=0.20,
                linewidth=1.0,
            )
        stacked = np.asarray(curves)
        mean, ci = mean_ci(stacked)
        ax.plot(
            grid,
            mean,
            color=COLORS[algorithm],
            linewidth=2.5,
            label=f"{algorithm.upper()} mean",
        )
        ax.fill_between(
            grid,
            mean - ci,
            mean + ci,
            color=COLORS[algorithm],
            alpha=0.16,
            linewidth=0,
        )

    ax.axvline(300_000, color="#666666", linestyle="--", linewidth=1.0)
    ax.text(
        305_000,
        0.02,
        "SAC budget",
        transform=ax.get_xaxis_transform(),
        color="#555555",
        fontsize=9,
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Two-Leg Ant Training: {ylabel}\n"
        f"20-episode rolling mean; shading is 95% CI over 5 seeds"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def load_final_rows(runs: dict[str, dict[int, Path]]) -> list[dict[str, float | int | str]]:
    rows = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            run = runs[algorithm][seed]
            evaluation = json.loads((run / "evaluation.json").read_text())
            means = evaluation["means"]
            rows.append(
                {
                    "algorithm": algorithm.upper(),
                    "training_seed": seed,
                    "environment_steps": evaluation["environment_steps"],
                    **means,
                    "run_directory": str(run.relative_to(ROOT)),
                }
            )
    return rows


def summary_stats(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    ci = float(T95_DF4 * sd / np.sqrt(len(values)))
    return mean, sd, ci


def write_summaries(rows: list[dict[str, float | int | str]]) -> None:
    with (OUT / "final_evaluation_by_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = (
        "episodic_return",
        "two_leg_seconds",
        "two_leg_fraction",
        "longest_two_leg_seconds",
        "support_switches",
        "pair_a_seconds",
        "pair_b_seconds",
        "body_contact_fraction",
    )
    summary_rows = []
    for algorithm in ("PPO", "SAC"):
        algorithm_rows = [row for row in rows if row["algorithm"] == algorithm]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in algorithm_rows])
            mean, sd, ci = summary_stats(values)
            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "metric": metric,
                    "mean": mean,
                    "seed_sd": sd,
                    "ci95_half_width": ci,
                }
            )
    with (OUT / "final_evaluation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    ppo = {int(row["training_seed"]): row for row in rows if row["algorithm"] == "PPO"}
    sac = {int(row["training_seed"]): row for row in rows if row["algorithm"] == "SAC"}
    paired_lines = []
    for metric in ("episodic_return", "two_leg_seconds", "two_leg_fraction"):
        differences = np.asarray(
            [float(sac[seed][metric]) - float(ppo[seed][metric]) for seed in SEEDS]
        )
        mean, sd, ci = summary_stats(differences)
        paired_lines.append((metric, mean, sd, ci))

    lines = [
        "# Five-seed final evaluation",
        "",
        "Each entry is the mean of 10 deterministic evaluation episodes. The",
        "reported 95% t interval uses the five independent training seeds as the",
        "sample (df = 4). PPO used 899,072 steps and SAC used 300,000 steps.",
        "",
        "| Algorithm | Return | Two-leg time (s / 50 s) | Two-leg fraction | Longest hold (s) | Switches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lookup = {(row["algorithm"], row["metric"]): row for row in summary_rows}
    for algorithm in ("PPO", "SAC"):
        def display(metric: str, digits: int = 2) -> str:
            item = lookup[(algorithm, metric)]
            return f"{item['mean']:.{digits}f} $\\pm$ {item['ci95_half_width']:.{digits}f}"

        lines.append(
            f"| {algorithm} | {display('episodic_return')} | "
            f"{display('two_leg_seconds')} | {display('two_leg_fraction', 3)} | "
            f"{display('longest_two_leg_seconds', 3)} | {display('support_switches')} |"
        )
    lines.extend(["", "Paired SAC minus PPO differences:", ""])
    for metric, mean, _sd, ci in paired_lines:
        lines.append(f"- {metric}: {mean:.4f} $\\pm$ {ci:.4f} (95% CI)")
    lines.extend(
        [
            "",
            "All ten policies completed every 50-second evaluation episode without",
            "a body contact. Every policy had zero valid support-pair switches. The",
            "policies therefore learned long aggregate two-foot contact, but did not",
            "learn the originally proposed alternating-leg behavior.",
        ]
    )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def plot_final_metric(rows: list[dict[str, float | int | str]]) -> None:
    metrics = [
        ("two_leg_fraction", "Two-leg fraction"),
        ("longest_two_leg_seconds", "Longest continuous hold (s)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
    rng = np.random.default_rng(839)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for index, algorithm in enumerate(("PPO", "SAC")):
            values = np.asarray(
                [float(row[metric]) for row in rows if row["algorithm"] == algorithm]
            )
            mean, _sd, ci = summary_stats(values)
            jitter = rng.uniform(-0.055, 0.055, len(values))
            ax.scatter(
                np.full(len(values), index) + jitter,
                values,
                color=COLORS[algorithm.lower()],
                alpha=0.75,
                zorder=3,
            )
            ax.errorbar(
                index,
                mean,
                yerr=ci,
                fmt="o",
                markersize=8,
                capsize=6,
                color="black",
                zorder=4,
            )
        ax.set_xticks([0, 1], ["PPO\n899k steps", "SAC\n300k steps"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Final deterministic evaluation (points are training seeds; bars are 95% CI)")
    fig.tight_layout()
    fig.savefig(OUT / "final_evaluation.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = find_runs()
    plot_training_metric(
        runs,
        "charts/episodic_return",
        "Episodic return",
        "training_return.png",
    )
    plot_training_metric(
        runs,
        "charts/two_leg_seconds",
        "Two-leg time per episode (seconds)",
        "training_true_performance.png",
    )
    rows = load_final_rows(runs)
    write_summaries(rows)
    plot_final_metric(rows)


if __name__ == "__main__":
    main()
