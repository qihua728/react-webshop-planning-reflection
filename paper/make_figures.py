from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "formal" / "analysis"
FIGURES = Path(__file__).resolve().parent / "figures"


def load_metrics() -> list[dict[str, float | str]]:
    with (ANALYSIS / "formal_metrics.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    numeric = [
        "mean_reward", "reward_ci_low", "reward_ci_high",
        "success_rate", "success_ci_low", "success_ci_high",
        "mean_steps", "steps_ci_low", "steps_ci_high",
        "mean_llm_calls", "llm_calls_ci_low", "llm_calls_ci_high",
    ]
    for row in rows:
        for field in numeric:
            row[field] = float(row[field])
    return rows


def asymmetric_error(row: dict, value: str, low: str, high: str) -> list[float]:
    return [float(row[value]) - float(row[low]), float(row[high]) - float(row[value])]


def performance(rows: list[dict]) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    rewards = [float(row["mean_reward"]) for row in rows]
    success = [float(row["success_rate"]) for row in rows]
    reward_err = np.array([
        asymmetric_error(row, "mean_reward", "reward_ci_low", "reward_ci_high")
        for row in rows
    ]).T
    success_err = np.array([
        asymmetric_error(row, "success_rate", "success_ci_low", "success_ci_high")
        for row in rows
    ]).T

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    bars_reward = ax.bar(
        x - width / 2, rewards, width, yerr=reward_err, capsize=4,
        color="#2878B5", label="Mean reward", edgecolor="white", linewidth=0.5,
    )
    bars_success = ax.bar(
        x + width / 2, success, width, yerr=success_err, capsize=4,
        color="#F39C34", label="Strict success rate", edgecolor="white", linewidth=0.5,
    )
    ax.set_ylabel("Score / proportion")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.52)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.bar_label(bars_reward, labels=[f"{v:.3f}" for v in rewards], padding=7, fontsize=8)
    ax.bar_label(bars_success, labels=[f"{v:.0%}" for v in success], padding=7, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def efficiency(rows: list[dict]) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    steps = [float(row["mean_steps"]) for row in rows]
    calls = [float(row["mean_llm_calls"]) for row in rows]
    steps_err = np.array([
        asymmetric_error(row, "mean_steps", "steps_ci_low", "steps_ci_high")
        for row in rows
    ]).T
    calls_err = np.array([
        asymmetric_error(row, "mean_llm_calls", "llm_calls_ci_low", "llm_calls_ci_high")
        for row in rows
    ]).T

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    bars_steps = ax.bar(
        x - width / 2, steps, width, yerr=steps_err, capsize=4,
        color="#55A868", label="Environment steps", edgecolor="white", linewidth=0.5,
    )
    bars_calls = ax.bar(
        x + width / 2, calls, width, yerr=calls_err, capsize=4,
        color="#8C6BB1", label="Total LLM calls", edgecolor="white", linewidth=0.5,
    )
    ax.set_ylabel("Mean count per episode")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 17.8)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.bar_label(bars_steps, labels=[f"{v:.2f}" for v in steps], padding=7, fontsize=8)
    ax.bar_label(bars_calls, labels=[f"{v:.2f}" for v in calls], padding=7, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "efficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    performance(rows)
    efficiency(rows)
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
