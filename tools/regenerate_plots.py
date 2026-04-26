"""Regenerate canonical plots as PNG (and SVG) from committed JSON artifacts.

Reads:  reports/training_summary.json
        reports/component_shift_summary.json
Writes: docs/reward_curve.png   (+ docs/reward_curve.svg)
        docs/component_shift.png (+ docs/component_shift.svg)

Run:    python tools/regenerate_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TRAIN_JSON = ROOT / "reports" / "training_summary.json"
COMP_JSON = ROOT / "reports" / "component_shift_summary.json"
REWARD_PNG = ROOT / "docs" / "reward_curve.png"
REWARD_SVG = ROOT / "docs" / "reward_curve.svg"
COMP_PNG = ROOT / "docs" / "component_shift.png"
COMP_SVG = ROOT / "docs" / "component_shift.svg"

_LABEL_ORDER = [
    "Fraud detection",
    "Decision accuracy",
    "Evidence quality",
    "Calibration",
    "Reasoning quality",
]


def regenerate_reward_curve() -> None:
    summary = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    log_history = summary.get("log_history", []) or []

    reward_steps, rewards, loss_steps, losses = [], [], [], []
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row:
            loss_steps.append(step)
            losses.append(row["loss"])
        rv = row.get("reward") or row.get("rewards/mean")
        if rv is not None:
            reward_steps.append(step)
            rewards.append(rv)

    if not (loss_steps or reward_steps):
        print("[WARN] no log_history rows; skipping reward curve")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    if losses:
        ax1.plot(loss_steps, losses, color="#26547c", linewidth=2, label="Training loss")
        ax1.set_ylabel("Loss", color="#26547c")
        ax1.tick_params(axis="y", labelcolor="#26547c")
    ax1.set_xlabel("Training step")
    ax1.grid(True, alpha=0.25)

    if rewards:
        ax2 = ax1.twinx()
        ax2.plot(
            reward_steps,
            rewards,
            color="#06a77d",
            linewidth=2,
            label="Mean reward (training scalar)",
        )
        ax2.set_ylabel(
            "Mean reward (training scalar — unbounded)", color="#06a77d"
        )
        ax2.tick_params(axis="y", labelcolor="#06a77d")
        ax2.annotate(
            "Note: training scalar is unbounded.\nSee eval table for [0,1] clamped scores.",
            xy=(0.02, 0.05),
            xycoords="axes fraction",
            fontsize=9,
            color="gray",
        )

    fig.suptitle(
        "ClaimCourt GRPO Training Progress (training scalar — not eval score)"
    )
    fig.tight_layout()
    fig.savefig(REWARD_PNG, dpi=180, format="png")
    fig.savefig(REWARD_SVG, dpi=180, format="svg")
    plt.close(fig)
    print(f"[OK] {REWARD_PNG}")
    print(f"[OK] {REWARD_SVG}")


def regenerate_component_shift() -> None:
    payload = json.loads(COMP_JSON.read_text(encoding="utf-8"))
    before = payload.get("before") or {}
    after = payload.get("after") or {}
    if not (before and after):
        print("[WARN] component_shift_summary.json missing before/after; skipping")
        return

    labels = [lbl for lbl in _LABEL_ORDER if lbl in before or lbl in after]
    before_values = [before.get(lbl, 0.0) for lbl in labels]
    after_values = [after.get(lbl, 0.0) for lbl in labels]
    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        [i - width / 2 for i in x],
        before_values,
        width,
        label="Before training",
        color="#7a869a",
    )
    ax.bar(
        [i + width / 2 for i in x],
        after_values,
        width,
        label="After training",
        color="#06a77d",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Component score (eval reward — clamped to [0, 1])")
    ax.set_xlabel("Reward component")
    ax.set_title(
        "ClaimCourt: component-score shift before vs after GRPO training (n=6 held-out)"
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for i, v in enumerate(before_values):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color="#7a869a")
    for i, v in enumerate(after_values):
        ax.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, color="#06a77d")

    fig.tight_layout()
    fig.savefig(COMP_PNG, dpi=180, format="png")
    fig.savefig(COMP_SVG, dpi=180, format="svg")
    plt.close(fig)
    print(f"[OK] {COMP_PNG}")
    print(f"[OK] {COMP_SVG}")


def main() -> None:
    regenerate_reward_curve()
    regenerate_component_shift()


if __name__ == "__main__":
    main()
