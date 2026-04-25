"""
train/generate_eval_report.py

Regenerates `reports/eval_report.json` and `reports/eval_report.md` from a
live DebateFloor environment using the canonical
`inference_debatefloor.py:STRATEGIES`.

Why this exists (NEW-1 / FATAL-4):
  - The previous reports/eval_report.json was 3 weeks old and had
    `variant_id: 0` and `evidence_quality: 0.0` for every row, contradicting
    the FATAL-3 + FATAL-4 server-side fixes.
  - PLAN.md mentioned `pre_validation_script.py --output ... --seeds ...`
    but those flags were never implemented in that script.
  - This is the dedicated regeneration tool.

What it does:
  - Sweeps every task registered in inference_debatefloor.STRATEGIES
    (currently 5 — clean_claim, contradictory_claim, distribution_shift_claim,
    coordinated_fraud, identity_fraud) × 5 distinct seeds
    (7, 11, 13, 19, 25) covering all 5 variant_ids
    (variant_id = abs(seed) % 5 — see app/tasks.py:548).
  - Per row captures: task_id, seed, done, reward, variant_id,
    evidence_quality, exploit_penalty.
  - Writes JSON (schema-compatible with the previous file) + Markdown.

Usage:
  $ python train/generate_eval_report.py [--base-url http://localhost:7860]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# Make the inference baseline importable from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference_debatefloor import (  # noqa: E402
    DebateFloorClient,
    STRATEGIES,
)


# Seeds chosen so that abs(seed) % 5 covers all 5 variants:
#   7 -> 2, 11 -> 1, 13 -> 3, 19 -> 4, 25 -> 0
SEEDS = [7, 11, 13, 19, 25]
TASKS = list(STRATEGIES.keys())


def run_one(client: DebateFloorClient, task_id: str, seed: int) -> dict:
    obs = client.reset(task_id=task_id, seed=seed)
    actions = STRATEGIES[task_id](client, obs)

    last = None
    steps = 0
    for action in actions:
        try:
            last = client.step(action)
            steps += 1
            if last.get("done"):
                break
        except Exception as exc:
            return {
                "task_id": task_id,
                "seed": seed,
                "done": False,
                "reward": 0.0,
                "variant_id": None,
                "evidence_quality": 0.0,
                "exploit_penalty": 0.0,
                "error": str(exc),
            }

    if last is None:
        return {
            "task_id": task_id,
            "seed": seed,
            "done": False,
            "reward": 0.0,
            "variant_id": None,
            "evidence_quality": 0.0,
            "exploit_penalty": 0.0,
            "error": "no steps executed",
        }

    obs = last.get("observation", {})
    metadata = obs.get("metadata", {}) or {}
    breakdown = obs.get("reward_breakdown", {}) or {}

    return {
        "task_id":         task_id,
        "seed":            seed,
        "done":            bool(last.get("done", False)),
        "reward":          round(float(last.get("reward", 0.0)), 4),
        "variant_id":      int(metadata.get("variant_id", 0)),
        "evidence_quality": round(float(breakdown.get("evidence_quality_score", 0.0)), 4),
        "exploit_penalty": round(float(metadata.get("exploit_penalty", 0.0)), 4),
        "steps":           steps,
    }


def write_markdown(payload: dict, path: Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Evaluation Report",
        "",
        f"Generated at: {payload['generated_at']}",
        f"Base URL: {payload['base_url']}",
        f"Tasks: {', '.join(sorted({r['task_id'] for r in rows}))}",
        f"Seeds: {', '.join(str(s) for s in sorted({r['seed'] for r in rows}))}",
        f"Distinct variant_ids: {sorted({r['variant_id'] for r in rows if r['variant_id'] is not None})}",
        "",
        "| Task | Seed | Variant | Steps | Done | Reward | Evidence Quality | Exploit Penalty |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["task_id"], x["seed"])):
        done_glyph = "yes" if r["done"] else "no"
        lines.append(
            f"| {r['task_id']} | {r['seed']} | {r['variant_id']} | "
            f"{r.get('steps', '-')} | {done_glyph} | "
            f"{r['reward']:.4f} | {r['evidence_quality']:.4f} | "
            f"{r['exploit_penalty']:.4f} |"
        )
    lines += [
        "",
        f"Average Reward: {payload['average_reward']:.4f}",
        f"Completion Rate: {payload['completion_rate'] * 100:.2f}%",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate reports/eval_report.{json,md}")
    parser.add_argument("--base-url", default="http://localhost:7860")
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "reports" / "eval_report.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "reports" / "eval_report.md"),
    )
    args = parser.parse_args()

    print(f"Generating eval report against {args.base_url}")
    print(f"Tasks: {TASKS}")
    print(f"Seeds: {SEEDS} (variant_ids: {sorted({abs(s) % 5 for s in SEEDS})})")
    print()

    rows = []
    for task_id in TASKS:
        for seed in SEEDS:
            client = DebateFloorClient(args.base_url)
            row = run_one(client, task_id, seed)
            rows.append(row)
            print(
                f"  {task_id:<28s} seed={seed:>3d} variant={row['variant_id']} "
                f"reward={row['reward']:.4f} ev_q={row['evidence_quality']:.4f} "
                f"exp_pen={row['exploit_penalty']:.4f} done={row['done']}"
            )

    completed = [r for r in rows if r.get("done")]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url":     args.base_url,
        "rows":         rows,
        "average_reward":  round(mean(r["reward"] for r in completed) if completed else 0.0, 4),
        "completion_rate": round(len(completed) / len(rows) if rows else 0.0, 4),
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(payload, out_md)

    print()
    print(f"Wrote {out_json} ({len(rows)} rows)")
    print(f"Wrote {out_md}")
    print(f"Average reward: {payload['average_reward']:.4f}")
    print(f"Completion rate: {payload['completion_rate'] * 100:.2f}%")

    distinct_variants = sorted({r["variant_id"] for r in rows if r["variant_id"] is not None})
    distinct_rewards = sorted({r["reward"] for r in rows})
    nonzero_evidence = sum(1 for r in rows if r["evidence_quality"] > 0.0)
    print()
    print("Invariants (the FATAL-3 / FATAL-4 acceptance criteria):")
    print(f"  distinct variant_ids       : {distinct_variants} (expected: > 1 distinct)")
    print(f"  distinct rewards           : {len(distinct_rewards)} unique values")
    print(f"  rows with evidence_quality > 0 : {nonzero_evidence} / {len(rows)}")

    failed = []
    if len(distinct_variants) <= 1:
        failed.append("FATAL-4 invariant: variant_ids still constant")
    if nonzero_evidence == 0:
        failed.append("FATAL-3 invariant: evidence_quality still zero everywhere")
    if len(distinct_rewards) <= 1:
        failed.append("rewards are constant — investigate")

    if failed:
        for f in failed:
            print(f"  FAIL: {f}")
        return 1
    print("  PASS: all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
