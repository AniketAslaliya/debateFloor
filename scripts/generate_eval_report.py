import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


def post(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_policy(base_url: str, task_id: str, seed: int) -> Dict[str, Any]:
    post(base_url, "/reset", {"task_id": task_id, "seed": seed})

    if task_id == "clean_claim":
        for doc in ["DOC-1", "DOC-2", "DOC-3"]:
            post(
                base_url,
                "/step",
                {
                    "action": {
                        "action_type": "validate_document",
                        "parameters": {"doc_id": doc},
                        "reasoning": "validate clean claim docs",
                    }
                },
            )
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "estimate_payout",
                    "parameters": {"amount_inr": 50000},
                    "reasoning": "estimate conservative payout",
                }
            },
        )
        out = post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "approve_claim",
                    "parameters": {"reason": "clean evidence trail", "payout_amount": 50000},
                    "reasoning": "approve clean claim",
                }
            },
        )
        return out

    if task_id == "contradictory_claim":
        for doc in ["DOC-10", "DOC-12", "DOC-13"]:
            post(
                base_url,
                "/step",
                {
                    "action": {
                        "action_type": "validate_document",
                        "parameters": {"doc_id": doc},
                        "reasoning": "detect contradictions",
                    }
                },
            )
        for flag_id, evidence in [
            ("date_mismatch", "incident date appears after admission date"),
            ("cost_inflation", "claimed cost is 2.4x standard rate"),
            ("signature_mismatch", "doctor signature differs from clinic reference"),
        ]:
            post(
                base_url,
                "/step",
                {
                    "action": {
                        "action_type": "flag_fraud_signal",
                        "parameters": {"flag_id": flag_id, "evidence": evidence},
                        "reasoning": "flag with explicit evidence",
                    }
                },
            )
        out = post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "deny_claim",
                    "parameters": {"reason": "multiple deterministic contradictions"},
                    "reasoning": "deny high-risk contradictory claim",
                }
            },
        )
        return out

    for doc in ["DOC-21", "DOC-22", "DOC-23"]:
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "collect cross-claim evidence",
                }
            },
        )

    for flag_id, evidence in [
        ("shared_repair_shop_far", "shared repair shop is unusually far from incident site"),
        ("shared_emergency_contact", "two claimants share same emergency contact phone"),
        ("near_identical_descriptions", "narratives are near-identical across linked claims"),
        ("recent_policy_cluster", "all policies purchased within 30 days before incident"),
    ]:
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "flag_fraud_signal",
                    "parameters": {"flag_id": flag_id, "evidence": evidence},
                    "reasoning": "flag coordinated-fraud indicator with evidence",
                }
            },
        )

    out = post(
        base_url,
        "/step",
        {
            "action": {
                "action_type": "request_investigation",
                "parameters": {
                    "target_claim_ids": ["CLM-GROUP-301", "CLM-GROUP-302", "CLM-GROUP-303"],
                    "reason": "linked multi-claim fraud pattern with consistent evidence",
                },
                "reasoning": "escalate full cluster for SIU review",
            }
        },
    )
    return out


def make_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Generated at: {report['generated_at']}",
        f"Base URL: {report['base_url']}",
        "",
        "| Task | Seed | Done | Reward | Variant | Evidence Quality | Exploit Penalty |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {task_id} | {seed} | {done} | {reward:.4f} | {variant_id} | {evidence_quality:.4f} | {exploit_penalty:.4f} |".format(
                task_id=row["task_id"],
                seed=row["seed"],
                done="yes" if row["done"] else "no",
                reward=row["reward"],
                variant_id=row["variant_id"],
                evidence_quality=row["evidence_quality"],
                exploit_penalty=row["exploit_penalty"],
            )
        )
    lines.append("")
    lines.append(f"Average Reward: {report['average_reward']:.4f}")
    lines.append(f"Completion Rate: {report['completion_rate']:.2%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded evaluation report for claim triage env.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--seeds", default="7,17,27", help="Comma-separated seed list")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    tasks = ["clean_claim", "contradictory_claim", "coordinated_fraud"]

    rows: List[Dict[str, Any]] = []
    for task_id in tasks:
        for seed in seeds:
            result = run_policy(base_url, task_id, seed)
            obs = result.get("observation", {})
            rb = obs.get("reward_breakdown", {})
            rows.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "done": bool(result.get("done", False)),
                    "reward": float(result.get("reward", 0.0) or 0.0),
                    "variant_id": int(obs.get("metadata", {}).get("variant_id", 0)),
                    "evidence_quality": float(rb.get("evidence_quality_score", 0.0) or 0.0),
                    "exploit_penalty": float(rb.get("exploit_penalty", 0.0) or 0.0),
                }
            )

    avg_reward = sum(r["reward"] for r in rows) / max(len(rows), 1)
    completion_rate = sum(1 for r in rows if r["done"]) / max(len(rows), 1)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "rows": rows,
        "average_reward": avg_reward,
        "completion_rate": completion_rate,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_report.json"
    md_path = out_dir / "eval_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(make_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
