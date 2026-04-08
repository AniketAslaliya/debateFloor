import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

import requests


TASKS = [
    "clean_claim",
    "contradictory_claim",
    "coordinated_fraud",
    "identity_fraud",
]


def post(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def step(base_url: str, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    return post(base_url, "/step", {"session_id": session_id, "action": action})


def run_policy(base_url: str, task_id: str, seed: int) -> Dict[str, Any]:
    reset = post(base_url, "/reset", {"task_id": task_id, "seed": seed})
    session_id = str(reset["session_id"])

    if task_id == "clean_claim":
        for doc in ["DOC-1", "DOC-2", "DOC-3"]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "validate clean claim docs",
                },
            )
        step(
            base_url,
            session_id,
            {
                "action_type": "estimate_payout",
                "parameters": {"amount_inr": 50000},
                "reasoning": "estimate conservative payout",
            },
        )
        return step(
            base_url,
            session_id,
            {
                "action_type": "approve_claim",
                "parameters": {"reason": "clean evidence trail", "payout_amount": 50000},
                "reasoning": "approve clean claim",
                "confidence": 0.95,
            },
        )

    if task_id == "contradictory_claim":
        for doc in ["DOC-10", "DOC-11", "DOC-12", "DOC-13"]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "detect contradictions",
                },
            )
        step(
            base_url,
            session_id,
            {
                "action_type": "lookup_policy_history",
                "parameters": {},
                "reasoning": "check for prior similar procedures",
            },
        )
        for flag_id, evidence in [
            ("date_mismatch", "incident date appears after admission date"),
            ("cost_inflation", "claimed cost is 2.4x the standard rate"),
            ("signature_mismatch", "doctor signature differs from clinic reference"),
            ("prior_similar_claim", "history shows the same appendectomy 8 months earlier"),
        ]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "flag_fraud_signal",
                    "parameters": {"flag_id": flag_id, "evidence": evidence},
                    "reasoning": "flag with explicit evidence",
                },
            )
        return step(
            base_url,
            session_id,
            {
                "action_type": "deny_claim",
                "parameters": {"reason": "multiple grounded contradictions"},
                "reasoning": "deny high-risk contradictory claim",
                "confidence": 0.8,
            },
        )

    if task_id == "coordinated_fraud":
        for doc in ["DOC-21", "DOC-22", "DOC-23"]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "collect cross-claim evidence",
                },
            )
        for claim_id in ["CLM-GROUP-302", "CLM-GROUP-303", "CLM-GROUP-304"]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "query_linked_claim",
                    "parameters": {"claim_id": claim_id},
                    "reasoning": "expand the linked-claim cluster",
                },
            )
        for flag_id, evidence in [
            ("shared_repair_shop_far", "shared repair shop is unusually far from each incident site"),
            ("shared_emergency_contact", "queried linked claims share the same emergency contact phone"),
            ("near_identical_descriptions", "narratives are near-identical across linked claims"),
            ("recent_policy_cluster", "all policies were purchased within 30 days before the incident"),
            ("clustered_policy_broker", "the surfaced fourth claim shares broker BRK-441"),
        ]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "flag_fraud_signal",
                    "parameters": {"flag_id": flag_id, "evidence": evidence},
                    "reasoning": "flag coordinated-fraud indicator with evidence",
                },
            )
        return step(
            base_url,
            session_id,
            {
                "action_type": "request_investigation",
                "parameters": {
                    "target_claim_ids": [
                        "CLM-GROUP-301",
                        "CLM-GROUP-302",
                        "CLM-GROUP-303",
                        "CLM-GROUP-304",
                    ],
                    "reason": "linked multi-claim fraud pattern with grounded evidence",
                },
                "reasoning": "escalate the full cluster for SIU review",
                "confidence": 0.9,
            },
        )

    if task_id == "identity_fraud":
        step(
            base_url,
            session_id,
            {
                "action_type": "verify_identity",
                "parameters": {},
                "reasoning": "cross-check registry and hospital records",
            },
        )
        step(
            base_url,
            session_id,
            {
                "action_type": "lookup_policy_history",
                "parameters": {},
                "reasoning": "confirm suspiciously recent policy age",
            },
        )
        for doc in ["DOC-31", "DOC-32", "DOC-33", "DOC-34"]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "validate identity-fraud evidence",
                },
            )
        for flag_id, evidence in [
            ("identity_mismatch", "national registry has no record matching the claimant identity"),
            ("hospital_no_record", "hospital record does not match the claimant name and DOB"),
            ("recent_policy_purchase", "policy was opened only days before the incident"),
            ("dob_inconsistency", "DOB on the ID proof conflicts with the policy application"),
        ]:
            step(
                base_url,
                session_id,
                {
                    "action_type": "flag_fraud_signal",
                    "parameters": {"flag_id": flag_id, "evidence": evidence},
                    "reasoning": "flag discovered identity-fraud evidence",
                },
            )
        return step(
            base_url,
            session_id,
            {
                "action_type": "deny_claim",
                "parameters": {"reason": "ghost claimant pattern confirmed"},
                "reasoning": "deny identity fraud claim",
                "confidence": 0.9,
            },
        )

    raise ValueError(f"Unsupported task '{task_id}'")


def _task_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for task_id, task_rows in grouped.items():
        summary[task_id] = {
            "avg_reward": mean(r["reward"] for r in task_rows),
            "avg_steps": mean(r["steps"] for r in task_rows),
            "completion_rate": sum(1 for r in task_rows if r["done"]) / float(len(task_rows)),
        }
    return summary


def make_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Generated at: {report['generated_at']}",
        f"Base URL: {report['base_url']}",
        "",
        "## Per-run Results",
        "",
        "| Task | Seed | Done | Reward | Steps | Variant | Discovered | Evidence Quality | Exploit Penalty |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {task_id} | {seed} | {done} | {reward:.4f} | {steps} | {variant_id} | {discovered_signal_count} | {evidence_quality:.4f} | {exploit_penalty:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            "| Task | Avg Reward | Avg Steps | Completion Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for task_id, summary in report["task_summary"].items():
        lines.append(
            f"| {task_id} | {summary['avg_reward']:.4f} | {summary['avg_steps']:.2f} | {summary['completion_rate']:.2%} |"
        )

    lines.append("")
    lines.append(f"Overall Average Reward: {report['average_reward']:.4f}")
    lines.append(f"Overall Completion Rate: {report['completion_rate']:.2%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded evaluation report for the insurance claim env.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--seeds", default="7,17,27,42,99", help="Comma-separated seed list")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for task_id in TASKS:
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
                    "steps": int(obs.get("step_number", 0) or 0),
                    "variant_id": int(obs.get("metadata", {}).get("variant_id", 0)),
                    "discovered_signal_count": len(obs.get("discovered_signals", []) or []),
                    "evidence_quality": float(rb.get("evidence_quality_score", 0.0) or 0.0),
                    "exploit_penalty": float(rb.get("exploit_penalty", 0.0) or 0.0),
                }
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "rows": rows,
        "task_summary": _task_summary(rows),
        "average_reward": mean(r["reward"] for r in rows) if rows else 0.0,
        "completion_rate": sum(1 for r in rows if r["done"]) / float(len(rows) or 1),
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
