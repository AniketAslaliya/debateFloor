import argparse
import json
import sys
from typing import Any, Dict, List

import requests


def post(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_json(base_url: str, path: str) -> Dict[str, Any]:
    resp = requests.get(f"{base_url}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"PASS: {message}")
    else:
        print(f"FAIL: {message}")
        raise AssertionError(message)


def run_clean_claim(base_url: str) -> Dict[str, Any]:
    post(base_url, "/reset", {"task_id": "clean_claim"})
    for doc in ["DOC-1", "DOC-2", "DOC-3"]:
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "validate",
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
                "reasoning": "estimate",
            }
        },
    )
    return post(
        base_url,
        "/step",
        {
            "action": {
                "action_type": "approve_claim",
                "parameters": {"payout_amount": 50000, "reason": "consistent docs"},
                "reasoning": "approve",
            }
        },
    )


def run_contradictory_claim(base_url: str) -> Dict[str, Any]:
    post(base_url, "/reset", {"task_id": "contradictory_claim"})
    for doc in ["DOC-10", "DOC-12", "DOC-13"]:
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "find contradictions",
                }
            },
        )
    for flag in ["date_mismatch", "cost_inflation", "signature_mismatch"]:
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "flag_fraud_signal",
                    "parameters": {"flag_id": flag, "evidence": "doc mismatch"},
                    "reasoning": "flag",
                }
            },
        )
    return post(
        base_url,
        "/step",
        {
            "action": {
                "action_type": "deny_claim",
                "parameters": {"reason": "multiple contradictions"},
                "reasoning": "deny",
            }
        },
    )


def run_coordinated_fraud(base_url: str) -> Dict[str, Any]:
    post(base_url, "/reset", {"task_id": "coordinated_fraud"})
    for doc in ["DOC-21", "DOC-22", "DOC-23"]:
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "validate_document",
                    "parameters": {"doc_id": doc},
                    "reasoning": "cross-claim check",
                }
            },
        )
    for flag in [
        "shared_repair_shop_far",
        "shared_emergency_contact",
        "near_identical_descriptions",
        "recent_policy_cluster",
    ]:
        post(
            base_url,
            "/step",
            {
                "action": {
                    "action_type": "flag_fraud_signal",
                    "parameters": {"flag_id": flag, "evidence": "linked claims"},
                    "reasoning": "flag",
                }
            },
        )
    return post(
        base_url,
        "/step",
        {
            "action": {
                "action_type": "request_investigation",
                "parameters": {
                    "target_claim_ids": ["CLM-GROUP-301", "CLM-GROUP-302", "CLM-GROUP-303"],
                    "reason": "coordinated pattern",
                },
                "reasoning": "escalate",
            }
        },
    )


def validate_terminal_result(name: str, payload: Dict[str, Any]) -> None:
    obs = payload.get("observation", {})
    reward = float(payload.get("reward", 0.0) or 0.0)
    done = bool(payload.get("done", False))

    check(done, f"{name}: done is true")
    check(0.0 <= reward <= 1.0, f"{name}: reward in [0.0, 1.0]")
    check(obs.get("status") == "closed", f"{name}: observation status is closed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an HF Space deployment quickly.")
    parser.add_argument(
        "--base-url",
        default="https://aniketasla-insurance-claim-env.hf.space",
        help="Base URL of deployed Space",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Evaluating Space: {base_url}")

    health = get_json(base_url, "/health")
    check(health.get("status") == "healthy", "/health reports healthy")

    tasks = get_json(base_url, "/tasks")
    task_list = tasks.get("tasks", [])
    check(len(task_list) >= 3, "/tasks returns at least 3 tasks")

    t1 = run_clean_claim(base_url)
    validate_terminal_result("clean_claim", t1)

    t2 = run_contradictory_claim(base_url)
    validate_terminal_result("contradictory_claim", t2)

    t3 = run_coordinated_fraud(base_url)
    validate_terminal_result("coordinated_fraud", t3)

    print("\nAll checks passed.")
    print(json.dumps({"clean_claim": t1, "contradictory_claim": t2, "coordinated_fraud": t3}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nEvaluation failed: {exc}")
        raise
