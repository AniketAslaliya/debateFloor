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


def reset(base_url: str, task_id: str) -> str:
    payload = post(base_url, "/reset", {"task_id": task_id})
    return str(payload["session_id"])


def step(base_url: str, session_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    return post(base_url, "/step", {"session_id": session_id, "action": action})


def check(cond: bool, message: str) -> None:
    if cond:
        print(f"PASS: {message}")
    else:
        print(f"FAIL: {message}")
        raise AssertionError(message)


def run_clean_claim(base_url: str) -> Dict[str, Any]:
    session_id = reset(base_url, "clean_claim")
    for doc in ["DOC-1", "DOC-2", "DOC-3"]:
        step(
            base_url,
            session_id,
            {
                "action_type": "validate_document",
                "parameters": {"doc_id": doc},
                "reasoning": "validate",
            },
        )
    step(
        base_url,
        session_id,
        {
            "action_type": "estimate_payout",
            "parameters": {"amount_inr": 50000},
            "reasoning": "estimate",
        },
    )
    return step(
        base_url,
        session_id,
        {
            "action_type": "approve_claim",
            "parameters": {"payout_amount": 50000, "reason": "consistent docs"},
            "reasoning": "approve",
        },
    )


def run_contradictory_claim(base_url: str) -> Dict[str, Any]:
    session_id = reset(base_url, "contradictory_claim")
    for doc in ["DOC-10", "DOC-11", "DOC-12", "DOC-13"]:
        step(
            base_url,
            session_id,
            {
                "action_type": "validate_document",
                "parameters": {"doc_id": doc},
                "reasoning": "find contradictions",
            },
        )
    step(
        base_url,
        session_id,
        {
            "action_type": "lookup_policy_history",
            "parameters": {},
            "reasoning": "check for prior similar claim",
        },
    )
    for flag, evidence in [
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
                "parameters": {"flag_id": flag, "evidence": evidence},
                "reasoning": "flag",
            },
        )
    return step(
        base_url,
        session_id,
        {
            "action_type": "deny_claim",
            "parameters": {"reason": "multiple contradictions"},
            "reasoning": "deny",
            "confidence": 0.8,
        },
    )


def run_coordinated_fraud(base_url: str) -> Dict[str, Any]:
    session_id = reset(base_url, "coordinated_fraud")
    for doc in ["DOC-21", "DOC-22", "DOC-23"]:
        step(
            base_url,
            session_id,
            {
                "action_type": "validate_document",
                "parameters": {"doc_id": doc},
                "reasoning": "cross-claim check",
            },
        )
    for claim_id in ["CLM-GROUP-302", "CLM-GROUP-303", "CLM-GROUP-304"]:
        step(
            base_url,
            session_id,
            {
                "action_type": "query_linked_claim",
                "parameters": {"claim_id": claim_id},
                "reasoning": "expand linked cluster",
            },
        )
    for flag, evidence in [
        ("shared_repair_shop_far", "shared repair shop is unusually far from incident site"),
        ("shared_emergency_contact", "queried claims share the same emergency contact phone"),
        ("near_identical_descriptions", "narratives are near-identical across linked claims"),
        ("recent_policy_cluster", "all policies were purchased within 30 days before the incident"),
        ("clustered_policy_broker", "the surfaced fourth claim shares broker BRK-441"),
    ]:
        step(
            base_url,
            session_id,
            {
                "action_type": "flag_fraud_signal",
                "parameters": {"flag_id": flag, "evidence": evidence},
                "reasoning": "flag",
            },
        )
    return step(
        base_url,
        session_id,
        {
            "action_type": "request_investigation",
            "parameters": {
                "target_claim_ids": ["CLM-GROUP-301", "CLM-GROUP-302", "CLM-GROUP-303", "CLM-GROUP-304"],
                "reason": "coordinated pattern",
            },
            "reasoning": "escalate",
            "confidence": 0.9,
        },
    )


def run_identity_fraud(base_url: str) -> Dict[str, Any]:
    session_id = reset(base_url, "identity_fraud")
    step(
        base_url,
        session_id,
        {
            "action_type": "verify_identity",
            "parameters": {},
            "reasoning": "check registry and hospital records",
        },
    )
    step(
        base_url,
        session_id,
        {
            "action_type": "lookup_policy_history",
            "parameters": {},
            "reasoning": "check policy age",
        },
    )
    for doc in ["DOC-31", "DOC-32", "DOC-33", "DOC-34"]:
        step(
            base_url,
            session_id,
            {
                "action_type": "validate_document",
                "parameters": {"doc_id": doc},
                "reasoning": "validate identity evidence",
            },
        )
    for flag, evidence in [
        ("identity_mismatch", "national registry has no record matching the claimant identity"),
        ("hospital_no_record", "hospital records do not match the claimant name and DOB"),
        ("recent_policy_purchase", "policy was opened only days before the incident"),
        ("dob_inconsistency", "DOB on the ID proof conflicts with the policy application"),
    ]:
        step(
            base_url,
            session_id,
            {
                "action_type": "flag_fraud_signal",
                "parameters": {"flag_id": flag, "evidence": evidence},
                "reasoning": "flag",
            },
        )
    return step(
        base_url,
        session_id,
        {
            "action_type": "deny_claim",
            "parameters": {"reason": "ghost claimant pattern confirmed"},
            "reasoning": "deny",
            "confidence": 0.9,
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
    check(len(task_list) >= 4, "/tasks returns at least 4 tasks")

    t1 = run_clean_claim(base_url)
    validate_terminal_result("clean_claim", t1)

    t2 = run_contradictory_claim(base_url)
    validate_terminal_result("contradictory_claim", t2)

    t3 = run_coordinated_fraud(base_url)
    validate_terminal_result("coordinated_fraud", t3)

    t4 = run_identity_fraud(base_url)
    validate_terminal_result("identity_fraud", t4)

    print("\nLive Space check completed against the deployed HF demo.")

    print("\nAll checks passed.")
    print(
        json.dumps(
            {
                "clean_claim": t1,
                "contradictory_claim": t2,
                "coordinated_fraud": t3,
                "identity_fraud": t4,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nEvaluation failed: {exc}")
        raise
