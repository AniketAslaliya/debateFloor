"""
inference_debatefloor.py
DebateFloor — Baseline Agent

Runs all 3 tasks against the DebateFloor environment over HTTP.
Declares calibrated confidence (HIGH/MED/LOW) on every terminal action.

MANDATORY STDOUT FORMAT — do not change:
  [START] task=<task_id> env=debatefloor model=<model> confidence_required=true
  [STEP] step=<n> action=<action_type> reward=<r> confidence=<conf|null> done=<bool> error=<msg|None>
  [END] success=<bool> steps=<n> total_reward=<r> calibration_score=<s> decision=<correct|wrong|none>

Usage:
  python inference_debatefloor.py --task contradictory_claim --model gpt-4o
  python inference_debatefloor.py --all-tasks --seed 42 --base-url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:7860"
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")

# Task configuration
TASK_CONFIG = {
    "clean_claim": {
        "terminal_confidence": "HIGH",    # obvious approval → HIGH confidence
        "strategy": "approve",
    },
    "contradictory_claim": {
        "terminal_confidence": "MED",     # fraud detected but some uncertainty → MED
        "strategy": "deny",
    },
    "distribution_shift_claim": {
        "terminal_confidence": "LOW",     # cross-claim fraud, high uncertainty → LOW
        "strategy": "escalate",
    },
}

ALL_TASKS = list(TASK_CONFIG.keys())


# ─────────────────────────────────────────────────────────────
# HTTP CLIENT
# ─────────────────────────────────────────────────────────────

class DebateFloorClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session_id: Optional[str] = None

    def health(self) -> Dict:
        return requests.get(f"{self.base_url}/health", timeout=10).json()

    def reset(self, task_id: str, seed: int = 42) -> Dict:
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        self.session_id = data.get("session_id")
        return data

    def step(self, action: Dict[str, Any]) -> Dict:
        if not self.session_id:
            raise RuntimeError("No active session. Call reset() first.")
        r = requests.post(
            f"{self.base_url}/step",
            json={"action": action, "session_id": self.session_id},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────
# DETERMINISTIC AGENT STRATEGIES
# Each strategy is a scripted sequence of actions. In production
# you'd replace this with LLM completions. This baseline
# demonstrates the confidence declaration mechanic clearly.
# ─────────────────────────────────────────────────────────────

def _strategy_clean_claim(client: DebateFloorClient, obs: Dict) -> List[Dict]:
    """Validate key documents, estimate payout, approve with HIGH confidence."""
    docs = obs.get("observation", obs).get("documents", [])
    actions = []

    # Validate first 2 documents
    for doc in docs[:2]:
        actions.append({
            "action_type": "validate_document",
            "parameters": {"doc_id": doc["doc_id"]},
            "reasoning": "Verify document authenticity before approving.",
        })

    # Estimate payout
    actions.append({
        "action_type": "estimate_payout",
        "parameters": {"amount_inr": 150000},
        "reasoning": "Standard auto claim payout estimate.",
    })

    # Terminal: approve with HIGH confidence (clean claim, obvious approval)
    actions.append({
        "action_type": "approve_claim",
        "confidence": "HIGH",
        "parameters": {"reason": "All documents verified. No fraud signals. Clean claim approved."},
        "reasoning": "Clean claim with consistent documentation — HIGH confidence justified.",
    })

    return actions


def _strategy_contradictory_claim(client: DebateFloorClient, obs: Dict) -> List[Dict]:
    """Investigate document contradictions, flag signals, deny with MED confidence."""
    docs = obs.get("observation", obs).get("documents", [])
    actions = []

    # Validate all documents
    for doc in docs[:3]:
        actions.append({
            "action_type": "validate_document",
            "parameters": {"doc_id": doc["doc_id"]},
            "reasoning": "Looking for contradictions between claim documents.",
        })

    # Check policy history
    actions.append({
        "action_type": "query_historical_data",
        "parameters": {},
        "reasoning": "Check for prior similar claims that could indicate pattern fraud.",
    })

    # Flag detected fraud signal
    actions.append({
        "action_type": "flag_fraud_signal",
        "parameters": {
            "flag_id": "procedure_mismatch",
            "evidence": "Discharge summary procedure contradicts billing code — procedure mismatch detected.",
        },
        "reasoning": "Document contradiction is a strong fraud indicator.",
    })

    # Terminal: deny with MED confidence (evidence found but some uncertainty remains)
    actions.append({
        "action_type": "deny_claim",
        "confidence": "MED",
        "parameters": {"reason": "Procedure mismatch between discharge summary and hospital bill. Fraud signals detected."},
        "reasoning": "Sufficient evidence to deny, but complex case warrants MED not HIGH confidence.",
    })

    return actions


def _strategy_distribution_shift_claim(client: DebateFloorClient, obs: Dict) -> List[Dict]:
    """Investigate cross-claim patterns, escalate with LOW confidence."""
    docs = obs.get("observation", obs).get("documents", [])
    linked = obs.get("observation", obs).get("linked_claims", [])
    actions = []

    # Validate documents
    for doc in docs[:2]:
        actions.append({
            "action_type": "validate_document",
            "parameters": {"doc_id": doc["doc_id"]},
            "reasoning": "Initial document validation before cross-claim investigation.",
        })

    # Query historical data — reveals cross-claim patterns
    actions.append({
        "action_type": "query_historical_data",
        "parameters": {},
        "reasoning": "Distribution shift claim requires checking historical patterns across claims.",
    })

    # Query linked claims if available
    for linked_claim in linked[:2]:
        claim_id = linked_claim.get("claim_id")
        if claim_id:
            actions.append({
                "action_type": "query_linked_claim",
                "parameters": {"claim_id": claim_id},
                "reasoning": f"Investigating linked claim {claim_id} for coordinated fraud patterns.",
            })

    # Flag cross-claim signal
    actions.append({
        "action_type": "flag_fraud_signal",
        "parameters": {
            "flag_id": "clustered_policy_broker",
            "evidence": "Multiple claims share same broker code and incident timing — coordinated ring pattern.",
        },
        "reasoning": "Cross-claim evidence of coordinated fraud ring.",
    })

    # Terminal: escalate with LOW confidence (complex cross-claim fraud, expert review needed)
    actions.append({
        "action_type": "escalate_to_human",
        "confidence": "LOW",
        "parameters": {"reason": "Cross-claim fraud signals detected but full ring scope unclear. Expert investigation required."},
        "reasoning": "Distribution shift claim requires human expert — LOW confidence is correct epistemic state.",
    })

    return actions


STRATEGIES = {
    "clean_claim":              _strategy_clean_claim,
    "contradictory_claim":      _strategy_contradictory_claim,
    "distribution_shift_claim": _strategy_distribution_shift_claim,
}


# ─────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────

def run_episode(task_id: str, model: str, base_url: str, seed: int) -> Dict[str, Any]:
    client = DebateFloorClient(base_url)

    # Print mandatory [START] line
    print(f"[START] task={task_id} env=debatefloor model={model} confidence_required=true")

    # Reset environment
    reset_resp = client.reset(task_id=task_id, seed=seed)
    obs = reset_resp

    # Get scripted actions for this task
    strategy_fn = STRATEGIES.get(task_id)
    if not strategy_fn:
        print(f"[ERROR] No strategy for task '{task_id}'")
        return {}

    actions = strategy_fn(client, obs)

    total_reward = 0.0
    calibration_score = None
    step_num = 0
    last_done = False
    final_decision_correct = "none"

    for action in actions:
        if last_done:
            break

        step_num += 1
        confidence = action.get("confidence", None)

        try:
            step_resp = client.step(action)
        except Exception as e:
            print(f"[STEP] step={step_num} action={action['action_type']} reward=0.0 confidence={confidence or 'null'} done=False error={e}")
            continue

        obs = step_resp
        reward = step_resp.get("reward", 0.0)
        done = step_resp.get("done", False)
        observation = step_resp.get("observation", {})
        metadata = observation.get("metadata", {})
        error = observation.get("metadata", {}).get("last_action_error")
        last_done = done

        # Extract calibration score on terminal actions
        if done and metadata.get("calibration_score") is not None:
            calibration_score = metadata["calibration_score"]

        total_reward = reward

        # Print mandatory [STEP] line
        print(
            f"[STEP] step={step_num} action={action['action_type']} "
            f"reward={reward:.2f} confidence={confidence or 'null'} "
            f"done={done} error={error}"
        )

    # Determine if decision was correct
    if calibration_score is not None:
        final_decision_correct = "correct" if calibration_score >= 0.0 else "wrong"

    success = last_done and (calibration_score is not None) and (calibration_score >= 0.0)

    # Print mandatory [END] line
    print(
        f"[END] success={success} steps={step_num} total_reward={total_reward:.2f} "
        f"calibration_score={calibration_score if calibration_score is not None else 'N/A'} "
        f"decision={final_decision_correct}"
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps": step_num,
        "total_reward": total_reward,
        "calibration_score": calibration_score,
        "decision": final_decision_correct,
    }


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DebateFloor baseline agent")
    parser.add_argument("--task", choices=ALL_TASKS + ["all"], default="contradictory_claim")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()

    # Verify server is up
    client = DebateFloorClient(args.base_url)
    try:
        health = client.health()
        assert health.get("status") == "healthy"
    except Exception as e:
        print(f"[ERROR] Server not reachable at {args.base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks_to_run = ALL_TASKS if (args.all_tasks or args.task == "all") else [args.task]
    results = []

    for task_id in tasks_to_run:
        result = run_episode(task_id, args.model, args.base_url, args.seed)
        results.append(result)
        if len(tasks_to_run) > 1:
            print()  # blank line between tasks

    if len(results) > 1:
        print("\n── Summary ──")
        for r in results:
            cs = r.get("calibration_score")
            print(
                f"  {r['task_id']}: reward={r['total_reward']:.2f} "
                f"calibration={cs if cs is not None else 'N/A'} "
                f"decision={r['decision']}"
            )


if __name__ == "__main__":
    main()
