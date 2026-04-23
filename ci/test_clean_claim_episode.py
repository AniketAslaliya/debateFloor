"""Run a full clean_claim episode and assert final reward >= 0.70."""
import requests

BASE = "http://localhost:7860"
r = requests.post(f"{BASE}/reset", json={"task_id": "clean_claim", "seed": 42})
r.raise_for_status()
session_id = r.json()["session_id"]

actions = [
    {"action_type": "validate_document", "parameters": {"doc_id": "DOC-1"}, "reasoning": "check"},
    {"action_type": "validate_document", "parameters": {"doc_id": "DOC-2"}, "reasoning": "check"},
    {"action_type": "validate_document", "parameters": {"doc_id": "DOC-3"}, "reasoning": "check"},
    {"action_type": "estimate_payout", "parameters": {"amount_inr": 50500}, "reasoning": "estimate"},
    {
        "action_type": "approve_claim",
        "confidence": "HIGH",
        "parameters": {"reason": "all consistent"},
        "reasoning": "approve",
    },
]

reward = 0.0
for a in actions:
    resp = requests.post(f"{BASE}/step", json={"action": a, "session_id": session_id})
    resp.raise_for_status()
    reward = resp.json()["reward"]
    done = resp.json()["done"]
    print(f"  {a['action_type']}  reward={reward:.3f}  done={done}")

assert reward >= 0.70, f"Final reward {reward} < 0.70"
print(f"clean_claim final reward: {reward}  PASS")
