# Insurance Claim Triage and Fraud Detection Environment

> A production-style [OpenEnv](https://github.com/meta-pytorch/OpenEnv) benchmark for LLM agents that must adjudicate insurance claims under uncertainty, fraud pressure, and operational constraints.

[![Validate Submission](https://github.com/AniketAslaliya/insuranceClaim/actions/workflows/validate.yml/badge.svg)](https://github.com/AniketAslaliya/insuranceClaim/actions/workflows/validate.yml)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-insurance--claim--env-blue)](https://huggingface.co/spaces/AniketAsla/insurance-claim-env)

---

## Overview

Insurance claims handling is a high-stakes workflow where agents must balance customer experience, fraud risk, payout accuracy, and investigation cost simultaneously. This environment models that tension directly as a structured decision problem.

An agent receives a claim file with supporting documents, linked claims, and incident data. It must investigate, flag anomalies, estimate payouts, and reach a final adjudication decision — all within a step budget and under a reward signal that penalizes both over-caution and fraud blindness.

The environment is built on the OpenEnv REST API spec and can be run locally or accessed as a live HuggingFace Docker Space.

---

## Task Portfolio

Three tasks of increasing complexity, each with 5 seeded variants for reproducible evaluation.

### `clean_claim` — Easy

A straightforward auto insurance claim with internally consistent documentation. No fraud signals are present. The correct decision is `approve_claim` with a payout estimate in the INR 45,000–55,000 band.

The challenge is efficiency: agents that over-investigate or raise false fraud signals are penalized even when the final decision is correct.

### `contradictory_claim` — Medium

A medical claim with three embedded contradictions:
- Incident date mismatch between the FIR and hospital admission record
- Suspicious cost inflation in the repair estimate
- Signature inconsistency across documents

The agent must validate documents, flag each signal with grounded evidence text, and reach a `deny_claim` or `request_investigation` decision. Vague or keyword-free evidence reduces the evidence quality score.

### `coordinated_fraud` — Hard

A multi-claim fraud ring involving three linked policies. Signals are distributed across claims and require multi-hop reasoning:

- A shared repair shop located 340 km away from all incident sites
- Near-identical damage descriptions across independent claims
- All three policies purchased within 11 days of each other
- A shared emergency contact discoverable only after querying two or more linked claims

The final decision must be `request_investigation` targeting the full cluster. Partial escalation (missing a linked claim) is penalized.

---

## Environment Contract

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode. Accepts `task_id`, `seed`, `session_id`. Returns `session_id` and initial observation. |
| `POST` | `/step` | Submit an action. Requires `session_id` and `action` body. Returns observation, reward, done. |
| `GET`  | `/state` | Current episode state for a session. |
| `GET`  | `/tasks` | Lists all available task IDs with descriptions. |
| `GET`  | `/schema` | JSON schema for action, observation, and state types. |
| `GET`  | `/health` | Liveness check. Returns `{"status": "healthy", "active_sessions": N}`. |

### Action Space

| Action | Required Parameters | Notes |
|--------|--------------------|----|
| `validate_document` | `doc_id` | Reveals embedded signals from the document |
| `request_information` | — | First call free; subsequent calls add SLA penalty |
| `flag_fraud_signal` | `flag_id`, `evidence` | Evidence text must reference discovered keywords |
| `estimate_payout` | `amount_inr` | Must be within the task's payout band for full credit |
| `query_linked_claim` | `claim_id` | `coordinated_fraud` only — unlocks full linked claim details |
| `approve_claim` | `reason` | Final decision |
| `deny_claim` | `reason` | Final decision |
| `request_investigation` | `target_claim_ids`, `reason` | Final decision; must target the full fraud cluster |

### Observation Fields

Every step returns a structured observation including:

- `claim_id`, `task_id`, `session_id`
- `claimant` — name, contact, policy details
- `incident` — date, location, description, severity
- `documents` — list of docs with metadata (each has a `doc_id`)
- `linked_claims` — stub list; full details revealed via `query_linked_claim`
- `action_history` — all prior steps in this episode
- `available_actions` — task-appropriate action grammar
- `step_number`, `max_steps`, `done`, `status`, `message`
- `reward`, `reward_breakdown` — per-component score breakdown
- `metadata` — `variant_id`, `evidence_hits`, `exploit_penalty`, `last_action_error`

---

## Reward Model

Reward is deterministic and clamped to `[0.0, 1.0]`. It is non-zero only after the first action, giving a clean `reward=0.0` at reset.

### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `fraud_detection_score` | 0.30 | Fraction of expected signals correctly found |
| `decision_accuracy` | 0.25 | Correct final decision for the task |
| `payout_accuracy` | 0.15 | Payout estimate within the task's target band |
| `efficiency_score` | 0.10 | Inverse step usage relative to budget |
| `consistency_score` | 0.10 | Correct escalation targets in `request_investigation` |
| `evidence_quality_score` | 0.10 | Evidence text quality for flagged signals |

### Penalties

- **False flag** — flagging a signal not in the ground truth
- **Exploit penalty** — looping `request_information` >2 times; duplicate signal flags
- **Low-quality evidence** — evidence text missing discovered keywords
- **Partial cluster escalation** — `request_investigation` missing linked claims in `coordinated_fraud`
- **Wrong decision** — strong penalty for approving a fraudulent claim or denying a clean one

---

## Seeded Variants

Each task supports 5 seed-driven variants that alter numeric surfaces (costs, dates, distances, contact names) while preserving logical structure and grading rules. Use seeds for reproducible evaluation and variance analysis.

```bash
# Run evaluation across seeds
python scripts/generate_eval_report.py --base-url http://127.0.0.1:7860 --seeds 7,17,27,42,99
```

Outputs:
- `reports/eval_report.json`
- `reports/eval_report.md`

---

## Baseline Scores

Run with `python inference.py --seed 42` using Qwen2.5-72B-Instruct via HuggingFace Inference API.

| Task | Mode | Score | Steps |
|------|------|-------|-------|
| `clean_claim` | Stabilized | 0.91 | 5 |
| `contradictory_claim` | Stabilized | 0.83 | 7 |
| `coordinated_fraud` | Stabilized | 0.76 | 11 |
| `clean_claim` | LLM-only | 0.74 | 6 |
| `contradictory_claim` | LLM-only | 0.51 | 9 |
| `coordinated_fraud` | LLM-only | 0.31 | 14 |

**Stabilized** mode applies deterministic task-critical action correction on top of the LLM output. **LLM-only** mode reflects raw model capability with no oracle assistance.

Reproduce:

```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py --seed 42             # stabilized
python inference.py --seed 42 --llm-only  # raw LLM
python inference.py --task coordinated_fraud --seed 17
```

---

## Quickstart

### Run locally

```bash
git clone https://github.com/AniketAslaliya/insuranceClaim
cd insuranceClaim
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860/` for the API root, or `/tasks` to see available tasks.

### Run with Docker

```bash
docker build -t insurance-claim-env:latest .
docker run -p 7860:7860 insurance-claim-env:latest
```

### Quick episode example

```python
import requests

BASE = "http://localhost:7860"

# Start episode
r = requests.post(f"{BASE}/reset", json={"task_id": "clean_claim", "seed": 42})
session_id = r.json()["session_id"]

# Validate documents
for doc_id in ["DOC-1", "DOC-2", "DOC-3"]:
    requests.post(f"{BASE}/step", json={
        "session_id": session_id,
        "action": {"action_type": "validate_document", "parameters": {"doc_id": doc_id}, "reasoning": "check docs"}
    })

# Estimate payout and approve
requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action": {"action_type": "estimate_payout", "parameters": {"amount_inr": 50500}, "reasoning": "within band"}
})
resp = requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action": {"action_type": "approve_claim", "parameters": {"reason": "all documents consistent"}, "reasoning": "approve"}
})
print(f"Final reward: {resp.json()['reward']}")  # ~0.91
```

---

## Testing

```bash
# Unit and reward tests
PYTHONPATH=. pytest tests/envs/test_insurance_claim_reward_and_exploit.py -q

# Smoke import (step-0 reward == 0.0 for all tasks)
PYTHONPATH=. python ci/smoke_import.py

# Full clean_claim episode (reward >= 0.70)
python ci/test_clean_claim_episode.py

# Concurrent session isolation
python ci/test_concurrent_sessions.py
```

The GitHub Actions workflow runs all of these plus a Docker build and live server smoke test on every push.

---

## Architecture

```
app/
  main.py          # FastAPI server, session management (UUID-keyed, 30min TTL)
  environment.py   # InsuranceClaimEnvironment — step/reset logic, signal discovery
  tasks.py         # Task definitions, reward computation, seeded variants
  models.py        # Pydantic v2 models for Action, Observation, State

ci/                # CI test scripts (no server import needed, use HTTP)
tests/             # Pytest reward and exploit unit tests
inference.py       # Baseline LLM agent (OpenAI client, stabilized + LLM-only modes)
scripts/           # Eval report generator, HF Space evaluator
```

Session isolation: every `/reset` call returns a `session_id`. All subsequent `/step` and `/state` calls must include it. Sessions are independent in-memory objects; concurrent episodes do not share state.

---

## Known Limitations

- Synthetic data only — no PHI or proprietary claims data
- Bounded action grammar — no free-form external tool calls
- `coordinated_fraud` requires explicit `query_linked_claim` calls; signals are not visible in the initial observation

## Future Extensions

- Richer document modalities (OCR uncertainty, image evidence)
- Explicit SLA and investigator workload budgets
- Configurable fraud-prior risk profiles per variant
- Multilingual claimant narratives
