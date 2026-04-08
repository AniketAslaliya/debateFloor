---
title: Insurance Claim Triage Environment
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Insurance Claim Triage and Fraud Detection Environment

> A production-style [OpenEnv](https://github.com/meta-pytorch/OpenEnv) benchmark for LLM agents that must adjudicate insurance claims under uncertainty, fraud pressure, and operational constraints.

[![CI](https://github.com/Mitalimehta02/insuranceClaim/actions/workflows/validate.yml/badge.svg)](https://github.com/Mitalimehta02/insuranceClaim/actions/workflows/validate.yml)
[![Hugging Face Space](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/AniketAsla/insurance-claim-env)

---

## Overview

Insurance claims handling is a high-stakes workflow where agents must balance customer experience, fraud risk, payout accuracy, and investigation cost at the same time. This environment models that tension as a structured decision problem rather than a single-step classifier.

An agent receives a claim file with supporting documents, linked claims, and incident data. It must investigate, flag anomalies, estimate payouts, and reach a final adjudication decision — all within a step budget and under a reward signal that penalizes both over-caution and fraud blindness.

The environment follows the OpenEnv REST contract and is packaged for Docker and Hugging Face Spaces deployment.

## Live Demo

Public demo Space: [AniketAsla/insurance-claim-env](https://huggingface.co/spaces/AniketAsla/insurance-claim-env)

Judge quick-check flow:

- open the Space and confirm the environment is live
- inspect `/health`, `/tasks`, and `/schema`
- run `scripts/hf_space_eval.py --base-url https://aniketasla-insurance-claim-env.hf.space`

## Problem Statement

The benchmark asks whether an agent can behave like a reliable claim investigator instead of a one-shot classifier. To do well, an agent must:

- inspect evidence incrementally
- discover fraud signals through the allowed investigative path
- avoid unsupported guessing
- make the right final decision
- stay efficient under a soft investigation budget

This repo is designed as a realistic claim-triage benchmark rather than a toy game.

## What This Project Achieves

- 4 tasks across easy, medium, and hard difficulty
- deterministic seeded variants for reproducibility
- typed OpenEnv-style actions, observations, and state
- grounded reward shaping with calibration and exploit penalties
- root-level `inference.py` with required `[START]`, `[STEP]`, and `[END]` logs
- Docker deployment plus local pre-submission validation

---

## Task Portfolio

Four tasks of increasing complexity, each with 5 seeded variants for reproducible evaluation.

### `clean_claim` — Easy

A straightforward auto insurance claim with internally consistent documentation. No fraud signals are present. The correct decision is `approve_claim` with a payout estimate in the INR 45,000–55,000 band.

The challenge is efficiency: agents that over-investigate or raise false fraud signals are penalized even when the final decision is correct. `lookup_policy_history` returns a clean 6-year history with no prior claims.

### `contradictory_claim` — Medium

A medical claim with four discoverable signals:
- Incident date mismatch between the claim form and hospital admission record
- Suspicious cost inflation (2.4× the standard procedure rate)
- Signature inconsistency across discharge documents
- **Prior similar claim** — identical procedure 8 months ago, discoverable only via `lookup_policy_history`

The agent must validate documents, flag each signal with grounded evidence text, and reach a `deny_claim` or `request_investigation` decision.

### `coordinated_fraud` — Hard

A multi-claim fraud ring with **dynamic ring expansion**. Starts with 3 visible linked claims; after querying any 2, a hidden 4th claim (`CLM-GROUP-304`) surfaces. Five discoverable signals:

- Shared repair shop located 300–380 km from all incident sites
- Near-identical accident descriptions across independent claims (similarity 0.90–0.95)
- All policies purchased within 30 days of the incident
- Shared emergency contact — discoverable after querying 2+ linked claims
- **Clustered policy broker** (`BRK-441`) — discoverable only by querying the 4th claim

The final decision must be `request_investigation` targeting all 4 claims. The agent must decide whether to spend budget querying the 4th claim for the broker signal or escalate with 4 signals.

### `identity_fraud` — Hard

A ghost claimant scenario where the policyholder does not exist. Four discoverable signals:

- **Identity mismatch** — national registry has no record matching the claimant name and ID
- **Hospital no record** — hospital system shows admission under a different name with DOB mismatch
- **Recent policy purchase** — policy opened only 5 days before the incident (30-day exclusion window); discoverable via `lookup_policy_history`
- **DOB inconsistency** — date of birth on ID proof conflicts with policy application

Use `verify_identity` to cross-check the registry (discovers 2 signals in one call), and `lookup_policy_history` to surface the suspicious policy age. Correct decisions: `deny_claim` or `request_investigation`.

---

## Environment Contract

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start a new episode. Accepts `task_id`, `seed`, `session_id`, and legacy `episode_id`. Returns `session_id` and the initial observation. |
| `POST` | `/step` | Submit an action. Requires `session_id` and `action` body. Returns observation, reward, done. |
| `GET`  | `/state` | Current episode state for a session. |
| `GET`  | `/tasks` | Lists all available task IDs with descriptions. |
| `GET`  | `/schema` | JSON schema for action, observation, and state types. |
| `GET`  | `/health` | Liveness check. Returns `{"status": "healthy", "active_sessions": N}`. |

### Action Space

| Action | Parameters | Available In | Notes |
|--------|-----------|-------------|-------|
| `validate_document` | `doc_id` | All tasks | Reveals embedded signals from the document |
| `request_information` | — | All tasks | First call free; subsequent calls add SLA penalty |
| `lookup_policy_history` | — | All tasks | Returns prior claim history, policy age, risk score. Triggers signal discovery in `contradictory_claim` and `identity_fraud` |
| `compare_documents` | `doc_id_a`, `doc_id_b` | All tasks | Cross-document tamper detection. Discovers signals from document pairs (e.g. date mismatch, DOB inconsistency). Duplicate comparison adds exploit penalty. |
| `flag_fraud_signal` | `flag_id`, `evidence` | All tasks | Evidence text must reference discovered keywords for quality credit |
| `estimate_payout` | `amount_inr` | All tasks | Must be within the task's payout band for full credit |
| `approve_claim` | `reason`, `confidence?` | All tasks | Final decision |
| `deny_claim` | `reason`, `confidence?` | All tasks | Final decision |
| `request_investigation` | `target_claim_ids`, `reason`, `confidence?` | All tasks | Final decision; must target the full fraud cluster |
| `query_linked_claim` | `claim_id` | `coordinated_fraud` | Unlocks full linked claim details for multi-hop reasoning |
| `verify_identity` | — | `identity_fraud` | Cross-checks claimant against national registry; discovers `identity_mismatch` + `hospital_no_record` in one call |

**`confidence` field:** All final decisions accept an optional `confidence: 0.0–1.0` field. When provided, it is scored against the task's ground-truth confidence via a Brier-style calibration score (see reward model).

### Observation Fields

Every step returns a structured observation:

- `claim_id`, `task_id`, `session_id`
- `claimant` — name, contact, policy details
- `incident` — date, location, type, description
- `documents` — list of docs with `doc_id` and metadata
- `linked_claims` — stub list for `coordinated_fraud`; full details unlocked via `query_linked_claim`
- `action_history` — all prior steps in this episode
- `available_actions` — task-specific valid action types
- `step_number`, `max_steps`, `done`, `status`, `message`
- `investigation_budget`, `budget_remaining` — soft budget; going over adds 0.02 penalty per unit
- `reward`, `reward_breakdown` — per-component score breakdown at every step
- `metadata` — `variant_id`, `evidence_hits`, `exploit_penalty`, `last_action_error`, `policy_history_checked`, `identity_verified`, `agent_confidence`, `budget_remaining`, `compared_pairs`

---

## Reward Model

Reward is deterministic and clamped to `[0.0, 1.0]`. It is zero at reset and grows with each meaningful action.

### Components

| Component | Weight | Description |
|-----------|--------|-------------|
| `fraud_detection_score` | 0.28 | Fraction of expected signals correctly found |
| `decision_accuracy` | 0.20 | 1.0 if final decision matches ground truth, else 0.0 |
| `payout_accuracy` | 0.11 | Payout estimate within the task's target band |
| `efficiency_score` | 0.10 | Inverse step usage relative to budget |
| `consistency_score` | 0.09 | For `coordinated_fraud`: quality of linked-claim targeting |
| `evidence_quality_score` | 0.14 | Fraction of flagged signals backed by keyword-grounded evidence |
| `calibration_score` | 0.08 | `1 − (agent_confidence − ground_truth_confidence)²`; only scored when `confidence` is provided on final decision |

### Penalties

| Penalty | Trigger |
|---------|---------|
| False flag | Flagging a signal not in the ground truth (0.10–0.25 per flag) |
| Wrong decision | Approving fraud or denying a clean claim (0.35) |
| Exploit penalty | Looping `request_information` >2×; duplicate signal flags; duplicate `lookup_policy_history`; duplicate `compare_documents` |
| Partial cluster | `request_investigation` missing linked claims in `coordinated_fraud` (0.20) |
| Query skip | `request_investigation` without querying ≥2 linked claims (0.15) |
| Budget overage | Each action unit over `investigation_budget` adds 0.02 to penalty |

**Integrity rule:** expected fraud signals do not receive reward credit until they are actually discovered through the permitted investigative path. Unsupported guessing is penalized.

---

## Seeded Variants

Each task supports 5 seed-driven variants that alter numeric surfaces (costs, dates, distances, policy ages) while preserving logical structure and grading rules.

```bash
python scripts/generate_eval_report.py --base-url http://127.0.0.1:7860 --seeds 7,17,27,42,99
```

---

## Baseline Scores

Run with `python inference.py --seed 42` using Qwen2.5-72B-Instruct.

| Task | Mode | Score | Steps |
|------|------|-------|-------|
| `clean_claim` | Stabilized | 0.91 | 5 |
| `contradictory_claim` | Stabilized | 0.83 | 7 |
| `coordinated_fraud` | Stabilized | 0.76 | 11 |
| `identity_fraud` | Stabilized | 0.86 | 11 |
| `clean_claim` | LLM-only | 0.74 | 6 |
| `contradictory_claim` | LLM-only | 0.51 | 9 |
| `coordinated_fraud` | LLM-only | 0.31 | 14 |
| `identity_fraud` | LLM-only | 0.40 | 13 |

**Stabilized** mode: deterministic canonical action sequence, LLM called for reasoning only.  
**LLM-only** mode: raw model output with minimal repair, no oracle overrides.

## Evaluation

`inference.py` is kept compatible with the dashboard requirements:

- root-level script name: `inference.py`
- OpenAI client usage with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- required structured stdout markers: `[START]`, `[STEP]`, and `[END]`
- deterministic default mode for reproducible judging

To produce a seeded multi-task report:

```bash
python scripts/generate_eval_report.py --base-url http://127.0.0.1:7860 --seeds 7,17,27,42,99
```

The report generator preserves `session_id`, evaluates all 4 tasks, and writes JSON plus Markdown summaries under `reports/`.

```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py --seed 42                          # all 4 tasks, stabilized
python inference.py --seed 42 --llm-only               # all 4 tasks, raw LLM
python inference.py --task identity_fraud --seed 17    # single task
```

---

## Quickstart

### Run locally

```bash
git clone https://github.com/Mitalimehta02/insuranceClaim.git
cd insuranceClaim
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t insurance-claim-env:latest .
docker run -p 7860:7860 insurance-claim-env:latest
```

### Episode example — `identity_fraud`

```python
import requests

BASE = "http://localhost:7860"

r = requests.post(f"{BASE}/reset", json={"task_id": "identity_fraud", "seed": 0})
session_id = r.json()["session_id"]

def step(action):
    return requests.post(f"{BASE}/step", json={"action": action, "session_id": session_id}).json()

# Cross-check registry — discovers identity_mismatch + hospital_no_record
step({"action_type": "verify_identity", "parameters": {}, "reasoning": "check registry"})

# Reveal policy age — discovers recent_policy_purchase
step({"action_type": "lookup_policy_history", "parameters": {}, "reasoning": "check policy age"})

# Validate ID document — discovers dob_inconsistency
step({"action_type": "validate_document", "parameters": {"doc_id": "DOC-34"}, "reasoning": "check id"})

# Flag all signals with grounded evidence
for flag_id, evidence in [
    ("identity_mismatch",    "registry has no record matching id 7821 identity mismatch"),
    ("hospital_no_record",   "hospital record shows patient not found name mismatch dob"),
    ("recent_policy_purchase", "policy purchased 5 days before incident 30-day exclusion window"),
    ("dob_inconsistency",    "dob on id 1988 does not match policy application 1986 inconsistency"),
]:
    step({"action_type": "flag_fraud_signal", "parameters": {"flag_id": flag_id, "evidence": evidence}, "reasoning": "flag"})

# Final decision with confidence
resp = step({"action_type": "deny_claim", "parameters": {"reason": "ghost claimant confirmed"},
             "reasoning": "deny", "confidence": 0.9})
print(f"Final reward: {resp['reward']}")         # ~0.86
print(f"Calibration:  {resp['reward_breakdown']['calibration_score']}")
```

---

## Testing

```bash
# 15 unit tests covering reward, exploits, new features
PYTHONPATH=. pytest tests/envs/test_insurance_claim_reward_and_exploit.py -q

# Step-0 reward == 0.0 for all 4 tasks
PYTHONPATH=. python ci/smoke_import.py

# Full clean_claim episode via HTTP (reward >= 0.70)
python ci/test_clean_claim_episode.py

# Concurrent session isolation
python ci/test_concurrent_sessions.py
```

CI runs on every push: unit tests and Docker build in parallel, then live server smoke test.

---

## Architecture

```
app/
  main.py          # FastAPI server — session management (UUID-keyed, 30-min TTL auto-cleanup)
  environment.py   # InsuranceClaimEnvironment — step/reset/signal discovery/calibration
  tasks.py         # 4 task definitions, reward computation, seeded variants, policy history
  models.py        # Pydantic v2 — Action (with confidence), Observation, State, RewardBreakdown

ci/                # HTTP-based CI test scripts
tests/             # Pytest unit tests (reward, exploits, calibration, new features)
inference.py       # Baseline LLM agent — 4 tasks, stabilized + LLM-only modes
.github/workflows/
  validate.yml     # Single CI orchestrator (unit-tests ∥ docker-build → server-smoke)
```

**Session isolation:** every `/reset` returns a `session_id`. All `/step` and `/state` calls include it. Sessions are independent in-memory objects with no shared state.

## Dashboard Compliance Checklist

This repository is aligned to the Scaler Round-1 checklist:

- real-world task, not a toy game
- full OpenEnv interface with typed models
- at least 3 tasks across difficulty levels
- deterministic reward in `[0.0, 1.0]`
- reproducible root-level `inference.py`
- required `[START]`, `[STEP]`, and `[END]` stdout logs
- `openenv.yaml`
- Dockerfile for deployment
- README with setup, environment description, action space, observation space, and evaluation notes
- local validator in `ci/validate_submission.py`

## Validation

```bash
PYTHONPATH=. python ci/validate_submission.py
PYTHONPATH=. python ci/smoke_import.py
PYTHONPATH=. pytest tests/envs/test_insurance_claim_reward_and_exploit.py -q
```

---

## Known Limitations

- Synthetic data only — no PHI or proprietary claims data
- Bounded action grammar — no free-form external tool calls
- `coordinated_fraud` signals are gated behind `query_linked_claim`; not visible in initial observation
- `identity_fraud` registry is deterministic (not a live external lookup)

## Future Extensions

- Richer document modalities (OCR uncertainty, image tampering detection)
- Explicit SLA and investigator workload budgets
- Configurable fraud-prior risk profiles per variant
- Multilingual claimant narratives

## Differentiation

This environment is intentionally stronger than a basic fraud benchmark because it combines:

- dynamic ring expansion in `coordinated_fraud`
- cross-document and cross-claim reasoning
- grounded discovery before reward credit
- budget-aware investigation
- confidence calibration as part of scoring
