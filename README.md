# Insurance Claim Triage and Fraud Detection Environment

This repository root is now configured as a submission-ready OpenEnv environment for insurance claim adjudication.

## Overview

The environment simulates insurance claim processing where an AI agent must:

- validate documents for consistency,
- detect fraud signals,
- estimate payout bands,
- and make a final adjudication decision.

It includes three deterministic tasks with increasing difficulty, rule-based graders, and reproducible rewards in the 0.0 to 1.0 range.

## Task Suite

### 1) clean_claim (easy)

- Scenario: Straightforward auto accident claim with complete and consistent documents.
- Correct final decision: approve_claim
- Correct payout band: INR 45,000 to INR 55,000
- Fraud signals: none (false fraud flags are heavily penalized)
- Max steps: 8

### 2) contradictory_claim (medium)

- Scenario: Medical claim with planted contradictions:
  - claim incident date appears after hospital admission,
  - claimed treatment cost is 2.4x standard,
  - discharge signature mismatch.
- Correct final decision: deny_claim or request_investigation
- Max steps: 12
- Partial credit: proportional to number of true fraud signals found

### 3) coordinated_fraud (hard)

- Scenario: Three linked claims with cross-claim fraud indicators:
  - same distant repair shop,
  - shared emergency contact,
  - near-identical accident descriptions,
  - clustered recent policy purchases.
- Correct final decision: request_investigation
- Max steps: 20
- Consistency behavior: rewards coherent escalation across all linked claims; partial/inconsistent targeting is penalized.

## Observation Space

Each step returns:

- claim_id: str
- task_id: str
- claimant: dict
- incident: dict
- documents: list[dict]
- linked_claims: list[dict]
- action_history: list[dict]
- available_actions: list[str]
- step_number: int
- max_steps: int
- flags_raised: list[str]
- status: open | investigating | decided | closed
- message: str
- reward_breakdown:
  - fraud_detection_score
  - decision_accuracy
  - payout_accuracy
  - efficiency_score
  - consistency_score
  - penalty
  - total

## Action Space

The agent can send:

- validate_document
- request_information
- flag_fraud_signal
- estimate_payout
- approve_claim
- deny_claim
- request_investigation

Action payload format:

- action_type: enum
- parameters: dict
- reasoning: str

## Reward Design

Reward is deterministic and clamped to [0.0, 1.0].

Components:

- fraud_detection_score
- decision_accuracy
- payout_accuracy
- efficiency_score
- consistency_score (task 3 focus)
- penalty (false flags, invalid actions, wrong final decisions, inconsistent escalation)

Final total = weighted sum - penalties, clamped to [0.0, 1.0].

## API Endpoints

The FastAPI service exposes:

- POST /reset
- POST /step
- GET /state
- GET /tasks
- GET /health
- GET /schema
- GET /docs

## Local Setup

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Baseline Inference

`inference.py`:

- uses OpenAI client for all LLM calls,
- reads only `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`,
- emits strict stdout lines in this format:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... score=... rewards=[...]`

Run:

```bash
$env:HF_TOKEN="<your_token>"
python inference.py
```

## Docker

Build and run:

```bash
docker build -t insurance-claim-env:latest .
docker run --rm -p 7860:7860 insurance-claim-env:latest
```

## Hugging Face Spaces

1. Create a new Docker Space.
2. Push this repository content.
3. Ensure Space is configured to run `Dockerfile`.
4. Keep required variables in Space secrets:
   - HF_TOKEN
   - API_BASE_URL (optional override)
   - MODEL_NAME (optional override)
5. Confirm health endpoint:
   - `https://<your-space>.hf.space/health`

## Baseline Score Notes

Scores are reproducible for fixed model + prompt behavior because tasks and graders are fully deterministic.

## Seeded Variance and Penalty Fairness

A lightweight benchmark is included to show that reward behavior is stable across seeded variants while still penalizing exploit-like behavior.

Run:

```bash
python scripts/generate_eval_report.py --base-url http://127.0.0.1:7860 --seeds 7,17,27
```

This writes:

- `reports/eval_report.json`
- `reports/eval_report.md`

What to verify in the report:

- Seeded variance: `variant_id` changes by seed, with small numeric shifts in documents/payout bands rather than random rule changes.
- Reward stability: policy quality ranking across tasks remains consistent across seeds.
- Penalty fairness: exploit penalties rise only for repeated low-signal behaviors (for example duplicate flags or request-information loops), while grounded evidence keeps `evidence_quality_score` high.
