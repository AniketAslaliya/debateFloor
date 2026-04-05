# Insurance Claim Triage and Fraud Detection Environment

A production-style OpenEnv benchmark for claim adjudication under uncertainty, fraud pressure, and operational constraints.

## Why This Environment Matters

Insurance claims handling is a real, high-impact workflow where agents must balance customer experience, fraud risk, payout accuracy, and investigation cost. This environment models that tension directly and evaluates whether an agent can:

- verify evidence reliability,
- detect fraud signals without over-flagging,
- make calibrated payout decisions,
- and escalate consistently for linked-risk scenarios.

## Benchmark Goals

This environment is designed for:

- policy evaluation for tool-using LLM agents,
- reward-shaping research for fraud-sensitive decision systems,
- robustness testing under seeded scenario variation,
- exploit-resistance analysis (reward hacking and low-signal loops).

## Task Portfolio

The benchmark includes three deterministic tasks with increasing complexity.

### 1. clean_claim (easy)

- Context: clean auto claim with consistent documentation.
- Expected final decision: approve_claim.
- Payout target band: INR 45,000 to INR 55,000.
- Core challenge: avoid false fraud escalation while keeping cycle-time efficient.

### 2. contradictory_claim (medium)

- Context: medical claim with embedded contradictions.
- Contradictions include incident-date mismatch, suspicious cost inflation, and signature inconsistency.
- Expected final decision: deny_claim or request_investigation.
- Core challenge: detect and justify true signals with grounded evidence.

### 3. coordinated_fraud (hard)

- Context: linked multi-claim ring with cross-claim patterns.
- Indicators include distant shared repair shop, shared emergency contact, near-identical narratives, and clustered recent policy purchases.
- Expected final decision: request_investigation.
- Core challenge: maintain consistency across linked claims and escalate the full cluster.

## Environment Contract

### Action Space

- validate_document
- request_information
- flag_fraud_signal
- estimate_payout
- approve_claim
- deny_claim
- request_investigation
- query_linked_claim (coordinated_fraud only: reveals full linked claim details for multi-hop reasoning)

### Observation Payload

Each step returns structured fields:

- claim_id
- task_id
- claimant
- incident
- documents
- linked_claims
- action_history
- available_actions
- step_number
- max_steps
- flags_raised
- status
- message
- reward_breakdown
- metadata (includes variant_id, evidence counters, exploit penalty, and last_action_error)

### API Endpoints

- POST /reset
- POST /step
- GET /state
- GET /tasks
- GET /schema
- GET /health
- GET /

## Reward Model

Reward is deterministic and clamped to [0.0, 1.0].

Final reward combines:

- fraud_detection_score
- decision_accuracy
- payout_accuracy
- efficiency_score
- consistency_score
- evidence_quality_score

Then subtracts cumulative penalties:

- false-flag penalties,
- wrong-decision penalty,
- partial-consistency penalty,
- exploit_penalty,
- action-error and workflow penalties.

### Fairness and Anti-Gaming Design

To reduce reward hacking:

- repeated request_information loops increase exploit penalty,
- duplicate signal-flagging increases exploit penalty,
- low-quality evidence text reduces evidence quality credit,
- incorrect final decisions receive strong penalties,
- partial cluster escalation in coordinated-fraud is penalized.

This creates a measurable trade-off between aggressive fraud hunting and precision.

## Seeded Variants and Robustness

Each task supports seed-driven variants that alter numeric surfaces (costs, dates, distances, timelines) while preserving logical structure and grading rules. This allows:

- reproducible evaluation,
- variance analysis across controlled perturbations,
- robustness testing without changing task intent.

Generate seeded evaluation snapshot:

```bash
python scripts/generate_eval_report.py --base-url http://127.0.0.1:7860 --seeds 7,17,27
```

Outputs:

- reports/eval_report.json
- reports/eval_report.md

## Baseline Scores

Run with: `python inference.py --seed 42`

| Task | Mode | Score | Steps | Model |
|------|------|-------|-------|-------|
| clean_claim | Stabilized | 0.91 | 5 | Qwen2.5-72B-Instruct |
| contradictory_claim | Stabilized | 0.83 | 7 | Qwen2.5-72B-Instruct |
| coordinated_fraud | Stabilized | 0.76 | 11 | Qwen2.5-72B-Instruct |
| clean_claim | LLM-only | 0.74 | 6 | Qwen2.5-72B-Instruct |
| contradictory_claim | LLM-only | 0.51 | 9 | Qwen2.5-72B-Instruct |
| coordinated_fraud | LLM-only | 0.31 | 14 | Qwen2.5-72B-Instruct |

Reproduce with:

```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py --seed 42              # stabilized mode
python inference.py --seed 42 --llm-only  # raw LLM mode
```

Note: Scores above are from actual runs. LLM-only scores reflect genuine model capability
on this environment without oracle assistance.

## Inference Script

inference.py uses OpenAI Client and required environment variables:

- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

The baseline operates in two modes (see `--llm-only` flag above). It calls the LLM every
step and optionally applies deterministic task-critical stabilization.

Stdout follows strict evaluator format:

- [START] task=... env=... model=...
- [STEP] step=... action=... reward=... done=... error=...
- [END] success=... steps=... rewards=...

## Local Development

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Validation and Dry Run Commands

Spec validation:

```bash
openenv validate .
```

Docker build:

```bash
docker build -t insurance-claim-env:latest .
```

Live Space evaluation:

```bash
python scripts/hf_space_eval.py --base-url https://aniketasla-insurance-claim-env.hf.space
```

Deterministic reward/exploit tests:

```bash
python -m pytest tests/envs/test_insurance_claim_reward_and_exploit.py -q
```

## Hugging Face Space Deployment

This project is deployed as a Docker Space and should expose port 7860.

Runtime expectations:

- health endpoint returns status healthy,
- tasks endpoint enumerates task set,
- reset and step complete episodes with bounded rewards.

## Practical Use Cases

- evaluating claim-triage copilots before enterprise integration,
- testing fraud-focused planning policies,
- comparing model families under decision-risk constraints,
- regression testing reward-shaping changes.

## Known Limitations

- synthetic data only; no PHI or proprietary claims data.
- bounded action grammar; no free-form external tool calls.
- concurrent sessions supported via session_id; sessions auto-expire after 30 minutes.

## Future Extensions

- richer document modalities (OCR uncertainty, image evidence),
- explicit SLA and investigator workload budgets,
- configurable fraud-prior risk profiles,
- multilingual claimant narratives.
