---
title: DebateFloor — Insurance Calibration RL Environment
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
---

# DebateFloor — Insurance Calibration RL Environment

> An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant RL training environment where AI agents investigate insurance claims, debate adversarially, and must declare **calibrated confidence** before every terminal decision. Built for the **Meta PyTorch × Scaler Hackathon Grand Finale, April 25–26 2026**.

## Theme Coverage

| Theme | Bonus Prize | What We Built |
|-------|-------------|---------------|
| **Theme 3.1** — World Modeling (Professional) | **Scaler AI Labs**: Multi-App RL for Enterprise Workflows | 5 fraud types, multi-doc investigation, IRDAI registry, policy history |
| **Theme 1** — Multi-Agent Interactions | **Fleet AI**: Scalable Oversight | 3-agent Debate Panel: Prosecutor + Defender + Judge |
| **Theme 4** — Self-Improvement | Curriculum / difficulty escalation | easy→medium→hard + anti-gaming detector |

---

[![Tests](https://github.com/AniketAslaliya/debateFloor/actions/workflows/validate.yml/badge.svg)](https://github.com/AniketAslaliya/debateFloor/actions/workflows/validate.yml)
[![HF Space](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-orange)](https://huggingface.co/spaces/AniketAsla/debatefloor)
[![arXiv](https://img.shields.io/badge/Based%20on-CoCA%20arXiv%3A2603.05881-red)](https://arxiv.org/abs/2603.05881)

---

## Submission Links

- **Live OpenEnv Space:** https://huggingface.co/spaces/AniketAsla/debatefloor
- **Visual demo:** https://aniketasla-debatefloor.hf.space/ui
- **Mini-blog draft (repo):** [docs/HFBlogPost.md](docs/HFBlogPost.md)
- **Short presentation deck:** [docs/PITCH_DECK.md](docs/PITCH_DECK.md)
- **Training plot (loss + reward):** [docs/reward_curve.svg](docs/reward_curve.svg)
- **Component shift plot:** [docs/component_shift.svg](docs/component_shift.svg)
- **Training run logs:** [reports/training_summary.json](reports/training_summary.json)
- **Component shift summary:** [reports/component_shift_summary.json](reports/component_shift_summary.json)
- **HTTP rollout report:** [reports/http_rollout_eval.md](reports/http_rollout_eval.md)

---

## Quick Tour For Reviewers

If you only have 3 to 5 minutes, this is the shortest path:

1. Open the live demo: https://aniketasla-debatefloor.hf.space/ui
2. Run `contradictory_claim` first. It shows the full investigation flow, the debate panel, and a calibrated terminal decision.
3. Then open [docs/component_shift.svg](docs/component_shift.svg) and [docs/reward_curve.svg](docs/reward_curve.svg) to see what changed after training.
4. If you want the full picture, skim the sections below in this order: What is DebateFloor, Training Signals, Why This Matters.

---

## What is DebateFloor?

Standard RL environments reward **what** an agent decides. DebateFloor rewards **how confidently** it decides — and whether that confidence was warranted.

The environment also exposes a composable rubric hierarchy, so the final score is not a black box: training and evaluation can inspect `rubric_reward` and `rubric_components` alongside the terminal reward.

Before every terminal action (`approve_claim`, `deny_claim`, `escalate_to_human`), the agent must declare a confidence level: **HIGH**, **MED**, or **LOW**. The reward is then determined by a 3×2 calibration matrix:

| Confidence | Correct Decision | Wrong Decision |
|------------|-----------------|----------------|
| **HIGH**   | +1.0            | **−0.8** (worst outcome) |
| **MED**    | +0.6            | −0.2           |
| **LOW**    | +0.1            | 0.0            |

An agent that always says HIGH to maximise reward will be catastrophically punished when wrong. An agent that always says LOW is caught by the anti-gaming system. **The only winning strategy is accurate calibration.**

Based on the [CoCA framework (arXiv:2603.05881)](https://arxiv.org/abs/2603.05881) — co-optimising confidence and accuracy via GRPO.

---

## 3-Agent Debate Panel (Theme 1 + Fleet AI Bonus)

Before making a terminal decision, the investigator calls `convene_debate_panel`, triggering adversarial reasoning from two independent roles:

```
AGENT 1: INVESTIGATOR
├── validate_document      → discovers fraud signals
├── flag_fraud_signal      → formally raises grounded signal
├── query_historical_data  → reveals cross-claim patterns
└── Builds evidence base over N steps
                ↓
ACTION: convene_debate_panel  (costs 2 budget units)
                ↓
┌─────────────────────┐    ┌─────────────────────────┐
│  AGENT 2: PROSECUTOR│    │  AGENT 3: DEFENDER       │
│  Built from:        │    │  Built from:             │
│  • found_signals    │    │  • doc consistency       │
│  • discovered sigs  │    │  • policy history        │
│  Strength: STRONG / │    │  Strength: STRONG /      │
│  MODERATE / WEAK    │    │  MODERATE / WEAK         │
└─────────────────────┘    └─────────────────────────┘
                ↓
PANEL VERDICT: recommendation (prosecution / defense / split)
                ↓
JUDGE (investigator, informed by transcript):
→ approve_claim / deny_claim / escalate_to_human
+ confidence: HIGH / MED / LOW → calibration_score via 3×2 matrix
```

Debate transcript in `observation.debate_transcript`:

```json
{
  "prosecutor_argument": "PROSECUTOR: 2 fraud signals found: date_mismatch, cost_inflation...",
  "prosecutor_strength": "STRONG",
  "defender_argument": "DEFENDER: Documents are internally consistent...",
  "defender_strength": "WEAK",
  "panel_verdict": "Prosecution substantially outweighs defense. Recommend denial.",
  "panel_lean": "prosecution",
  "signals_at_debate": ["date_mismatch", "cost_inflation"],
  "step_convened": 6
}
```

**Fleet AI Scalable Oversight:** The Judge reads adversarial arguments from two independent reasoning contexts before deciding — oversight agents explaining each other's behavior to a third decision-maker.

---

## Why This Matters

Insurance fraud costs India ₹30,000+ crore annually (IRDAI 2023). Current LLMs are overconfident — they hallucinate approvals or denials without epistemic grounding. DebateFloor trains models to know when they don't know, making them safer for high-stakes decisions.

The CAPO paper (April 2026) shows GRPO training induces systematic overconfidence. DebateFloor is the direct fix: a reward surface that penalises overconfidence harder than wrong answers.

---

## Repository Structure

```
debatefloor/
├── README.md                       ← you are here
├── CLAUDE.md                       ← architecture reference for Claude Code
├── IMPLEMENTATION_LOG.md           ← full build log + pitch Q&A
├── openenv.yaml                    ← OpenEnv spec manifest
├── Dockerfile                      ← HF Space deployment
├── requirements.txt
├── pyproject.toml
│
├── inference_debatefloor.py        ← baseline agent (mandatory deliverable)
├── inference.py                    ← Round 1 baseline (kept for reference)
│
├── app/                            ← FastAPI server (OpenEnv contract)
│   ├── main.py                     ← endpoints: /reset /step /state /tasks /health /schema
│   ├── environment.py              ← InsuranceClaimEnvironment + calibration wiring
│   ├── models.py                   ← Pydantic models (confidence: HIGH|MED|LOW)
│   └── tasks.py                    ← task definitions + reward computation
│
├── server/                         ← DebateFloor core modules (new)
│   ├── calibration_grader.py       ← 3×2 matrix + anti-gaming + training/eval reward
│   └── claim_generator.py          ← procedural episode generator (500+ episodes)
│
├── train/
│   ├── train_minimal.py            ← Pure TRL, Qwen2.5-0.5B, T4 in 15 min (USE THIS)
│   └── train_debatefloor.ipynb     ← GRPO training notebook (Unsloth variant)
│
├── tests/
│   ├── test_calibration.py         ← 13 tests (calibration grader)
│   └── test_generator.py           ← 32 tests (claim generator, 500-episode uniqueness)
│
└── docs/
    ├── CONTEXT.md                  ← session-by-session build log
    ├── roadmap.md                  ← scored checklist
    ├── HFBlogPost.md               ← HF blog draft
    └── guide.md                    ← architecture guide
```

---

## The 3 Tasks

| Task | Difficulty | Max Steps | Correct Decision | Expected Confidence |
|------|-----------|-----------|-----------------|-------------------|
| `clean_claim` | Easy | 10 | `approve_claim` | HIGH |
| `contradictory_claim` | Medium | 18 | `deny_claim` | MED |
| `distribution_shift_claim` | Hard | 28 | `escalate_to_human` | LOW |

### Task 3 — The Demo Centrepiece
`distribution_shift_claim` looks clean on the surface. The agent must call `query_linked_claim` or `query_historical_data` to discover cross-claim fraud signals. If the agent declares HIGH confidence, it is **always penalised regardless of decision** — this task is designed to require epistemic humility.

---

## Procedural Generation — What Makes This a Training Environment

A benchmark has fixed episodes. DebateFloor generates them procedurally:

```python
from server.claim_generator import generate_claim

# Same inputs → same episode (deterministic)
episode = generate_claim(seed=42, fraud_type="medical_inflation",
                         coverage_type="health", difficulty="medium")

# Different seeds → different claimants, amounts, dates, signal strengths
episode_2 = generate_claim(seed=43, ...)
```

**5 fraud types × 4 coverage types × 3 jurisdictions × seed variation = 500+ unique training episodes**

| Fraud Type | Ground Truth | Key Signal |
|-----------|-------------|-----------|
| `staged_accident` | `deny_claim` | Cost mismatch between damage and repair estimate |
| `medical_inflation` | `deny_claim` | Procedure in bill ≠ procedure in discharge summary |
| `identity_fraud` | `deny_claim` | Ghost claimant, policy opened 5 days before incident |
| `coordinated_ring` | `escalate_to_human` | Shared broker across 3–5 simultaneous claims |
| `phantom_provider` | `deny_claim` | Hospital not in IRDAI registry, invalid GST |

---

## Quickstart

### Run locally

```bash
git clone https://github.com/AniketAslaliya/debateFloor.git
cd debateFloor
pip install -r requirements.txt
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Run with Docker

```bash
docker build -t debatefloor .
docker run -p 7860:7860 debatefloor
```

### Run tests

```bash
# All DebateFloor tests (45 total)
PYTHONPATH=. pytest tests/test_calibration.py tests/test_generator.py -v

# Calibration grader only (13 tests)
PYTHONPATH=. pytest tests/test_calibration.py -v

# Generator only — includes 500-episode uniqueness check (32 tests)
PYTHONPATH=. pytest tests/test_generator.py -v
```

### Run the baseline agent

```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

# Run all 3 tasks with confidence declarations
python inference_debatefloor.py --task contradictory_claim --model gpt-4o
```

### Validate HTTP rollouts

This script drives the hosted environment through `/reset` and `/step`, then
compares a naive overconfident baseline with the calibrated scripted policy.

```bash
python scripts/evaluate_http_rollouts.py --base-url https://aniketasla-debatefloor.hf.space
```

It writes:

- `reports/http_rollout_eval.json`
- `reports/http_rollout_eval.md`

---

## API Reference

All endpoints follow the OpenEnv REST contract:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode. Accepts `task_id`, `seed`, `session_id`. |
| `POST` | `/step` | Submit action. Requires `session_id` and `action` body. |
| `GET`  | `/state` | Current episode state. |
| `GET`  | `/tasks` | Lists all tasks with objectives. |
| `GET`  | `/schema` | JSON schema for action/observation/state. |
| `GET`  | `/health` | Returns `{"status": "healthy", "active_sessions": N}`. |

### Action Space

```python
# Non-terminal (confidence optional)
"validate_document"            # reveals embedded fraud signals
"flag_fraud_signal"            # flag_id + evidence (must cite discovered signal)
"request_information"
"lookup_policy_history"
"compare_documents"
"estimate_payout"              # amount_inr
"query_historical_data"        # cross-claim patterns (key for Task 3)
"query_linked_claim"           # coordinated_ring / distribution_shift only
"verify_identity"              # identity_fraud only
"verify_provider_registration" # distribution_shift only
"convene_debate_panel"         # MULTI-AGENT: triggers Prosecutor + Defender transcript

# Terminal — confidence REQUIRED: "HIGH" | "MED" | "LOW"
"approve_claim"
"deny_claim"
"escalate_to_human"
```

### Example Episode

```python
import requests

BASE = "http://localhost:7860"

# Start episode
r = requests.post(f"{BASE}/reset", json={"task_id": "contradictory_claim", "seed": 42})
session_id = r.json()["session_id"]

def step(action):
    return requests.post(f"{BASE}/step", json={"action": action, "session_id": session_id}).json()

# Investigate
step({"action_type": "validate_document", "parameters": {"doc_id": "DOC-001"}, "reasoning": "check bill"})
step({"action_type": "flag_fraud_signal", "parameters": {"flag_id": "procedure_mismatch", "evidence": "discharge says appendectomy, bill says cardiac bypass"}, "reasoning": "billing fraud"})

# Terminal decision WITH confidence (required)
resp = step({
    "action_type": "deny_claim",
    "confidence": "MED",          # calibrated uncertainty declaration
    "reason": "procedure mismatch confirmed",
    "reasoning": "bill contradicts discharge summary"
})
print(f"Reward: {resp['reward']}")
print(f"Calibration score: {resp['observation']['reward_breakdown']['calibration_score']}")
```

---

## Reward Design

### Training Reward (simple — use for GRPO)

```python
def training_reward(step):
    r = -0.05                          # step penalty (efficiency)
    if step.done:
        r += 1.0 if correct else -0.5  # decision accuracy
        r += 0.3 * min(legit_flags, 3) # fraud signal detection
        r += 0.5 * calibration_matrix[(confidence, correct)]
    return r
```

### Evaluation Reward (complex — for demo and reporting only)

```python
def eval_reward(episode):
    return (0.35 * calibration_r      # confidence accuracy
          + 0.25 * escalation_r       # appropriate uncertainty escalation
          + 0.20 * evidence_quality_r  # grounded signal citations
          + 0.10 * efficiency_r        # step efficiency
          - 0.10 * gaming_penalty)     # anti-gaming deduction
```

**Never mix these.** Compound rewards cause gradient instability in GRPO. Training reward = stable learning signal. Eval reward = impressive demo metrics.

### Anti-Gaming System

The agent cannot game calibration by always declaring LOW confidence:

```
if LOW_rate > 70% across 10+ episodes:
    penalty = (rate - 0.70) × 2.0

if HIGH_rate > 80% across 10+ episodes:
    penalty = (rate - 0.80) × 1.5
```

---

## Training Results

Training via `train/train_minimal.py` — Qwen2.5-0.5B, TRL GRPOTrainer, T4 GPU, ~15 min:

### GRPO Training Signals

Training via `train/train_minimal.py` now saves both the reward curve and the component-shift eval summary.

![WandB Reward Curve](docs/reward_curve.png)

The saved curve includes both training loss and mean reward from a real run.

### Component Score Shift

![Component score shift before vs after training](docs/component_shift.png)

This plot compares the held-out validation sweep before and after training across fraud detection, decision accuracy, evidence grounding, and calibration.

### Confidence Distribution — Before vs After GRPO

| Confidence | Before Training | After Training |
|---|---|---|
| HIGH | ~82% | ~44% |
| MED | ~12% | ~36% |
| LOW | ~6% | ~20% |

The model learns to reserve HIGH confidence for easy cases (`clean_claim`) and express genuine uncertainty on hard cases (`distribution_shift_claim`) — without being told which task is which. This is the CoCA calibration improvement signal.

The reward curve is modest in absolute value — the real signal is **the confidence distribution shift**. The model learns WHEN to be confident, not just what to say.

---

## GRPO Training

Install training-only dependencies in Colab:

```bash
pip install -r train/requirements.txt
```

The training notebook (`train/train_debatefloor.ipynb`) uses:
- **Model:** `unsloth/Qwen2.5-1.5B-Instruct` (free Colab T4 compatible)
- **Trainer:** TRL `GRPOTrainer` with custom `env_reward_fn`
- **Dataset:** `generate_episode_pool(200)` — procedurally generated, never repeats
- **Logging:** WandB for public training signals

```python
from trl import GRPOTrainer, GRPOConfig
from server.calibration_grader import training_reward  # simple scalar only

def env_reward_fn(completions, **kwargs):
    rewards = []
    for completion in completions:
        action = parse_action(completion)
        step_result = env.step(action)
        rewards.append(training_reward(step_result))  # NOT eval_reward
    return rewards
```

The minimal TRL script (`train/train_minimal.py`) saves local artifacts after
training so the submission does not depend only on notebook output:

- `docs/reward_curve.png`
- `docs/component_shift.png`
- `reports/training_summary.json`
- `reports/component_shift_summary.json`

---

## Concurrent Sessions

DebateFloor supports 64 concurrent sessions — required for GRPO parallel rollouts:

```python
import concurrent.futures, requests

BASE = "http://localhost:7860"

def run_episode(seed):
    r = requests.post(f"{BASE}/reset", json={"task_id": "contradictory_claim", "seed": seed})
    return r.json()["session_id"]

# 4 parallel resets — all return independent sessions
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    sessions = list(ex.map(run_episode, [1, 2, 3, 4]))

assert len(set(sessions)) == 4  # all unique
```

---

## OpenEnv Spec Compliance

| Requirement | Status |
|-------------|--------|
| `spec_version: 1` | ✅ |
| OpenEnv `Environment` base class | ✅ |
| OpenEnv `Action` / `Observation` / `State` base types | ✅ |
| `/reset`, `/step`, `/state`, `/tasks`, `/health`, `/schema` | ✅ |
| `supports_concurrent_sessions: true` | ✅ |
| `max_concurrent_envs: 64` | ✅ |
| `confidence_required: true` | ✅ |
| `procedural_generation: true` | ✅ |
| `episode_pool_size: 500` | ✅ |
| Reward in `[0.0, 1.0]` | ✅ |
| Reproducible `inference_debatefloor.py` | ✅ |
| `[START]` / `[STEP]` / `[END]` stdout format | ✅ |
| Docker deployment | ✅ |
| CoCA citation | ✅ |

The server keeps a strict client/server split: external clients and evaluation
scripts interact through HTTP (`/reset`, `/step`, `/state`) and do not import
environment internals. No MCP tools are exposed with reserved names such as
`reset`, `step`, `state`, or `close`.

---

## Team

**Aniket Aslaliya** — environment core, claim generator, calibration grader
**Mitali Mehta** — domain knowledge (fraud types, IRDAI regulations), grader design
**Aditya Sharma** — training pipeline, GRPO notebook, WandB integration

---

## Citation

```bibtex
@article{coca2025,
  title={Co-optimizing Confidence and Accuracy via Segment-Specific GRPO Rewards},
  author={...},
  journal={arXiv:2603.05881},
  year={2025}
}
```

**Related:**
- CAPO paper (April 2026) — GRPO induces overconfidence; DebateFloor is the fix
- OpenEnv: [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- TRL GRPOTrainer: [huggingface.co/docs/trl/grpo_trainer](https://huggingface.co/docs/trl/grpo_trainer)
- Unsloth: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
