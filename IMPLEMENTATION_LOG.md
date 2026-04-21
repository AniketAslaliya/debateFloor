# DebateFloor — Implementation Log
## Meta PyTorch × Scaler Hackathon Grand Finale | April 25–26, 2026

> **Purpose:** This document is the single source of truth for what was built, why decisions were made, what failed, and what succeeded. Use it verbatim during the pitch Q&A.

---

## Why "DebateFloor"?

Standard RL environments reward *what* an agent decides. DebateFloor rewards *how confidently* it decides — and whether that confidence was warranted. Like a debate floor, you cannot just state a position. You must declare the strength of your conviction (HIGH / MED / LOW) before every terminal action, and you are scored on whether that declaration matched reality. A confident wrong answer is punished harder than a cautious wrong answer. The name captures the epistemic accountability baked into every reward signal.

---

## System Architecture

```
Round 1 (inherited)          DebateFloor additions (new)
─────────────────────        ──────────────────────────────
FastAPI insurance env   →    + Procedural Claim Generator   (Layer 1)
InsuranceClaimAction    →    + confidence: HIGH|MED|LOW     (Layer 2)
Simple reward           →    + Calibration Grader 3×2       (Layer 3)
                        →    + Split training/eval reward    (Layer 4)
                        →    + GRPO Colab notebook           (Layer 5)
                        →    + HF blog + web demo            (Layer 6)
```

Based on CoCA paper: arXiv:2603.05881 — "Co-optimising Confidence and Accuracy via Segment-Specific GRPO Rewards"

---

## Session 0 — April 19, 2026 (Project Init)

### What happened
- Locked problem statement: Theme 3.1, insurance calibration domain
- Decided DebateFloor = Round 1 environment + calibration grader + procedural generator
- Created all planning documents (CLAUDE.md, CONTEXT.md, SKILL.md, ROADMAP.md)
- Assigned team roles: Aniket (env core), Mitali (domain + grader), Aditya (training pipeline)

### Key decisions
- **Keep insurance domain** — switching domains with 5 days left is high risk
- **Split training vs evaluation reward** — compound rewards cause GRPO gradient instability
- **Confidence as Literal enum not float** — avoids Brier score complexity, cleaner matrix lookup

### Files created
| File | Purpose |
|------|---------|
| CLAUDE.md | Claude Code context — architecture, constraints, never-do list |
| context.md | Session log |
| skill.md | Token-efficient patterns |
| roadmap.md | Scored checklist |

### Estimated score: 38/100

---

## Session 1 — April 21, 2026

### What was built

#### Phase 0 — Repo Structure
Created missing folders and placeholder files so the full intended tree exists:
- `server/claim_generator.py` (placeholder)
- `server/calibration_grader.py` (full implementation)
- `inference_debatefloor.py` (placeholder)
- `train/train_debatefloor.ipynb` (placeholder notebook)
- `docs/CONTEXT.md` (session log, moved from root)
- `tests/test_calibration.py` (13 tests)

#### Phase 1A — models.py update
Changed `InsuranceClaimAction.confidence` from `Optional[float]` (Brier-score style) to `Optional[Literal["HIGH", "MED", "LOW"]]` (matrix lookup style).

Added `model_post_init` validator: terminal actions (`approve_claim`, `deny_claim`, `escalate_to_human`) raise `ValueError` if `confidence is None`. This enforces calibration at the data model level — the environment cannot accidentally accept an uncalibrated terminal decision.

Added `confidence_required: bool = True` to `InsuranceClaimObservation` so the agent always knows calibration is required.

Changed `calibration_score` in `InsuranceClaimReward` from `float` (always present, Brier) to `Optional[float]` (None until terminal action, then matrix value).

#### Phase 1B — server/calibration_grader.py
The core innovation. Full implementation:

```python
CALIBRATION_MATRIX = {
    ("HIGH", True):   1.0,   # confident + right = best
    ("HIGH", False): -0.8,   # confident + wrong = WORST — key design choice
    ("MED",  True):   0.6,
    ("MED",  False): -0.2,
    ("LOW",  True):   0.1,
    ("LOW",  False):  0.0,   # at least it knew it didn't know
}
```

**Why HIGH+WRONG = -0.8 and not -1.0?** Leaves room for gaming penalty to push below without hitting the clamp wall. Deliberate.

**Anti-gaming detector:** fires if LOW confidence rate > 70% or HIGH rate > 80% across 10+ episodes. Penalty = `(rate - threshold) * multiplier`. This is the answer to the judge question "can't the agent just always say LOW?".

**Training reward (simple):**
```
r = -0.05 (always)
if done: r += 1.0 (correct) or -0.5 (wrong)
         r += 0.3 * min(legit_flags, 3)
         r += 0.5 * calibration_matrix_value
```

**Eval reward (complex, for demo only):**
```
0.35 * calibration_r + 0.25 * escalation_r + 0.20 * evidence_r + 0.10 * efficiency_r - 0.10 * gaming_penalty
```

### Test results
```
pytest tests/test_calibration.py -v
13 passed in 0.14s
```

All 13 tests:
- `test_high_correct_returns_1_point_0` ✅
- `test_high_wrong_returns_minus_0_point_8` ✅
- `test_med_correct_returns_0_point_6` ✅
- `test_all_outputs_in_valid_range[HIGH-True-1.0]` ✅
- `test_all_outputs_in_valid_range[HIGH-False--0.8]` ✅
- `test_all_outputs_in_valid_range[MED-True-0.6]` ✅
- `test_all_outputs_in_valid_range[MED-False--0.2]` ✅
- `test_all_outputs_in_valid_range[LOW-True-0.1]` ✅
- `test_all_outputs_in_valid_range[LOW-False-0.0]` ✅
- `test_systematic_low_triggers_gaming_penalty` ✅
- `test_systematic_high_triggers_gaming_penalty` ✅
- `test_gaming_detector_needs_10_episodes_minimum` ✅
- `test_training_reward_step_penalty_applied` ✅

### What failed / decisions reversed
- Initially considered keeping `confidence` as `float` (0.0–1.0) for Brier score. Rejected: the 3×2 matrix is the demo-able innovation. A float confidence is just another number. A Literal enum forces the agent to commit to a category — that's what makes the "debate floor" metaphor work and what makes the reward surface explainable to judges.

### Files created/modified
| File | Status | Notes |
|------|--------|-------|
| `server/calibration_grader.py` | 🟢 COMPLETE | Core innovation |
| `app/models.py` | 🟢 UPDATED | confidence Literal, validator |
| `tests/test_calibration.py` | 🟢 COMPLETE | 13/13 passing |
| `server/claim_generator.py` | 🟡 PLACEHOLDER | Next session |
| `inference_debatefloor.py` | 🟡 PLACEHOLDER | Mandatory deliverable |
| `train/train_debatefloor.ipynb` | 🟡 PLACEHOLDER | Mandatory deliverable |
| `docs/CONTEXT.md` | 🟢 CREATED | Session log |

### Estimated score: 46/100 (+8 from Session 0)

---

## Pending / Next Sessions

### Session 2 — Priority: claim_generator.py + inference_debatefloor.py
- Build parametric template engine (5 fraud types × 4 coverage × 3 jurisdictions)
- Seed variation → 500+ unique episodes
- ClaimScenario as Pydantic model
- inference_debatefloor.py — HTTP baseline agent with mandatory stdout format

### Session 3 — OpenEnv server + concurrent sessions
- app/main.py: wire calibration_grader into /step endpoint
- openenv.yaml: add confidence_required, procedural_generation, concurrent support
- Test 4 parallel reset() calls

### Session 4 — GRPO Training notebook
- Colab: Unsloth Qwen2.5-1.5B + TRL GRPOTrainer
- Wire env_reward_fn to training_reward (simple scalar only)
- Produce visible reward curve
- Log to WandB

### Session 5 — Demo + HF deployment
- HF Space: validate /health, /tasks, /schema
- HF blog post
- Before/after transcript
- Confidence distribution histogram

---

## Pitch Q&A — Pre-loaded Answers

**Q: Why not just use Brier score for calibration?**
A: Brier score is a continuous loss — it gives you a gradient but no clear signal about *what kind* of miscalibration the agent is making. Our 3×2 matrix has asymmetric penalties: HIGH+WRONG is -0.8, but LOW+WRONG is 0.0. This teaches the agent that overconfidence is worse than under-confidence — a specific epistemic lesson that Brier score cannot express.

**Q: Can't the agent just always say LOW confidence to avoid punishment?**
A: We detect this. If LOW confidence rate exceeds 70% over 10+ episodes, `detect_confidence_gaming()` fires a progressive penalty of `(rate - 0.7) × 2.0`. Same for HIGH > 80%. The only winning strategy is accurate calibration.

**Q: Why split training and evaluation reward?**
A: GRPO gradients become unstable when multiple reward components with different signs fire simultaneously. Our training reward is a simple shaped scalar (+1.0/-0.5 with bonuses). The complex 6-component evaluation reward is computed separately for demo and reporting only. This is the same principle used in the CoCA paper.

**Q: What makes this a training environment vs a benchmark?**
A: A benchmark has fixed episodes. Our procedural generator takes `(seed, fraud_type, coverage, difficulty)` and generates a unique claim scenario. 5 fraud types × 4 coverage × 3 jurisdictions × seed variation = 500+ distinct training episodes. The agent cannot memorise answers.

**Q: Why insurance domain specifically?**
A: Insurance fraud detection is high-stakes, information-asymmetric, and requires genuine uncertainty quantification — exactly the conditions where calibration matters. An agent that says HIGH confidence on a staged accident it can't prove costs the insurer. Our environment directly penalises that failure mode.

---

## Score Tracker

| Criterion | Session 0 | Session 1 | Target |
|-----------|-----------|-----------|--------|
| Innovation (40%) | 22 | 28 | 35 |
| Storytelling (30%) | 16 | 16 | 26 |
| Reward curve (20%) | 2 | 2 | 16 |
| Pipeline (10%) | 2 | 4 | 9 |
| **Total** | **38** | **46** | **86** |
