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

### Session 2 — April 21, 2026

#### Phase 1C — server/claim_generator.py
Parametric template engine. 5 fraud type builders, each varying by signal_strength (derived from difficulty × rng). ClaimScenario is a Pydantic model.

Key design choices:
- `signal_strength = DIFFICULTY_SIGNAL_STRENGTH[difficulty] * rng.uniform(0.85, 1.0)` — adds intra-difficulty variation so agents can't pattern-match on difficulty alone
- `_incident_date(rng)` — date is deterministic per seed, prevents date memorisation
- `coordinated_ring` always produces 3–5 linked claims and always returns `escalate_to_human` — the only task where escalation is correct
- `phantom_provider` uses fake hospital names generated from name pool — unverifiable by design
- `generate_episode_pool(count=500)` iterates across all combinations until count reached

**500 unique episodes confirmed:** `test_500_unique_episodes_no_duplicates` passed — all 500 claim_ids are distinct.

#### Phase 1D — openenv.yaml
Fully rewritten from Round 1 benchmark yaml to DebateFloor training env yaml. Key additions:
- `supports_concurrent_sessions: true` + `max_concurrent_envs: 64` — required for GRPO parallel rollouts
- `confidence_required: true` + `procedural_generation: true` + `episode_pool_size: 500`
- `distribution_shift_claim` task added (replaces `coordinated_fraud` from Round 1)
- Calibration matrix values documented inline
- `never_mix: true` flag under reward section

#### Test results
```
pytest tests/test_calibration.py tests/test_generator.py -v
45 passed in 0.36s
```

New tests (32):
- `test_same_seed_returns_same_claim` ✅
- `test_different_seeds_return_different_claims` ✅
- `test_claim_id_encodes_seed_and_fraud_type` ✅
- `test_all_fraud_types_generate_correctly[*5]` ✅
- `test_fraud_types_have_signals[*5]` ✅
- `test_clean_claim_approves_and_no_signals` ✅
- `test_coordinated_ring_has_linked_claims` ✅
- `test_coordinated_ring_escalates` ✅
- `test_identity_fraud_has_verify_action` ✅
- `test_easy/medium/hard_max_steps` ✅
- `test_easy_has_low_ambiguity / test_hard_has_high_ambiguity` ✅
- `test_all_coverage_types_generate[*4]` ✅
- `test_500_unique_episodes_no_duplicates` ✅
- `test_pool_covers_all_fraud_types` ✅
- `test_invalid_*_raises[*3]` ✅
- `test_ambiguity_always_in_0_1_range` ✅

#### Files created/modified
| File | Status |
|------|--------|
| `server/claim_generator.py` | 🟢 COMPLETE |
| `openenv.yaml` | 🟢 UPDATED |
| `tests/test_generator.py` | 🟢 COMPLETE (32 tests) |

#### Estimated score: 58/100 (+12 from Session 1)

### Session 3 — April 21, 2026

#### Codebase restructure
- Removed root duplicates: `calibrationGrader.py`, `testCalibration.py`, `context.md`
- Moved planning docs to `docs/`: roadmap, guide, skill, HFBlogPost
- Updated `app/main.py` title to DebateFloor
- Full README.md rewrite for DebateFloor problem statement

#### Environment wiring
- `app/environment.py`: imported `calibration_grader`, wired 3×2 matrix into terminal actions. `deny_claim MED correct` → `calibration_score=0.6` verified live.
- Added `escalate_to_human`, `query_historical_data`, `verify_provider_registration` actions
- `app/tasks.py`: new actions in ACTION_COSTS
- `confidence_required=True` now appears in every observation

#### inference_debatefloor.py
- Full mandatory deliverable built
- 3 deterministic strategies (clean=HIGH, contradictory=MED, distribution_shift=LOW)
- [START]/[STEP]/[END] mandatory stdout format
- `DebateFloorClient` HTTP wrapper

#### Smoke test result
```
deny_claim + confidence=MED on contradictory_claim → calibration_score=0.6 ✅
confidence_required=True in observation ✅
/health → {"status": "healthy"} ✅
```

#### Estimated score: 68/100 (+10 from Session 2)

---

### Session 4 — April 21, 2026

#### train/train_debatefloor.ipynb — 14 cells, complete GRPO pipeline

**Cell structure:**
1. Install (unsloth, trl, pydantic, wandb)
2. Configuration (model, episodes, WandB key)
3. WandB login
4. Load Qwen2.5-1.5B in 4-bit + LoRA adapters
5. Import `training_reward` from `server.calibration_grader`
6. Generate 200-episode dataset via `generate_episode_pool()` → HF Dataset
7. Reward function: `parse_model_output` → `training_reward` (simple scalar only)
8. **Baseline eval (BEFORE)** — records confidence distribution + calibration score per fraud type
9. **GRPOTrainer** — `num_generations=4`, WandB reward curve logging
10. **Post-training eval (AFTER)** — records confidence distribution shift
11. Confidence distribution histogram → `docs/confidence_distribution.png`
12. Before/after transcript on hardest case (coordinated_ring, hard)
13. Save model checkpoint + optional HF Hub push
14. WandB summary + deliverables checklist

**Key design decisions:**
- `reward_funcs=debatefloor_reward_fn` receives `(completions, ground_truth, expected_signals)` — simple scalar, no compound rewards
- Bad format (no DECISION/CONFIDENCE parsed) → -0.2 penalty — teaches the model to follow format
- `legitimate_flags` estimated by scanning response text for expected signal keywords — proxy for fraud detection quality
- `per_device_train_batch_size=4` + `gradient_accumulation_steps=4` → effective batch 16, fits T4
- `fp16=True` (T4 doesn't support bf16)

**Deliverables produced by notebook:**
- `docs/baseline_results.json` — before/after calibration scores per fraud type
- `docs/confidence_distribution.png` — histogram for pitch deck
- `debatefloor_grpo_qwen2.5_1.5b/` — model checkpoint

#### Estimated score: 82/100 (+14 from Session 3)

### Session 5 — Pending: HF Space deployment + blog post
- Deploy to HF Space, validate /health returns 200
- Publish HF blog post (400 words, reward curve screenshot, before/after transcript)
- Run pre_validation_script.py — all green
- Concurrent session test (4 parallel resets)

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

| Criterion | S0 | S1 | S2 | S3 | S4 | Target |
|-----------|----|----|----|----|-----|--------|
| Innovation (40%) | 22 | 28 | 34 | 34 | 35 | 35 |
| Storytelling (30%) | 16 | 16 | 16 | 18 | 20 | 26 |
| Reward curve (20%) | 2 | 2 | 2 | 2 | 16 | 16 |
| Pipeline (10%) | 2 | 4 | 6 | 8 | 9 | 9 |
| **Total** | **38** | **46** | **58** | **68** | **82** | **86** |
