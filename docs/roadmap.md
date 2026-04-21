# ROADMAP.md — DebateFloor Build Checklist
## Every task scored. Every dependency mapped. No fluff.

---

## SCORING REALITY

| Criterion | Now | Target | Delta |
|-----------|-----|--------|-------|
| Innovation (40%) | 22 | 35 | +13 |
| Storytelling (30%) | 16 | 26 | +10 |
| Reward curve (20%) | 2 | 16 | +14 |
| Pipeline (10%) | 2 | 9 | +7 |
| **TOTAL** | **38** | **86** | **+48** |

With all extras completed: **95+**

---

## PHASE 0 — PROJECT SETUP (Do before any code)
**Time estimate: 2 hours | Score impact: foundation**

- [ ] Fork github.com/AniketAslaliya/insuranceClaim → rename to debatefloor
- [ ] Create branches: main, feature/calibration, feature/generator, feature/training
- [ ] Copy CLAUDE.md, CONTEXT.md, SKILL.md, ROADMAP.md to root
- [ ] Create folder structure: server/, train/, tests/, docs/
- [ ] Verify local environment: `pip install openenv-core trl unsloth fastapi`
- [ ] Verify Docker is installed: `docker --version`
- [ ] Test Round 1 code still runs: `uvicorn app.main:app --port 7860`

---

## PHASE 1 — CORE FILES (Day 1–2)
**Time estimate: 8 hours | Score impact: +25 pts**

### 1A. models.py — Add confidence field
**Score: +2 pts | Time: 30 min | Owner: Aditya**
- [ ] Add `confidence: Optional[Literal["HIGH", "MED", "LOW"]] = None` to ClaimAction
- [ ] Add `confidence_required: bool = False` to Observation
- [ ] Add `calibration_score: Optional[float] = None` to StepResult
- [ ] Run `pytest tests/test_models.py`

### 1B. server/calibration_grader.py — The core innovation
**Score: +8 pts | Time: 2 hours | Owner: Mitali**
- [ ] Implement 3×2 calibration matrix
- [ ] Implement anti-gaming detector (LOW > 70% = penalty)
- [ ] Implement anti-gaming detector (HIGH > 80% = penalty)
- [ ] Clamp output to [-1.0, 1.0]
- [ ] Write test: `test_high_correct_returns_1.0()`
- [ ] Write test: `test_high_wrong_returns_minus_0.8()`
- [ ] Write test: `test_systematic_low_triggers_penalty()`
- [ ] Write test: `test_output_always_in_bounds()`

### 1C. server/claim_generator.py — Transforms benchmark to env
**Score: +12 pts | Time: 4 hours | Owner: Aniket**
- [ ] Define 5 fraud type templates (use Mitali's domain knowledge)
  - [ ] staged_accident
  - [ ] medical_inflation
  - [ ] identity_fraud
  - [ ] coordinated_ring (needs linked_claims support)
  - [ ] phantom_provider
- [ ] Implement `generate_claim(seed, fraud_type, coverage, difficulty)`
- [ ] Verify 500+ unique episodes (test with 500 different seeds)
- [ ] Write test: `test_same_seed_returns_same_claim()`
- [ ] Write test: `test_different_seeds_return_different_claims()`
- [ ] Write test: `test_all_fraud_types_generate_correctly()`
- [ ] Write test: `test_500_unique_episodes_no_duplicates()`

### 1D. openenv.yaml — Spec compliance
**Score: +2 pts | Time: 30 min | Owner: Aditya**
- [ ] Add `supports_concurrent_sessions: true`
- [ ] Add `max_concurrent_envs: 64`
- [ ] Add `confidence_required: true`
- [ ] Add `procedural_generation: true`
- [ ] Add `episode_pool_size: 500`
- [ ] Update action_space with confidence field
- [ ] Update task list to include distribution_shift_claim

---

## PHASE 2 — ENVIRONMENT INTEGRATION (Day 2–3)
**Time estimate: 4 hours | Score impact: +6 pts**

### 2A. server/insurance_env.py — Wire calibration grader
**Score: +3 pts | Time: 2 hours | Owner: Aniket**
- [ ] Import calibration_grader
- [ ] Import claim_generator
- [ ] Replace fixed scenario loading with `generate_claim(seed=..., ...)`
- [ ] Add confidence validation in `step()`: if terminal action and no confidence → return error
- [ ] Add `calibration_score` to StepResult on terminal actions
- [ ] Track episode_history for anti-gaming detection
- [ ] Test: concurrent session isolation (two sessions don't share state)

### 2B. app/main.py — Minor additions
**Score: +1 pt | Time: 1 hour | Owner: Aditya**
- [ ] Verify all 6 required endpoints exist
- [ ] Add session_id support for concurrent sessions
- [ ] Test: 4 parallel POST /reset calls return different episodes

### 2C. inference_debatefloor.py — Mandatory deliverable
**Score: +2 pts | Time: 1 hour | Owner: Aniket**
- [ ] Copy Round 1 inference.py as base
- [ ] Add confidence declaration to terminal action prompt
- [ ] Update stdout format to include confidence and calibration_score fields
- [ ] Test: runs all 3 tasks without error
- [ ] Test: completes in under 20 minutes
- [ ] Test: [START]/[STEP]/[END] format exactly matches spec

---

## PHASE 3 — DEPLOYMENT (Day 3)
**Time estimate: 2 hours | Score impact: +4 pts**

### 3A. HuggingFace Space deployment
- [ ] Fork Round 1 HF Space → rename to debatefloor
- [ ] Push updated code
- [ ] Run pre_validation_script.py → all green
- [ ] Verify /health returns 200
- [ ] Verify /reset returns valid observation
- [ ] Verify Docker build succeeds
- [ ] Test concurrent sessions from two terminals

### 3B. README.md update
- [ ] Add CoCA citation: "Implements CoCA framework (arXiv:2603.05881)"
- [ ] Add link to HF blog (once written)
- [ ] Add link to Colab training notebook
- [ ] Add confidence_required explanation
- [ ] Add before/after transcript example

---

## PHASE 4 — TRAINING (Day 3–4)
**Time estimate: 6 hours | Score impact: +14 pts**

### 4A. train/train_debatefloor.ipynb — Mandatory deliverable
**Score: +14 pts | Time: 4 hours | Owner: Aditya**
- [ ] Set up Colab T4 runtime
- [ ] Install: `pip install trl unsloth openenv-core`
- [ ] Load model: `unsloth/Qwen2.5-1.5B-Instruct` (free T4 compatible)
- [ ] Implement `env_reward_fn()` using TRAINING reward (not eval reward)
- [ ] Wire `GRPOTrainer` with `environment_factory` pattern
- [ ] Run 50 episodes minimum on `contradictory_claim`
- [ ] Add `report_to="wandb"` for public reward curve link
- [ ] Generate reward curve (even modest upward trend = success)
- [ ] Save: reward curve screenshot as `docs/reward_curve.png`
- [ ] Save: trained model checkpoint

### 4B. Baseline recording
- [ ] Run inference_debatefloor.py on untrained model
- [ ] Record: confidence distribution (% HIGH/MED/LOW per task)
- [ ] Record: calibration_score per task
- [ ] Save as: `docs/baseline_results.json`

### 4C. Post-training evaluation
- [ ] Run inference_debatefloor.py on trained model
- [ ] Compare confidence distribution before vs after
- [ ] Generate: `docs/confidence_distribution.png` (histogram)
- [ ] Record before/after transcript for Task 3 (distribution_shift_claim)

---

## PHASE 5 — MANDATORY DELIVERABLES (Day 4)
**Time estimate: 3 hours | Score impact: +6 pts**

### 5A. HuggingFace mini-blog
**Score: +6 pts | Time: 2 hours | Owner: Mitali**
- [ ] Write 400-word post (use template in docs/hf_blog_post.md)
- [ ] Include: reward curve screenshot
- [ ] Include: before/after transcript snippet
- [ ] Include: link to HF Space, Colab notebook, GitHub repo
- [ ] Publish on HuggingFace
- [ ] Link from README.md
- [ ] Post on LinkedIn (tag @Meta @HuggingFace @PyTorch @Scaler)

---

## PHASE 6 — PITCH PREP (Day 4–5)
**Time estimate: 3 hours | Score impact: +3 pts storytelling**

### 6A. Pitch rehearsal
- [ ] Time the 3-minute pitch (must be 2m 50s – 3m 00s)
- [ ] Rehearse 3× as a team
- [ ] Rehearse Q&A answers cold:
  - [ ] "Can't it always say LOW confidence?" → anti-gaming detector
  - [ ] "Why insurance domain?" → IRDAI circulars + crore liability + domain knowledge
  - [ ] "Reward curve is modest" → confidence distribution shift is the real signal
  - [ ] "What does training teach that fine-tuning doesn't?" → emergent calibration vs label fitting
- [ ] Assign roles: Aniket pitches, Mitali runs demo screen, Aditya has backup laptop

---

## PHASE 7 — EXTRAS (If time permits, Day 5 / onsite)
**Score impact: +9 pts → pushes total to 95+**

### Extra A: WandB public reward curve (+3 pts)
- [ ] Add `report_to="wandb"` to GRPOConfig
- [ ] Create public WandB project: debatefloor-insurance-rl
- [ ] Share link in README and HF blog
- [ ] Show live during pitch

### Extra B: Investigator Agent — Scaler sub-prize (+4 pts)
- [ ] Create `server/investigator_agent.py` — rule-based agent
- [ ] Add `request_investigation(claim_id)` action
- [ ] Investigator returns structured report with fraud signals
- [ ] Main agent uses report in decision
- [ ] Add `InsuranceWorkflowEnv` class alongside main env

### Extra C: Confidence distribution histogram (+2 pts)
- [ ] Script: `docs/generate_charts.py`
- [ ] Before: ~85% HIGH, 12% MED, 3% LOW
- [ ] After: ~45% HIGH, 35% MED, 20% LOW
- [ ] Save as: `docs/confidence_distribution.png`

### Extra D: Live web interface (+2 pts)
- [ ] Set `ENABLE_WEB_INTERFACE=true` in HF Space environment variables
- [ ] Test: full Task 3 episode visible in web browser
- [ ] Use as live demo during pitch

---

## PRE-PITCH FINAL CHECKLIST
**Run this the morning of April 25**

```
□ HF Space /health returns 200
□ pre_validation_script.py passes all checks
□ inference_debatefloor.py runs all 3 tasks without error
□ Colab notebook runs end-to-end
□ WandB curve is public and accessible
□ HF blog is live and linked from README
□ Reward curve screenshot saved locally (backup)
□ Before/after transcripts saved locally (backup)
□ Pitch timed at 2m 55s
□ All 3 Q&A answers practiced cold
□ Aditya has backup laptop with everything loaded
□ Phone battery > 80%
```

---

## STUCK? USE THIS DECISION TREE

```
Something is broken →
  Is it Docker?     → Use SKILL 7 step 1
  Is it /health?    → Use SKILL 7 step 2
  Is it a grader?   → Use SKILL 7 step 3
  Is it concurrent? → Use SKILL 7 step 4
  Is it timeout?    → Use SKILL 7 step 5
  Still stuck?      → Post in hackathon Discord, ask mentor
                      Email: help_openenvhackathon@scaler.com

Context getting noisy in Claude Code? →
  Run /clear
  Re-read CONTEXT.md
  Re-read CLAUDE.md
  Then start fresh session

Reward curve is flat? →
  Check: are you using TRAINING reward (simple) not EVAL reward?
  Check: is model receiving non-zero reward on at least some episodes?
  Check: run 10 episodes manually and print raw reward values
  If still flat → show confidence distribution shift instead
```