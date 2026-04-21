# DebateFloor — Insurance Calibration RL Environment
## Claude Code Context File | Updated: April 2026

---

## 🎯 WHAT THIS PROJECT IS

**DebateFloor** is an OpenEnv-compliant RL training environment for the
Meta PyTorch x Scaler Hackathon Grand Finale (April 25–26, Bangalore).

It trains LLMs to make insurance claim decisions AND declare calibrated
confidence simultaneously — penalising overconfidence harder than wrong
answers. Based on CoCA framework (arXiv:2603.05881).

**Team:** Aniket Aslaliya (lead), Mitali Mehta, Aditya Sharma
**Repo:** github.com/AniketAslaliya/debatefloor
**HF Space:** huggingface.co/spaces/AniketAsla/debatefloor
**Deadline:** April 25, 2026 — 48-hour onsite hackathon

---

## 🏗️ PROJECT STRUCTURE

```
debatefloor/
├── CLAUDE.md                  ← YOU ARE HERE
├── CONTEXT.md                 ← session-by-session progress log
├── SKILL.md                   ← token-efficient patterns
├── ROADMAP.md                 ← checklist with point scores
├── openenv.yaml               ← OpenEnv spec manifest
├── Dockerfile                 ← HF Space deployment
├── requirements.txt
├── inference_debatefloor.py   ← baseline agent script (MANDATORY)
├── app/
│   └── main.py                ← FastAPI server (endpoints)
├── server/
│   ├── insurance_env.py       ← main environment class
│   ├── claim_generator.py     ← procedural episode generator (NEW)
│   └── calibration_grader.py  ← 3×2 matrix reward (NEW CORE)
├── models.py                  ← Pydantic typed models
├── train/
│   └── train_debatefloor.ipynb ← Colab GRPO training (MANDATORY)
├── docs/
│   ├── CONTEXT.md             ← session log
│   └── hf_blog_post.md        ← HuggingFace mini-blog
└── tests/
    ├── test_generator.py
    ├── test_calibration.py
    └── test_env.py
```

---

## ⚡ CRITICAL COMMANDS

```bash
# Install
pip install openenv-core fastapi uvicorn pydantic trl unsloth

# Run locally
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# Validate before every push
python pre_validation_script.py

# Run tests
pytest tests/ -v

# Docker build test
docker build -t debatefloor . && docker run -p 7860:7860 debatefloor

# Run inference baseline
python inference_debatefloor.py --task contradictory_claim --model gpt-4o
```

---

## 🧠 CORE ARCHITECTURE — UNDERSTAND THIS BEFORE EDITING

### The 3-Layer Innovation Stack

```
Layer 1: Procedural Claim Generator
  → seed + fraud_type + coverage + difficulty → ClaimScenario
  → 5 fraud types × 4 coverage × 3 jurisdictions × seeds = 500+ episodes
  → THIS is what makes it a training env, not a benchmark

Layer 2: Calibration Grader (THE CORE INNOVATION)
  → Takes (decision, confidence, ground_truth, episode_history)
  → Returns calibration_reward via 3×2 matrix
  → Anti-gaming detector prevents systematic LOW confidence exploit
  → Based on CoCA paper arXiv:2603.05881

Layer 3: Split Reward Design (CRITICAL — never mix these)
  → TRAINING reward: simple shaped scalar (stable GRPO gradients)
  → EVALUATION reward: full 6-component (for demo and reporting)
```

### The 3×2 Calibration Matrix
```python
MATRIX = {
    ("HIGH", True):  1.0,   # confident + right = best
    ("HIGH", False): -0.8,  # confident + wrong = WORST
    ("MED",  True):  0.6,   # uncertain + right = good
    ("MED",  False): -0.2,  # uncertain + wrong = acceptable
    ("LOW",  True):  0.1,   # very uncertain + right = weak
    ("LOW",  False):  0.0,  # very uncertain + wrong = at least knew
}
```

### OpenEnv API Contract
```
POST /reset           → returns Observation (loads new claim episode)
POST /step            → takes ClaimAction, returns StepResult
GET  /state           → returns current episode State
GET  /tasks           → lists available tasks
GET  /health          → returns {"status": "healthy"}
GET  /schema          → returns action/observation schema
```

---

## 📋 OPENENV SPEC REQUIREMENTS (non-negotiable)

```yaml
# ALL of these must be in openenv.yaml
spec_version: 1
supports_concurrent_sessions: true   # CRITICAL for GRPO parallel rollouts
max_concurrent_envs: 64
confidence_required: true            # DebateFloor innovation
procedural_generation: true          # transforms benchmark → training env
episode_pool_size: 500
```

### Action Space
Every terminal action MUST include confidence field:
```python
class ClaimAction(BaseModel):
    action: Literal[
        "validate_document", "flag_fraud_signal",
        "request_information", "query_historical_data",
        "estimate_payout", "approve_claim",
        "deny_claim", "escalate_to_human"
    ]
    confidence: Optional[Literal["HIGH", "MED", "LOW"]] = None
    # confidence REQUIRED for terminal actions (approve/deny/escalate)
    document_id: Optional[str] = None
    evidence_text: Optional[str] = None
    reason: Optional[str] = None
```

---

## 🎓 THE 3 TASKS

| Task | Difficulty | Max Steps | Fraud Type | Expected Confidence |
|------|-----------|-----------|------------|-------------------|
| clean_claim | Easy | 10 | None | HIGH |
| contradictory_claim | Medium | 18 | medical_inflation | MED |
| distribution_shift_claim | Hard | 28 | coordinated_ring | LOW + escalate |

### Task 3 — The Demo Centrepiece
Distribution shift claim: looks clean on surface but has cross-claim signals
in historical data. Agent must call `query_historical_data()` to find the
fraud cluster. HIGH confidence = wrong regardless of decision.

---

## 🏋️ TRAINING vs EVALUATION REWARD — NEVER MIX

```python
# TRAINING REWARD (simple — stable GRPO gradients)
def training_reward(step):
    r = 0.0
    if step.done:
        r += 1.0 if correct else -0.5
        r += 0.3 * legitimate_fraud_flags
        r += CALIBRATION_MATRIX[(confidence, correct)] * 0.5
    r -= 0.05  # step penalty
    return r

# EVALUATION REWARD (complex — for demo and reporting only)
def eval_reward(episode):
    return (0.35 * calibration_r + 0.25 * escalation_r +
            0.20 * evidence_quality_r + 0.10 * efficiency_r
            - gaming_penalty)
```

---

## 📊 MANDATORY DELIVERABLES CHECKLIST

Before pitching, ALL must be true:
- [ ] HF Space /health returns 200
- [ ] openenv.yaml validates (run pre_validation_script.py)
- [ ] 3 tasks with graders all return scores in [0.0, 1.0]
- [ ] inference_debatefloor.py runs without error, outputs [START]/[STEP]/[END]
- [ ] Colab notebook produces visible reward curve (even if modest)
- [ ] HuggingFace mini-blog published and linked from README
- [ ] Docker builds successfully
- [ ] Concurrent sessions work (test with 4 parallel reset() calls)
- [ ] CoCA citation in README

---

## 🔒 ANTI-GAMING RULES — CRITICAL FOR Q&A

If a judge asks "can't the agent just always say LOW confidence?":
→ detect_confidence_gaming() fires if LOW > 70% of episodes
→ Progressive penalty: (low_rate - 0.7) * 2.0 subtracted from reward
→ Same penalty for HIGH > 80% (systematic overconfidence)
→ Only winning strategy = accurate calibration matching task difficulty

---

## 📝 STDOUT FORMAT — DO NOT DEVIATE

inference_debatefloor.py MUST produce exactly:
```
[START] task=contradictory_claim env=debatefloor model=gpt-4o confidence_required=true
[STEP] step=1 action=validate_document reward=0.0 confidence=null done=False error=None
[STEP] step=2 action=flag_fraud_signal reward=0.15 confidence=null done=False error=None
[STEP] step=3 action=deny_claim reward=0.65 confidence=MED done=True error=None
[END] success=True steps=3 total_reward=0.80 calibration_score=0.60 decision=correct
```

---

## 🚫 NEVER DO THESE

1. NEVER mix training reward with evaluation reward
2. NEVER use confidence=null on terminal actions (approve/deny/escalate)
3. NEVER hardcode claim amounts — always use generator with seed
4. NEVER skip concurrent session support — GRPO will silently break
5. NEVER import from the old insuranceClaim repo directly
6. NEVER push without running pre_validation_script.py first
7. NEVER use pip without --break-system-packages on this machine

---

## 🎯 SCORING TARGET

| Criterion | Current | Target | Key Action |
|-----------|---------|--------|-----------|
| Innovation (40%) | 22 | 35 | procedural gen + calibration grader |
| Storytelling (30%) | 16 | 26 | HF blog + pitch rehearsal |
| Reward curve (20%) | 2 | 16 | Colab notebook + training run |
| Pipeline (10%) | 2 | 9 | concurrent sessions + validation |
| **TOTAL** | **38** | **86** | |

With extras (WandB, investigator agent, web interface): **95+**

---

## 📚 KEY REFERENCES

- CoCA paper: arXiv:2603.05881 (confidence co-optimisation via GRPO)
- CAPO paper: arXiv Apr 2026 (GRPO induces overconfidence — what we fix)
- OpenEnv docs: github.com/openenv/openenv
- TRL GRPOTrainer: huggingface.co/docs/trl/grpo_trainer
- Unsloth: github.com/unslothai/unsloth
- Round 1 repo: github.com/AniketAslaliya/insuranceClaim (reference only)