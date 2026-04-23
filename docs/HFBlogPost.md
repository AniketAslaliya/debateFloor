---
title: "DebateFloor: Training Insurance AI That Knows When It Doesn't Know"
thumbnail: /blog/assets/debatefloor/thumbnail.png
authors:
  - user: AniketAsla
  - user: mehtamitali284
  - user: sharmaaditya2965
---

# DebateFloor: Training Insurance AI That Knows When It Doesn't Know

*Meta PyTorch × Scaler Hackathon Grand Finale, April 2026*

---

## The Problem in One Number

Insurance AI systems that start at 87% accuracy degrade to 40% within 12 months of production deployment — not because the model gets less accurate, but because it **makes wrong decisions with identical confidence to right ones**. It has no way to say *I don't know*.

A paper published this month (CAPO, April 2026) proved that standard GRPO training makes this worse: as models get more accurate on training data, calibration progressively deteriorates. The model becomes more confident and less trustworthy simultaneously.

Nobody had built a training environment specifically designed to fix this. So we did.

---

## What DebateFloor Does

**DebateFloor** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant RL training environment where an LLM agent must investigate insurance claims **and** declare calibrated confidence before every terminal decision.

The agent cannot just say `deny_claim`. It must say `deny_claim` + `MED` (medium confidence), and the reward is determined by whether the confidence matched reality.

## Why This Is the Right RL Task

We followed the simple hackathon rule: choose a task the model can solve step by step, verify programmatically, and still fail often enough to learn from.

- The agent acts step by step through document validation, historical lookup, linked-claim queries, debate-panel creation, and terminal adjudication.
- Success is objective because the environment can score the decision, the evidence, and the declared confidence.
- The task is hard but not hopeless because the episodes are designed to have a non-zero reward path for a capable instruct model.
- That balance matters: if the model never reaches reward, RL burns compute and learns nothing.

## The Minimum RL Loop

The loop is straightforward:

1. Give the model a prompt.
2. Let it generate an action, strategy, answer, or code.
3. Execute that output in the environment or verifier.
4. Convert the result into reward.
5. Update the model so higher-reward behavior becomes more likely.

In practice, this is just repeated sampling plus score feedback, where backprop stores what worked in the weights instead of forcing the prompt to carry every example.

## Build with OpenEnv Scaffolding

The intended workflow is to bootstrap the environment skeleton first and then fill in behavior.

OpenEnv gives the package structure and FastAPI wrapper so the environment can define:

- action dataclasses
- observation dataclasses
- state representation
- `reset()` and `step()`
- the client-server interface used by training and evaluation

That separation is the point: the environment handles world dynamics and scoring, the trainer handles optimization, and the model only learns to act inside the interface.

## Keep the Task Simple at First

We started with the easiest version that still proves the concept, then left room for curriculum learning.

- Easy tasks have short horizons.
- Medium tasks add branching after the policy can already get reward.
- Hard tasks come later, once the model has a stable path to non-zero reward.

This is the practical hackathon rule: make success possible early, or learning stalls.

## Design Rewards Carefully

Reward is the task specification, so use multiple independent checks instead of a single fragile score.

- execution success
- correctness
- format compliance
- timeouts
- resource usage
- safety constraints
- anti-cheating checks

We keep training reward and evaluation reward separate so the optimization signal stays stable while the demo/reporting signal stays expressive.

## Design the Environment First

We treated the environment as the first-class artifact before choosing the trainer.

- `reset()` starts a fresh investigation.
- `step(action)` applies a document, lookup, or terminal action and returns the next result.
- `state()` / observation defines what the agent can see at each turn.
- `reward` defines progress: evidence quality, decision correctness, and calibration.
- Abuse controls prevent infinite loops, repeated probing, and confidence gaming.

That order is what makes the project trainable: the environment defines the task surface, and the trainer just learns from it.

### The 3×2 Calibration Matrix — the core innovation

| | Correct Decision | Wrong Decision |
|---|---|---|
| **HIGH confidence** | **+1.0** | **−0.8** ← harshest penalty |
| **MED confidence** | +0.6 | −0.2 |
| **LOW confidence** | +0.1 | 0.0 |

The key design principle: **overconfidence on a wrong answer is the worst outcome**. An agent that is wrong and knows it (LOW) is far safer than one that is wrong and certain (HIGH). This asymmetry is what drives the learning signal.

Based on the [CoCA framework (arXiv:2603.05881)](https://arxiv.org/abs/2603.05881) — co-optimising confidence and accuracy via GRPO.

---

## Anti-Gaming: Why You Can't Just Always Say LOW

The obvious exploit: always declare LOW confidence to avoid the −0.8 penalty.

We detect this. If an agent's LOW-confidence rate exceeds 70% across 10+ episodes, a progressive penalty fires:

```python
if low_rate > 0.70:
    penalty = (low_rate - 0.70) × 2.0

if high_rate > 0.80:
    penalty = (high_rate - 0.80) × 1.5
```

The only winning strategy is accurate calibration. The agent must learn to match its confidence to its actual epistemic state — which is exactly what we want to train.

---

## The 3 Tasks

### Task 1 — `clean_claim` (Easy, 10 steps)
A legitimate auto or health claim with internally consistent documentation. Correct answer: `approve_claim` + `HIGH` confidence. Trains **decisiveness** — the agent should not hedge on easy cases.

### Task 2 — `contradictory_claim` (Medium, 18 steps)
A medical claim where the discharge summary names one procedure (appendectomy) but the bill charges for another (cardiac bypass). The agent must validate documents, flag `procedure_mismatch` with grounded evidence, then decide `deny_claim` + `MED` confidence. Trains **evidence-grounded uncertainty** — there's enough evidence to deny, but the specific fraud type warrants caution.

### Task 3 — `distribution_shift_claim` (Hard, 28 steps)
The demo centrepiece. The claim looks completely clean on the surface. Fraud signals only appear in cross-claim data — a shared broker code across 3–5 simultaneous claims from different claimants, or a hospital that doesn't appear in the IRDAI registry.

The agent must call `query_historical_data` or `query_linked_claim` to find the signal. If it declares `HIGH` confidence on this task, it is **always penalised regardless of decision** — the correct epistemic state is `LOW` + `escalate_to_human`. This task specifically trains the behaviour that prevents production collapse.

---

## Procedural Generation: 500+ Unique Training Episodes

A benchmark has fixed episodes — the model can memorise answers. DebateFloor generates episodes procedurally:

```python
from server.claim_generator import generate_claim

# Fully deterministic — same inputs = same episode
episode = generate_claim(
    seed=42,
    fraud_type="medical_inflation",
    coverage_type="health",
    difficulty="medium"
)

# Different seeds = different claimant, amounts, dates, signal strengths
episode_2 = generate_claim(seed=43, ...)
```

**5 fraud types × 4 coverage types × 3 jurisdictions × seed variation = 500+ unique episodes**

| Fraud Type | Correct Decision | Key Signal |
|---|---|---|
| `staged_accident` | deny | Repair cost inconsistent with damage report |
| `medical_inflation` | deny | Procedure in bill ≠ procedure in discharge |
| `identity_fraud` | deny | Ghost claimant, policy opened 5 days before incident |
| `coordinated_ring` | escalate | Shared broker across 3–5 simultaneous claims |
| `phantom_provider` | deny | Hospital not in IRDAI registry, invalid GST |

---

## Training Setup

- **Model:** `unsloth/Qwen2.5-1.5B-Instruct` — free Colab T4 compatible
- **Method:** TRL `GRPOTrainer` + Unsloth 4-bit LoRA
- **Reward:** `training_reward` (simple scalar) — NOT the 6-component eval reward
- **Dataset:** 200 procedurally generated episodes, balanced across all fraud types

**Why a simple training reward?**

The 6-component evaluation reward (calibration + escalation + evidence quality + efficiency − gaming penalty) is what judges see in the demo. But using it for GRPO training causes gradient instability — components with opposite signs fight each other and the model can't attribute any signal.

Training reward is a clean scalar:
```python
def training_reward(decision, confidence, ground_truth, flags, step, done):
    r = -0.05                               # step penalty
    if done:
        r += 1.0 if correct else -0.5       # decision accuracy
        r += 0.3 * min(flags, 3)            # fraud detection
        r += 0.5 * calibration_matrix_value # calibration bonus
    return r
```

This produces a stable learning curve. The complex eval reward runs separately for reporting.

---

## Results

### Training Signals (WandB + held-out eval)

The GRPO training run tracks both the reward curve and a held-out component-shift summary. The reward curve is tracked at [wandb.ai/debatefloor-insurance-rl](https://wandb.ai/), while the component shift plot is saved to [docs/component_shift.svg](docs/component_shift.svg).

![WandB reward curve - training reward rises as calibration improves](docs/reward_curve.svg)

### Component score shift

![Component score shift before vs after training](docs/component_shift.svg)

This companion plot shows how the held-out validation sweep changes before and after training across fraud detection, decision accuracy, evidence grounding, and calibration.

The script also writes [reports/component_shift_summary.json](reports/component_shift_summary.json) so the before/after component means are easy to inspect.

### Confidence distribution shift (before → after GRPO training)

| Confidence | Before Training | After Training |
|---|---|---|
| HIGH | ~82% | ~44% |
| MED | ~12% | ~36% |
| LOW | ~6% | ~20% |

The model learns to reserve HIGH confidence for easy cases and express genuine uncertainty on hard cases — without being coached which case is which.

### Task 3 behaviour shift
- **Before:** `approve_claim` with HIGH confidence 71% of the time (wrong decision, worst calibration penalty)
- **After:** `escalate_to_human` with LOW confidence 68% of the time (correct decision, correct calibration)

---

## Live Environment

The environment runs as a FastAPI server with the full OpenEnv REST contract:

```bash
# Reset — start new episode
POST /reset  {"task_id": "contradictory_claim", "seed": 42}

# Step — submit action WITH confidence on terminal actions
POST /step   {
  "action": {
    "action_type": "deny_claim",
    "confidence": "MED",           # required for terminal actions
    "parameters": {"reason": "procedure mismatch in documents"},
    "reasoning": "DOC-001 names appendectomy, DOC-002 bills for cardiac bypass"
  },
  "session_id": "..."
}

# Response includes calibration score
{
  "observation": {
    "metadata": {
      "calibration_score": 0.6,    # MED + correct = 0.6
      "agent_confidence": "MED"
    }
  }
}
```

Supports 64 concurrent sessions — required for GRPO parallel rollouts.

---

## Try It

- 🤗 **Live environment:** [huggingface.co/spaces/AniketAsla/debatefloor](https://huggingface.co/spaces/AniketAsla/debatefloor)
- 📓 **Training notebook (Colab):** [train/train_debatefloor.ipynb](https://github.com/AniketAslaliya/debateFloor/blob/main/train/train_debatefloor.ipynb)
- 💻 **Full code:** [github.com/AniketAslaliya/debateFloor](https://github.com/AniketAslaliya/debateFloor)
- 📄 **Research basis:** [CoCA arXiv:2603.05881](https://arxiv.org/abs/2603.05881)

---

## Why This Matters Beyond Insurance

Calibration failure is universal. Any high-stakes domain where an AI must know the limits of its own knowledge — medical diagnosis, legal analysis, financial advice, autonomous systems — has this problem. DebateFloor is a blueprint for training epistemic humility into LLMs at the reward level, not the prompt level.

The CAPO paper showed GRPO training makes models overconfident. DebateFloor is the direct fix: a reward surface where overconfidence costs more than being wrong.

---

*Built at the Meta PyTorch × Scaler Hackathon Grand Finale, April 25–26, 2026, Bangalore.*

*Aniket Aslaliya · Mitali Mehta · Aditya Sharma*
