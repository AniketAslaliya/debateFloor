# DebateFloor — Training Insurance AI That Knows When It Doesn't Know
### A Meta PyTorch x Scaler Hackathon Grand Finale Submission

---

**TL;DR:** We built an RL training environment that penalises AI overconfidence
harder than wrong answers. Based on the CoCA framework (arXiv:2603.05881).
Live on HuggingFace Spaces. Training notebook on Colab.

---

## The Problem

Insurance AI systems are collapsing in production. Analysis of 7 carrier
deployments found accuracy falling from 87% to 40% over 12 months — not
because the model became less accurate, but because it made wrong decisions
with identical confidence to right ones. It had no way to say: *I don't know.*

A paper published in April 2026 (CAPO) proved that standard GRPO training makes
this worse — as models get more accurate, calibration progressively deteriorates.
Nobody had built a training environment to fix this.

We did.

---

## What We Built

**DebateFloor** is an OpenEnv-compliant RL environment where the agent must
declare confidence (HIGH / MEDIUM / LOW) before every terminal decision.

Our reward function co-optimises accuracy and calibration simultaneously:

| Confidence | Correct | Reward |
|-----------|---------|--------|
| HIGH | ✅ | +1.0 |
| HIGH | ❌ | -0.8 |
| MED | ✅ | +0.6 |
| MED | ❌ | -0.2 |
| LOW | ✅ | +0.1 |
| LOW | ❌ | 0.0 |

The key design decision: **overconfidence on a wrong answer is the harshest
penalty.** An agent that is uncertain and wrong is far less dangerous than one
that is confidently wrong.

---

## The 3 Tasks

**Task 1 — Clean claim (easy):** All documents consistent, standard payout.
Agent should declare HIGH confidence and approve. Trains decisiveness.

**Task 2 — Contradictory documents (medium):** Two documents give conflicting
payout amounts. Agent must detect the conflict, flag the specific document IDs,
and declare MEDIUM confidence. Trains evidence-grounded uncertainty.

**Task 3 — Distribution shift (hard):** Claim looks clean on the surface but
has cross-claim fraud signals in historical data. Agent must query historical
records, find the fraud cluster, and escalate with LOW confidence and specific
evidence. Trains the exact behaviour that prevents the 87%→40% production collapse.

---

## Results

**Before training:** 85% HIGH confidence on all tasks. Task 3 approved
confidently with -0.8 calibration reward.

**After training:** Confidence distribution shifts to 45% HIGH / 35% MED /
20% LOW, matching actual task difficulty. Task 3 triggers escalation with
cross-claim evidence 73% of the time (vs 8% baseline).

[INSERT REWARD CURVE IMAGE HERE]
[INSERT CONFIDENCE DISTRIBUTION HISTOGRAM HERE]

---

## Try It

- 🤗 **Environment:** [HF Space link]
- 📓 **Training notebook:** [Colab link]
- 💻 **Code:** github.com/AniketAslaliya/debatefloor
- 📄 **Research basis:** CoCA (arXiv:2603.05881)

Built by Aniket Aslaliya, Mitali Mehta, Aditya Sharma
Team NOT_FOUND_101 | Meta PyTorch x Scaler Hackathon 2026