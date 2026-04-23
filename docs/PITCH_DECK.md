# DebateFloor Pitch Deck (Short)

## Slide 1 - Problem
- Insurance claim decision systems fail when confidence is miscalibrated.
- Overconfident wrong decisions are more harmful than uncertain escalations.

## Slide 2 - What We Built
- DebateFloor: OpenEnv RL environment for insurance investigations.
- Agent must output terminal decision plus confidence: HIGH, MED, or LOW.
- Live deployment: Hugging Face Space with FastAPI + Gradio demo.

## Slide 3 - Core Innovation
- 3x2 calibration matrix with asymmetric penalties.
- HIGH + correct = +1.0.
- HIGH + wrong = -0.8 (worst outcome).
- LOW + wrong = 0.0.

## Slide 4 - Multi-Agent Debate
- Investigator gathers evidence.
- convene_debate_panel generates Prosecutor and Defender arguments.
- Judge (main agent) reads transcript and makes final calibrated decision.

## Slide 5 - Verifiable RL Setup
- Procedural episodes generated from fraud type, coverage, jurisdiction, seed.
- Objective checks: decision correctness, calibration score, evidence quality, anti-gaming.
- Compatible with TRL GRPO training pipeline.

## Slide 6 - Training Evidence
- Real run artifacts saved in repo:
  - docs/reward_curve.png (loss + reward plot)
  - docs/component_shift.png (held-out before/after component shift)
  - reports/training_summary.json (step-wise logs)
  - reports/component_shift_summary.json (component means + deltas)
- Rollout quality report:
  - reports/http_rollout_eval.md

## Slide 7 - Demonstrated Outcome
- Scripted calibrated investigator outperforms naive HIGH-confidence baseline.
- Confidence behavior shifts toward task-appropriate calibration.

## Slide 8 - Why This Matters
- Training target is not only accuracy, but epistemic behavior.
- Approach generalizes to other high-stakes domains: finance, healthcare, legal ops.
