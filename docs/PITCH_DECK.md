# DebateFloor Pitch Deck (Short)

## Slide 1 - Problem
- Insurance claim decision systems fail when confidence is miscalibrated.
- Overconfident wrong decisions are more harmful than uncertain escalations.

## Slide 2 - What We Built
- DebateFloor: OpenEnv RL environment for insurance investigations.
- Agent must output terminal decision plus confidence: HIGH, MED, or LOW.
- Live deployment: Hugging Face Space with FastAPI + Gradio demo.

## Slide 2.25 - Build with OpenEnv Scaffolding
- OpenEnv bootstraps the environment skeleton as a Python package plus FastAPI wrapper.
- We define action, observation, and state first, then fill in behavior.
- The environment owns scoring and dynamics; the trainer only optimizes inside the interface.

## Slide 2.5 - Why This Is the Right RL Task
- The model acts step by step.
- Success is programmatically verifiable.
- The task is hard enough to matter, but still yields reward for a capable base model.
- We start from a capable instruct model, add light formatting scaffolding, and use RL for improvement.

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
- Design the environment first: reset, step, state/observation, reward, anti-abuse.
- Procedural episodes generated from fraud type, coverage, jurisdiction, seed.
- Objective checks: decision correctness, calibration score, evidence quality, anti-gaming.
- Compatible with TRL GRPO training pipeline.

## Slide 5.5 - Keep the Task Simple First
- Start with the easiest useful task, not the hardest benchmark.
- Move to medium branching only after the model reaches non-zero reward.
- Add the hardest cases after the agent can already succeed on simpler ones.
- Curriculum only works if success is reachable early.

## Slide 5.75 - Design Rewards Carefully
- Reward is the task specification, so use multiple independent checks.
- Check execution success, correctness, format compliance, timeouts, resource usage, safety, and anti-cheating separately.
- Keep training reward and evaluation reward separate.

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
