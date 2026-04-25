# DebateFloor — Pre-Evaluation Fix Plan (Live Status)

**Status:** Pre-submission hardening — second pass after first-round fixes  
**Deadline:** April 25–26 2026 Grand Finale  
**Last validated:** April 25 2026, 17:00 IST (against current repo state)  
**Priority order:** FATAL → CRITICAL → HIGH → MEDIUM

> **What changed in this revision:** First-pass fixes have been applied to most items.
> This document now reflects what is **actually in the code today**, what is still
> broken, and the exact remaining solution for each.

---

## Status Legend

- **PASS** — Implemented in code, verified against current files
- **PARTIAL** — Code change present but breaks a related contract (test, eval artifact, README, or downstream call site)
- **FAIL** — Promised fix not actually applied to the code path that runs
- **STALE** — Fix is in code but committed artifacts have not been regenerated, so judges will read old data

---

## Current Status Summary

| # | Issue | Status | Blocker for Submission? |
|---|---|---|---|
| FATAL-1 | Training loop never connects to environment | **PASS** | Resolved |
| FATAL-2 | Training evidence shows zero improvement | **PARTIAL** | Yes — README + summary contradict |
| FATAL-3 | Evidence quality is 0.0 in all eval rows | **FAIL** | Yes — wrong `flag_id`s still in code |
| FATAL-4 | `variant_id` always 0 | **STALE** | Yes — eval_report.json never regenerated |
| FATAL-5 | Rubric is decorative; echoes env reward | **PARTIAL** | Yes — test now broken & contradicts fix |
| CRITICAL-1 | No Unsloth usage | **PASS** | Resolved |
| CRITICAL-2 | Training and eval reward use different math | **PARTIAL** | No, but visible in README |
| HIGH-1 | `coordinated_fraud` missing from `openenv.yaml` | **PASS** | Resolved |
| HIGH-2 | Anti-gaming detector disabled across sessions | **FAIL** | Yes — global store exists but is never written |
| HIGH-3 | `server/app.py` violates client/server separation | **PASS** | Resolved |
| HIGH-4 | Training loss 0.005 = model collapse | **PARTIAL** | No, but loss still 0.005 |
| MEDIUM-1 | reward_fn used keyword matching | **PASS** | Resolved (subsumed by FATAL-1 fix) |
| MEDIUM-2 | WandB curve caption ambiguous | **PASS** | Resolved |
| **NEW-1** | Stale `reports/eval_report.json` (3 weeks old) | **FAIL** | Yes |
| **NEW-2** | `tests/envs/test_debatefloor_rubric.py` is broken | **FAIL** | Yes — pytest fails |
| **NEW-3** | README results table contradicts JSON | **FAIL** | Yes — storytelling 30% |
| **NEW-4** | `inference_debatefloor.py` missing strategies for 2 of 5 tasks | **FAIL** | Medium |
| **NEW-5** | Rubric component-name vocabulary drift | **FAIL** | Medium |
| **NEW-6** | README install command is missing deps + wrong TRL pin | **FAIL** | Yes — reviewer reproduction |

**Bottom line:** 7 of the 13 originally listed items are *not* fully resolved
and 6 new issues need attention. Total estimated remaining work: **2–3 hours
of code/text fixes + one re-training run.**

---

## Table of Contents

### Originally Tracked Issues
1. [FATAL-1](#fatal-1--training-loop-never-connects-to-the-environment-pass) — Training loop never connects to env — **PASS**
2. [FATAL-2](#fatal-2--training-evidence-shows-zero-improvement-partial) — Training evidence shows zero improvement — **PARTIAL**
3. [FATAL-3](#fatal-3--evidence-quality-is-00-in-all-eval-rows-fail) — Evidence quality 0.0 in all eval rows — **FAIL**
4. [FATAL-4](#fatal-4--variant_id-is-always-0-stale) — variant_id always 0 — **STALE**
5. [FATAL-5](#fatal-5--rubric-is-decorative-it-echoes-the-environments-own-reward-partial) — Rubric is decorative — **PARTIAL**
6. [CRITICAL-1](#critical-1--no-unsloth-usage-pass) — No Unsloth — **PASS**
7. [CRITICAL-2](#critical-2--training-reward-and-eval-reward-use-completely-different-math-partial) — Training vs eval reward labelling — **PARTIAL**
8. [HIGH-1](#high-1--coordinated_fraud-task-missing-from-openenvyaml-pass) — `coordinated_fraud` missing from YAML — **PASS**
9. [HIGH-2](#high-2--anti-gaming-detector-is-effectively-disabled-during-training-fail) — Anti-gaming disabled across sessions — **FAIL**
10. [HIGH-3](#high-3--serverapppy-violates-clientserver-separation-principle-pass) — `server/app.py` separation — **PASS**
11. [HIGH-4](#high-4--training-loss-0005-indicates-model-collapse-or-no-real-gradient-partial) — Loss 0.005 = collapse — **PARTIAL**
12. [MEDIUM-1](#medium-1--reward_fn-uses-keyword-string-matching-instead-of-env-signals-pass) — Keyword matching reward — **PASS**
13. [MEDIUM-2](#medium-2--wandb-curve-caption-ambiguous-pass) — WandB caption — **PASS**

### Newly Discovered Issues (not in original plan)
14. [NEW-1](#new-1--stale-reportseval_reportjson--md-fail) — Stale `eval_report.json` / `.md`
15. [NEW-2](#new-2--testsenvstest_debatefloor_rubricpy-is-broken-by-the-fatal-5-fix-fail) — Broken rubric test
16. [NEW-3](#new-3--readme-results-table-contradicts-the-actual-json-fail) — README contradicts artifacts
17. [NEW-4](#new-4--inference_debatefloorpy-has-no-strategies-for-2-of-5-tasks-fail) — Missing inference strategies
18. [NEW-5](#new-5--rubric-component-name-vocabulary-drift-fail) — Component-name drift
19. [NEW-6](#new-6--readme-install-command-misses-deps-and-pins-too-old-trl-fail) — README install command broken

### Verification & Sequencing
20. [Quick wins](#quick-wins--do-these-last-they-take--30-minutes-total)
21. [Final fix priority order](#fix-priority-order-day-of-evaluation--remaining-work-only)
22. [Verification checklist](#verification-checklist-final)

---

## FATAL-1 — Training loop never connects to the environment (**PASS**)

### Original problem
`train/train_minimal.py` generated episodes statically and never called `/reset`/`/step`.

### Current state — RESOLVED
File `train/train_minimal.py` now:
- Lines 128–166: `run_episode_via_http(task_id, seed, decision, confidence, reason, base_url)` posts to `/reset` then `/step` and returns `float(step_resp.json()["reward"])`.
- Lines 89–125: `_wait_for_env()` and `_start_env_server_if_needed()` block training start until `/health` returns `{"status":"healthy"}`.
- Lines 238–307: `reward_fn` reads `task_id` and `seed` from the GRPO dataset row, calls `run_episode_via_http` per completion, returns the live reward to `GRPOTrainer`.
- Line 562: `_start_env_server_if_needed(ENV_BASE_URL)` runs before `wandb.init()`.

### Verification (passes today)
- The "kill-switch test" from `HACKATHON_CONSTRAINTS.md` MR-2 holds: turn off the env, training raises `RuntimeError("Environment not reachable …")` after 15 retries.
- `reports/training_summary.json` line 10–13 records a real reward curve `mean_start: 0.0453 → mean_end: 0.3318` from the live env HTTP path.

### No further action required for FATAL-1.

---

## FATAL-2 — Training evidence shows zero improvement (**PARTIAL**)

### Original problem
`reports/training_summary.json` showed `Decision accuracy: 0.0 → 0.0` after training.

### What was fixed
- `reports/training_summary.json` (current): `Decision accuracy: 0.3333 → 0.6667` — non-zero improvement on at least one component.
- `mean_reward_before: 0.0453`, `mean_reward_after_training: 0.3318` — real curve in the JSON.
- `eval_reward_before` and `eval_reward_after` are now separate top-level keys.

### What is still broken
1. **Three of four components do not improve:**
   - `Fraud detection`: 0.3333 → 0.3333
   - `Evidence quality`: 0.3333 → 0.3333
   - `Calibration`: 0.3333 → 0.2 (regression)

   Only `Decision accuracy` moves. Judges scanning the table will see "1 of 4 metrics improved, 1 of 4 regressed."

2. **README headline numbers do not exist anywhere in the JSON.**
   README line 50: `Mean reward: −0.34 → +0.83`. Actual JSON: `0.0453 → 0.3318`. The −0.34/+0.83 numbers are not produced by any current code path.

3. **`reports/component_shift_summary.json` is stale** and contradicts `training_summary.json`:
   - `component_shift_summary.json`: `Calibration: -0.8 → -0.2`
   - `training_summary.json`: `Calibration: 0.3333 → 0.2`
   
   These are the same metric in two different files showing different values.

### Remaining solution

**Step 1 — Re-run training with stronger settings (cost: ~30 min on T4 or A10G credits)**
```python
EPISODES = 500          # was 300
EPOCHS   = 3
num_generations = 8     # was 6
max_completion_length = 128
```
Goal: lift `Fraud detection` and `Evidence quality` off the 0.33 floor.
This requires the model to actually call `validate_document` + `flag_fraud_signal` with a *correct* `flag_id` — which today it cannot, because the prompt only asks for `DECISION/CONFIDENCE/REASON`. Either (a) add a multi-action prompt format, or (b) accept that those two components stay at 0.33 and flag this honestly in the README.

**Step 2 — Replace README headline numbers with actual JSON values**
In `README.md` lines 48–54, replace:
```
| **Mean reward** | −0.34 | **+0.83** |
| **HIGH-confidence episodes** | ~82% | **~44%** |
| **Debate panel convened (hard task)** | 41% | **73%** |
```
With:
```
| **Training reward (live env scalar)** | 0.0453 | **0.3318** (+632%) |
| **Decision accuracy (eval)** | 0.3333 | **0.6667** (+100%) |
| **Calibration score (eval)** | 0.3333 | 0.2000 (regressed; under investigation) |
```
Plus a one-line caveat: "Reward components for Fraud detection and Evidence quality
are flat at 0.3333 because the current prompt format only requests a single terminal
action; multi-step investigative actions are validated separately via `pre_validation_script.py`."

**Step 3 — Regenerate `reports/component_shift_summary.json`**
After Step 1 completes, `save_training_artifacts()` writes both files. Verify they
agree on every common key.

---

## FATAL-3 — Evidence quality is 0.0 in all eval rows (**FAIL**)

### Original problem
The scripted baseline raised wrong `flag_id`s, so `_evidence_total > 0` never triggered.

### Current state — STILL BROKEN
File `inference_debatefloor.py` was **not** corrected:

```python
# inference_debatefloor.py line 154 — still wrong
"flag_id": "procedure_mismatch",
```

`procedure_mismatch` is not in `contradictory_claim`'s `expected_signals`, which `app/tasks.py` lines 200–204 declare as:
```python
expected_signals=[
    "date_mismatch",
    "cost_inflation",
    "signature_mismatch",
    "prior_similar_claim",
],
```

Same problem at line 213:
```python
"flag_id": "clustered_policy_broker",   # for distribution_shift_claim
```
But `distribution_shift_claim` declares `expected_signals = ["shared_repair_shop_far", "shared_emergency_contact", …]` (`app/tasks.py` line 308). The flag is dropped silently → `_evidence_hits` never increments → `evidence_quality_score` stays 0.0.

### Why the other plan's fix did not land
The PLAN.md page for FATAL-3 was written but the corresponding edit to `inference_debatefloor.py` was never made.

### Remaining solution

**Edit `inference_debatefloor.py` `_strategy_contradictory_claim()` (around line 150):**
```python
# REPLACE the wrong flag with one that IS in expected_signals
actions.append({
    "action_type": "flag_fraud_signal",
    "parameters": {
        "flag_id": "date_mismatch",
        "evidence": (
            "Claim form records incident date 2026-02-20 but hospital admission "
            "on 2026-02-17 — date mismatch confirmed across documents."
        ),
    },
    "reasoning": "Date inconsistency is a strong fraud indicator grounded in evidence.",
})
actions.append({
    "action_type": "flag_fraud_signal",
    "parameters": {
        "flag_id": "cost_inflation",
        "evidence": "Hospital bill rate is 2.4× regional standard — cost inflation pattern.",
    },
    "reasoning": "Inflated cost vs benchmark suggests billing fraud.",
})
```

**Edit `_strategy_distribution_shift_claim()` (around line 210):**
```python
# REPLACE clustered_policy_broker with one that IS in expected_signals
actions.append({
    "action_type": "flag_fraud_signal",
    "parameters": {
        "flag_id": "shared_emergency_contact",
        "evidence": "Multiple linked claims share emergency contact phone +91-9000002222.",
    },
    "reasoning": "Shared emergency contact across simultaneous claims indicates coordinated ring.",
})
```

**Verification keyword hints** that must match `app/tasks.py` `get_evidence_keyword_hints()` (lines 645–663):
- `date_mismatch` → keywords: `date`, `admission`, `mismatch`, `incident` ✓ (evidence string above contains all four)
- `cost_inflation` → keywords: `cost`, `rate`, `2.4`, `inflation`, `overbilled` ✓
- `shared_emergency_contact` → keywords: `contact`, `phone`, `emergency`, `shared`, `9000002222` ✓

After this edit, re-run:
```bash
PYTHONPATH=. uvicorn app.main:app --port 7860 &
sleep 5
python inference_debatefloor.py --all-tasks --seed 7 --base-url http://localhost:7860
python pre_validation_script.py --base-url http://localhost:7860
```
Then commit the regenerated `reports/eval_report.json`.

---

## FATAL-4 — variant_id is always 0 (**STALE**)

### Original problem
Eval script did not pass `seed` in the POST body, so `build_runtime_task` always got seed=None → `variant_id = abs(seed) % 5 = 0`.

### Current state
- **Server-side code is correct:** `app/main.py` line 91 forwards `body.seed` to `env.reset(...)`. `app/environment.py` reset path passes `seed` to `build_runtime_task`. `inference_debatefloor.py` line 72 sends `seed` in the JSON body of `/reset`.
- **Stale artifact:** `reports/eval_report.json` is dated **2026-04-03** (3 weeks old) and still contains:
  ```json
  { "task_id": "clean_claim",         "seed": 7,  "variant_id": 0 },
  { "task_id": "clean_claim",         "seed": 17, "variant_id": 0 },
  { "task_id": "contradictory_claim", "seed": 7,  "variant_id": 0 },
  …
  ```
  All 6 rows have `variant_id: 0` and identical reward `0.825`. The fix exists in
  code but the JSON judges will read was never regenerated.

### Remaining solution

**After fixing FATAL-3** (so the same regen pass also produces non-zero
`evidence_quality`), run:
```bash
PYTHONPATH=. uvicorn app.main:app --port 7860 &
sleep 5
python pre_validation_script.py --base-url http://localhost:7860 \
    --output reports/eval_report.json \
    --output-md reports/eval_report.md \
    --seeds 7,17,42 --tasks clean_claim,contradictory_claim,distribution_shift_claim,coordinated_fraud,identity_fraud
```

**Sanity check before commit:**
```python
import json
data = json.load(open("reports/eval_report.json"))
variants = {row["variant_id"] for row in data["rows"]}
assert len(variants) > 1, f"variant_id still constant: {variants}"
evidence = [row["evidence_quality"] for row in data["rows"]]
assert any(e > 0 for e in evidence), "evidence_quality still zero everywhere"
print("✅ eval_report.json passes both invariants")
```

Then `git add reports/eval_report.json reports/eval_report.md && git commit`.

---

## FATAL-5 — Rubric is decorative; it echoes the environment's own reward (**PARTIAL**)

### Original problem
`DebateFloorRubric.forward()` summed env-derived components only → `obs.rubric_reward == obs.reward` always.

### What was fixed (`app/rubrics.py`)
- Added `_ReasoningQualityRubric` (lines 48–70): scans `action.reasoning` for evidence keywords, returns `min(1.0, hits/4.0)`. Independent of env reward.
- `DebateFloorRubric._weights` (lines 94–101) now allocates 0.20 weight to `reasoning_quality`.
- `forward()` (lines 103–109) blends env-derived components with reasoning_quality, then clamps to `[0,1]`.

### What is still broken
**`tests/envs/test_debatefloor_rubric.py` was never updated**, so it now:

1. **Asserts the property the fix invalidates** (line 28):
   ```python
   assert obs.rubric_reward == pytest.approx(obs.reward)
   ```
   This is exactly what HACKATHON_CONSTRAINTS.md AR-2 says is wrong. With the
   new rubric, this assertion can fail (and *should* fail when reasoning_quality
   diverges from env reward).

2. **Expects component keys that no longer exist** (lines 29–39):
   ```python
   assert set(obs.rubric_components) == {
       "fraud_detection", "decision_accuracy",
       "payout_accuracy",         # ← not in new rubric
       "efficiency_score",
       "consistency_score",       # ← not in new rubric
       "evidence_quality_score",
       "calibration_score",
       "penalty",
       "total",
   }
   ```
   `payout_accuracy` and `consistency_score` were renamed/removed during the
   rubric rewrite. The test fails immediately on this set comparison.

A reviewer running `pytest tests/envs/test_debatefloor_rubric.py` today gets a
red bar — much worse for the submission than no test at all.

### Remaining solution

**Replace the test body:**
```python
# tests/envs/test_debatefloor_rubric.py
from __future__ import annotations
import pytest
from app.environment import InsuranceClaimEnvironment
from app.models import InsuranceClaimAction
from app.rubrics import DebateFloorRubric


def test_environment_uses_debatefloor_rubric() -> None:
    env = InsuranceClaimEnvironment()
    assert isinstance(env.rubric, DebateFloorRubric)


def test_rubric_components_are_exposed_on_step() -> None:
    env = InsuranceClaimEnvironment()
    env.reset(task_id="contradictory_claim", seed=42)

    obs = env.step(
        InsuranceClaimAction(
            action_type="deny_claim",
            confidence="MED",
            parameters={"reason": "date mismatch confirmed"},
            reasoning="Date mismatch and cost inflation found across documents — clear fraud signals.",
        )
    )

    # Rubric value is well-formed
    assert 0.0 <= obs.rubric_reward <= 1.0

    # New canonical key set (must match app/rubrics.py:component_scores())
    expected_keys = {
        "fraud_detection",
        "decision_accuracy",
        "calibration_score",
        "evidence_quality_score",
        "efficiency_score",
        "reasoning_quality",   # ← NEW independent signal
        "penalty",
        "total",
    }
    assert set(obs.rubric_components) == expected_keys

    # Independent rubric MAY differ from env reward — do NOT assert equality
    # (this is the AR-2 contract from HACKATHON_CONSTRAINTS.md)
    assert obs.rubric_components["reasoning_quality"] >= 0.0


def test_rubric_can_diverge_from_env_reward() -> None:
    """Independent rubric must be able to disagree with env reward."""
    env = InsuranceClaimEnvironment()
    env.reset(task_id="contradictory_claim", seed=42)

    # Correct decision but no reasoning → reasoning_quality=0, env may still award
    obs_no_reasoning = env.step(
        InsuranceClaimAction(
            action_type="deny_claim",
            confidence="MED",
            parameters={"reason": ""},
            reasoning="",   # empty
        )
    )
    assert obs_no_reasoning.rubric_components["reasoning_quality"] == 0.0
```

Run: `pytest tests/envs/test_debatefloor_rubric.py -v` — must be green.

---

## CRITICAL-1 — No Unsloth usage (**PASS**)

### Current state — RESOLVED

`train/train_minimal.py`:
- Lines 72–79: `from unsloth import FastLanguageModel` with graceful fallback to plain transformers if Unsloth import fails.
- Lines 583–599: `FastLanguageModel.from_pretrained(load_in_4bit=True)` + `FastLanguageModel.get_peft_model(r=16, …, use_gradient_checkpointing="unsloth")`.
- Line 682: `model.save_pretrained_merged("./debatefloor_checkpoint", tok, save_method="merged_16bit")`.

`train/requirements.txt`:
- Line 12: `unsloth` (will need `[colab-new]` extras when installed in Colab; line 11 comment documents the Colab install command).

### No further action required.

---

## CRITICAL-2 — Training reward and eval reward use completely different math (**PARTIAL**)

### What was fixed
- `wandb.init()` config (lines 564–580) tags the run with `reward_type: env_http_reward` and notes the eval scale separately.
- `training_summary.json` saves both `training_reward_curve` (unbounded) and `eval_reward_before/after` (clamped) under separate keys.
- `save_training_artifacts()` plot annotation: "training scalar is unbounded. See eval table for [0,1] clamped scores."

### What is still broken
README presents one "Mean reward" row mixing both:
```
| **Mean reward** | −0.34 | **+0.83** |
```
- −0.34 looks like an unbounded training scalar.
- +0.83 looks like an eval-clamped score.
- Neither value is reproducible from any committed JSON.

### Remaining solution
Already covered in [FATAL-2 Step 2](#fatal-2--training-evidence-shows-zero-improvement-partial). Replace the row with two clearly-labelled rows: one for training scalar, one for eval-clamped score, both citing their JSON source.

---

## HIGH-1 — coordinated_fraud task missing from openenv.yaml (**PASS**)

### Current state — RESOLVED

`openenv.yaml` lines 34–75 list all 5 tasks: `clean_claim`, `contradictory_claim`,
`distribution_shift_claim`, `coordinated_fraud`, `identity_fraud`.

`app/tasks.py` line 509 `list_tasks_summary()` iterates the full `TASKS` dict, so
`GET /tasks` returns all 5 task IDs.

### No further action required.

---

## HIGH-2 — Anti-gaming detector is effectively disabled during training (**FAIL**)

### Original problem
`self._episode_history` lives on each `InsuranceClaimEnvironment` instance, but
`app/main.py` creates one env per `session_id`. With 64 concurrent GRPO sessions,
each session sees ≤2 episodes — far below `MIN_HISTORY_FOR_GAMING_DETECTION = 10`.

### What was scaffolded
- `app/session_store.py` was created with:
  - `_global_confidence_history: deque(maxlen=500)`
  - `_confidence_history_lock: Lock()`
  - `record_episode_confidence(confidence)` — thread-safe append + return snapshot
  - `get_confidence_distribution()` — returns counts for `/stats`
- `app/main.py` line 17: `from .session_store import get_confidence_distribution`
- `app/main.py` lines 154–157: `/stats` endpoint exists and returns the distribution.

### What is still broken
**No code anywhere calls `record_episode_confidence`.** The global deque is
permanently empty.

`app/environment.py` lines 446–451 still uses the per-instance store:
```python
self._calibration_score = compute_calibration_reward(
    effective_decision, conf_str, effective_ground_truth,
    self._episode_history,        # ← per-session, resets every episode
)
self._episode_history.append({"confidence": conf_str})  # ← also per-session
```

`/stats` will report `episodes_recorded: 0` forever, which silently fails the
"anti-gaming is active" claim in the YAML and the README.

### Remaining solution

**Edit `app/environment.py` (around line 28 and lines 446–451):**

```python
# At the top of app/environment.py — add the import
from .session_store import record_episode_confidence

# In the terminal-action branch, REPLACE lines 446–451 with:
global_history = record_episode_confidence(conf_str)
self._calibration_score = compute_calibration_reward(
    effective_decision, conf_str, effective_ground_truth,
    global_history,        # ← cross-session shared history
)
# Optional: also keep self._episode_history for per-session debug/observability
self._episode_history.append({"confidence": conf_str})
```

**Verification (must pass after the edit):**
```bash
PYTHONPATH=. uvicorn app.main:app --port 7860 &
sleep 4
for i in 1 2 3 4 5 6 7 8 9 10 11; do
  SID=$(curl -sX POST http://localhost:7860/reset \
    -H "Content-Type: application/json" \
    -d "{\"task_id\":\"clean_claim\",\"seed\":$i}" | jq -r .session_id)
  curl -sX POST http://localhost:7860/step \
    -H "Content-Type: application/json" \
    -d "{\"action\":{\"action_type\":\"approve_claim\",\"confidence\":\"HIGH\"},\"session_id\":\"$SID\"}" \
    > /dev/null
done
curl -s http://localhost:7860/stats | jq
# Must show:  episodes_recorded ≥ 11,  gaming_detection_active: true
```

If `episodes_recorded` is still 0, the import or call site is wrong.

---

## HIGH-3 — server/app.py violates client/server separation principle (**PASS**)

### Current state — RESOLVED

`server/app.py` is a real entry point:
```python
import uvicorn
from app.main import app  # noqa: F401 — re-exported for uvicorn discovery

__all__ = ["app"]

def serve(host="0.0.0.0", port=7860, workers=1):
    uvicorn.run("server.app:app", host=host, port=port, workers=workers)

if __name__ == "__main__":
    serve()
```

This is "Option A (minimal)" from the original plan — sufficient for AR-4
compliance. Option B (moving the FastAPI app instantiation) was not done and
is not required.

### No further action required.

---

## HIGH-4 — Training loss 0.005 indicates model collapse or no real gradient (**PARTIAL**)

### Original problem
`training_loss: 0.005647` — too low for genuine GRPO learning over 100 episodes.

### What was fixed
- `EPISODES`: 100 → 300 (`train_minimal.py` line 56).
- `EPOCHS`: 2 → 3.
- `num_generations`: 4 → 6 (line 641 — note: lowered from PLAN's recommended 8 to fit T4 VRAM without Unsloth).
- Reward variance is now logged per batch (`reward_fn` lines 293–305) and emitted to WandB as `train/reward_variance`.

### What is still warning-level
- Latest `training_summary.json` line 8: `"training_loss": 0.005260027962633305` — essentially unchanged from the original symptom.
- `reward_fn` only **prints** when variance < 0.01; the `HACKATHON_CONSTRAINTS.md` Part 4 CF-1 pattern says `raise RuntimeError`.
- Reward did rise (0.045 → 0.332) so *some* learning is happening — the loss number alone is not necessarily a problem for GRPO, but combined with the flat 3-of-4 components, it merits scrutiny.

### Remaining solution

**Step 1 — Convert variance warning to a hard guard (matches CF-1 contract).**

In `train/train_minimal.py` lines 292–306, change:
```python
if variance < 0.01:
    print(f"  ⚠️  Low reward variance ({variance:.4f}) — GRPO gradient may be near zero")
```
to:
```python
if variance < 0.01:
    # Allow first 2 batches to warm up; raise after that
    if getattr(reward_fn, "_warmup_done", False):
        raise RuntimeError(
            f"Reward variance collapsed to {variance:.4f}. GRPO will not learn. "
            "Check reward_fn output and dataset diversity."
        )
    print(f"  ⚠️  Low reward variance ({variance:.4f}) — warming up")
reward_fn._warmup_done = True
```

**Step 2 — When you re-train (FATAL-2 Step 1), bump `num_generations` back to 8**
if you have HF credits / A10G+ — more generations per prompt produces more
within-group variance, which is what GRPO actually learns from. T4 may OOM at 8;
A10G/A100 will not.

---

## MEDIUM-1 — reward_fn uses keyword string matching instead of env signals (**PASS**)

### Current state — RESOLVED

This was subsumed by the FATAL-1 fix. `reward_fn` (lines 238–307) now sources
reward exclusively from POST `/step`. The keyword-matching path
(`_score_completion_keyword`, lines 391–415) is retained only as a fallback for
the eval harness when the env is unreachable.

### No further action required.

---

## MEDIUM-2 — WandB curve caption ambiguous (**PASS**)

### Current state — RESOLVED

- `save_training_artifacts()` lines 515–518: matplotlib annotation reads
  *"Note: training scalar is unbounded. See eval table for [0,1] clamped scores."*
- Figure title (line 519): *"DebateFloor GRPO Training Progress (training scalar — not eval score)"*
- Y-axis (line 513): *"Mean reward (training scalar — unbounded)"*
- README has a `> Note on reward scale` block.

### No further action required.

---

## NEW-1 — Stale `reports/eval_report.json` + `.md` (**FAIL**)

### Discovery
Both files are dated **2026-04-03** (3 weeks before today). They contain the
exact `variant_id: 0` / `evidence_quality: 0.0` / constant `0.825 reward`
rows that FATAL-3 and FATAL-4 were supposed to fix.

A judge searching the canonical filename `eval_report.json` will see the broken
3-week-old data and ignore the newer `component_eval_detailed.json`.

### Solution
**Option A (preferred):** Regenerate after FATAL-3 fix:
```bash
python pre_validation_script.py --base-url http://localhost:7860 \
    --output reports/eval_report.json \
    --output-md reports/eval_report.md
```
Verify with the assertion script in [FATAL-4 Remaining solution](#fatal-4--variant_id-is-always-0-stale).

**Option B (acceptable):** Delete both files and rename
`component_eval_detailed.json` → `eval_report.json` if the new file's schema
matches what `pre_validation_script.py` expects.

---

## NEW-2 — `tests/envs/test_debatefloor_rubric.py` is broken by the FATAL-5 fix (**FAIL**)

### Discovery
Already detailed in [FATAL-5](#fatal-5--rubric-is-decorative-it-echoes-the-environments-own-reward-partial).
The test file was not updated when the rubric was rewritten and now:
- Asserts equality with env reward (the property FATAL-5 was meant to break).
- References component names (`payout_accuracy`, `consistency_score`) that
  no longer exist in `app/rubrics.py`.

### Solution
Replace the test body with the version in [FATAL-5 Remaining solution](#fatal-5--rubric-is-decorative-it-echoes-the-environments-own-reward-partial).
Then `pytest tests/envs/test_debatefloor_rubric.py -v` must be green.

---

## NEW-3 — README results table contradicts the actual JSON (**FAIL**)

### Discovery
README lines 48–54:
```
| **Mean reward** | −0.34 | **+0.83** |
| **HIGH-confidence episodes** | ~82% | **~44%** |
| **Debate panel convened (hard task)** | 41% | **73%** |
```

None of those numbers are computed by any current code path:
- `−0.34 / +0.83` does not match any field in `training_summary.json`.
- `82% / 44%` HIGH-confidence rate is not measured anywhere; closest signal
  would be `/stats` distribution (which is permanently 0 because of HIGH-2).
- `41% / 73%` debate-panel-convened rate is not tracked.

### Solution
Already covered in [FATAL-2 Step 2](#fatal-2--training-evidence-shows-zero-improvement-partial).
Replace the table with values that exist in committed JSON. If you want to
keep the HIGH-confidence and debate-panel rows, add the metrics to the eval
script:

```python
# In pre_validation_script.py or run_component_eval.py
from collections import Counter
confidence_dist = Counter(row["agent_confidence"] for row in rows)
high_rate = confidence_dist["HIGH"] / sum(confidence_dist.values())

debate_episodes = sum(1 for row in rows if row.get("debate_convened"))
debate_rate = debate_episodes / len(rows)

summary["high_confidence_rate"] = high_rate
summary["debate_panel_convene_rate"] = debate_rate
```

Then cite those JSON keys in the README.

---

## NEW-4 — `inference_debatefloor.py` has no strategies for 2 of 5 tasks (**FAIL**)

### Discovery
After HIGH-1 added `coordinated_fraud` and `identity_fraud` to the YAML and to
`app/tasks.py`, `inference_debatefloor.py` still defines `STRATEGIES` for only
3 tasks (lines 237–241):
```python
STRATEGIES = {
    "clean_claim":              _strategy_clean_claim,
    "contradictory_claim":      _strategy_contradictory_claim,
    "distribution_shift_claim": _strategy_distribution_shift_claim,
}
```

Running `python inference_debatefloor.py --all-tasks` will hit:
```
[ERROR] No strategy for task 'coordinated_fraud'
[ERROR] No strategy for task 'identity_fraud'
```

### Solution

Add two strategies. Use `expected_signals` from `app/tasks.py` to pick valid `flag_id`s:

```python
def _strategy_coordinated_fraud(client: DebateFloorClient, obs: Dict) -> List[Dict]:
    """Coordinated ring — query linked claims, flag shared signals, escalate LOW."""
    actions = []
    docs = obs.get("observation", obs).get("documents", [])
    linked = obs.get("observation", obs).get("linked_claims", [])

    for doc in docs[:2]:
        actions.append({
            "action_type": "validate_document",
            "parameters": {"doc_id": doc["doc_id"]},
            "reasoning": "Validating documents before cross-claim investigation.",
        })

    for lc in linked[:2]:
        cid = lc.get("claim_id")
        if cid:
            actions.append({
                "action_type": "query_linked_claim",
                "parameters": {"claim_id": cid},
                "reasoning": f"Investigating linked claim {cid} for coordinated patterns.",
            })

    actions.append({
        "action_type": "flag_fraud_signal",
        "parameters": {
            "flag_id": "shared_emergency_contact",
            "evidence": "Multiple linked claims share emergency contact +91-9000002222.",
        },
        "reasoning": "Shared emergency contact across simultaneous claims = ring indicator.",
    })
    actions.append({
        "action_type": "flag_fraud_signal",
        "parameters": {
            "flag_id": "shared_repair_shop_far",
            "evidence": "Repair shop 'RapidFix Kota' is hundreds of km from accident sites.",
        },
        "reasoning": "Shared distant repair shop is geographically improbable.",
    })

    actions.append({
        "action_type": "escalate_to_human",
        "confidence": "LOW",
        "parameters": {"reason": "Coordinated ring suspected; expert review required."},
        "reasoning": "Ring scope unclear — LOW is the calibrated answer.",
    })
    return actions


def _strategy_identity_fraud(client: DebateFloorClient, obs: Dict) -> List[Dict]:
    """Identity fraud — verify identity, flag mismatch, deny MED."""
    actions = []
    docs = obs.get("observation", obs).get("documents", [])

    for doc in docs[:2]:
        actions.append({
            "action_type": "validate_document",
            "parameters": {"doc_id": doc["doc_id"]},
            "reasoning": "Validating ID documents.",
        })

    actions.append({
        "action_type": "verify_identity",
        "parameters": {},
        "reasoning": "Cross-checking claimant identity against national registry.",
    })

    actions.append({
        "action_type": "flag_fraud_signal",
        "parameters": {
            "flag_id": "identity_mismatch",
            "evidence": "National ID registry returns no record matching policy holder name 7821.",
        },
        "reasoning": "Identity mismatch confirmed via verify_identity.",
    })
    actions.append({
        "action_type": "flag_fraud_signal",
        "parameters": {
            "flag_id": "hospital_no_record",
            "evidence": "Hospital admission record has no matching patient name on the claim form.",
        },
        "reasoning": "Hospital lookup confirms ghost claimant.",
    })

    actions.append({
        "action_type": "deny_claim",
        "confidence": "MED",
        "parameters": {"reason": "Identity mismatch confirmed; ghost claimant indicators."},
        "reasoning": "Deny with MED — strong evidence but document forgery cannot be 100% certain.",
    })
    return actions


# Register both
STRATEGIES = {
    "clean_claim":              _strategy_clean_claim,
    "contradictory_claim":      _strategy_contradictory_claim,
    "distribution_shift_claim": _strategy_distribution_shift_claim,
    "coordinated_fraud":        _strategy_coordinated_fraud,
    "identity_fraud":           _strategy_identity_fraud,
}
```

Also update the top-level `TASK_CONFIG` (lines 39–52) to include the two new tasks.

---

## NEW-5 — Rubric component-name vocabulary drift (**FAIL**)

### Discovery
Three places use three different vocabularies for the same components:

| Source | Names used |
|---|---|
| `app/rubrics.py` `_weights` (lines 94–101) | `fraud_detection`, `decision_accuracy`, `calibration_score`, `evidence_quality_score`, `efficiency_score`, `reasoning_quality` |
| `app/rubrics.py` `component_scores()` (lines 111–122) | Same six + `penalty` + `total` |
| `tests/envs/test_debatefloor_rubric.py` (lines 29–39) | Includes `payout_accuracy` and `consistency_score` (which **don't exist** in the current rubric) |
| `train/train_minimal.py` `_COMPONENT_LABELS` (lines 184–188) | Uses display labels `Fraud detection`, `Decision accuracy`, `Evidence quality`, `Calibration` |
| `reports/training_summary.json` | Uses display labels (matches train_minimal) |

### Solution
Pick **one canonical key set** and propagate. Recommended:
- Programmatic keys (snake_case): `fraud_detection`, `decision_accuracy`,
  `calibration_score`, `evidence_quality_score`, `efficiency_score`,
  `reasoning_quality`, `penalty`, `total`.
- Display labels (in JSON/README/plots): map via a single dict in
  `train/train_minimal.py`:
  ```python
  _COMPONENT_LABELS = [
      ("fraud_detection",        "Fraud detection"),
      ("decision_accuracy",      "Decision accuracy"),
      ("evidence_quality_score", "Evidence quality"),
      ("calibration_score",      "Calibration"),
      ("reasoning_quality",      "Reasoning quality"),  # ← add this
  ]
  ```

After updating, `_score_completion_via_http` should also surface `reasoning_quality`
from the rubric so the before/after table covers it (otherwise the new rubric
component is invisible to judges).

---

## NEW-6 — README install command misses deps and pins too-old TRL (**FAIL**)

### Discovery
README line 238:
```bash
pip install trl>=0.9.0 transformers peft accelerate datasets wandb matplotlib
```

Issues:
1. **TRL >=0.9.0** is too old. `train/train_minimal.py` line 52 imports `GRPOConfig, GRPOTrainer` which were added in TRL 0.10. `train/requirements.txt` correctly pins `trl>=0.12.0`.
2. **Missing `unsloth`** — but `train_minimal.py` requires it (CRITICAL-1) and degrades to vanilla transformers only as a fallback.
3. **Missing `requests`** — used by `run_episode_via_http`.
4. **Missing `openenv-core`** — needed because `train_minimal.py` imports `from server.calibration_grader import …` and the env server in turn imports `openenv.core.env_server.interfaces`.

A reviewer copy-pasting this line gets `ImportError: cannot import name 'GRPOConfig'` and stops.

### Solution
Replace README lines 235–240 with:
```bash
git clone https://github.com/AniketAslaliya/debateFloor.git && cd debateFloor

# Use the canonical pinned requirements
pip install -r requirements.txt          # env server deps
pip install -r train/requirements.txt    # training deps incl. Unsloth, TRL>=0.12

# Optional (Colab T4): use the Unsloth nightly for best 4-bit speed
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

PYTHONPATH=. python train/train_minimal.py
```

---

## Quick Wins — do these last, they take < 30 minutes total

### QW-1 — Run pre_validation_script.py against the live Space
```bash
python pre_validation_script.py --base-url https://huggingface.co/spaces/AniketAsla/debatefloor
```
All checks must be green. Pin the Space (Settings → Pin Space) so judges don't see a cold-start delay.

### QW-2 — Verify `/tasks` returns all 5 task IDs against the live Space
```python
import requests
r = requests.get("https://aniketasla-debatefloor.hf.space/tasks").json()
ids = {t["task_id"] for t in r["tasks"]}
assert ids == {"clean_claim", "contradictory_claim", "coordinated_fraud",
               "identity_fraud", "distribution_shift_claim"}
```

### QW-3 — Confirm Colab badge in README opens the right notebook
README line 17 already has the badge. Click it from a logged-out browser to
ensure GitHub serves the public notebook.

### QW-4 — Commit regenerated artifacts
After [FATAL-3, FATAL-4, NEW-1, FATAL-2 re-train]:
```bash
git add reports/ docs/reward_curve.svg docs/component_shift.svg \
        inference_debatefloor.py app/environment.py \
        tests/envs/test_debatefloor_rubric.py README.md
git commit -m "fix: complete second-pass FATAL/HIGH fixes; regenerate eval artifacts"
git push
```

### QW-5 — `/rollout` endpoint already exists (`app/main.py` lines 160–185)
Verify it works against the live Space:
```bash
curl -X POST "https://aniketasla-debatefloor.hf.space/rollout?task_id=contradictory_claim&seed=42" | jq
```
Should return a step-by-step trace ending in a terminal action.

### QW-6 — Make sure `/stats` actually reports non-zero (depends on HIGH-2 fix)
```bash
curl -s https://aniketasla-debatefloor.hf.space/stats | jq
# AFTER HIGH-2 fix + a few episodes, must show episodes_recorded > 0
```

---

## Fix Priority Order (Day-of-Evaluation, **Remaining Work Only**)

| # | Issue | Fix Type | Est. Time | Blocking? |
|---|-------|----------|-----------|-----------|
| 1 | **HIGH-2**: Wire `record_episode_confidence` in `environment.py` | code, 1 file | 10 min | Yes — `/stats` claims fail |
| 2 | **FATAL-3**: Fix `flag_id`s in `inference_debatefloor.py` | code, 1 file | 15 min | Yes — eval evidence quality |
| 3 | **NEW-2 / FATAL-5**: Update `tests/envs/test_debatefloor_rubric.py` | test, 1 file | 15 min | Yes — pytest fails |
| 4 | **NEW-1 / FATAL-4**: Regenerate `reports/eval_report.json` + `.md` | run + commit | 10 min | Yes — stale variant_id=0 |
| 5 | **NEW-3 / FATAL-2 / CRITICAL-2**: Rewrite README results table | docs, 1 file | 15 min | Yes — storytelling 30% |
| 6 | **NEW-6**: Fix README install command | docs, 1 file | 2 min | Yes — reviewer reproduction |
| 7 | **NEW-4**: Add `_strategy_coordinated_fraud` + `_strategy_identity_fraud` | code, 1 file | 30 min | Medium — `--all-tasks` errors |
| 8 | **HIGH-4 / CF-1**: Convert variance warning → `raise RuntimeError` | code, 1 file | 5 min | No — but Part 4 contract |
| 9 | **NEW-5**: Reconcile component-name vocabulary | code, 2 files | 20 min | No — but visible in artifacts |
| 10 | **FATAL-2 Step 1**: Re-run training with bigger settings (use HF credits) | training | 30 min on A10G | Yes — lift flat components |
| 11 | **FATAL-2 Step 3**: Regenerate `component_shift_summary.json` | output of #10 | auto | Yes — drops contradiction |

**Total remaining time: ~2 hours of work + 1 training run.**

> **Recommendation:** Do items 1–9 *before* spending any HF credits.
> All 9 are zero-compute logic/text fixes. Once the pipeline is provably
> correct end-to-end (run all `pytest`, `pre_validation_script`, and the
> 11-call `/stats` check), spend the credits on item 10 with confidence.

---

## Verification Checklist (Final)

Every item below must be `true` before submitting. Tick them in order; an
earlier failure invalidates later items.

### Live Environment
- [ ] `/health` returns `{"status": "healthy"}` on the live HF Space
- [ ] `/tasks` returns all 5 task IDs on the live Space
- [ ] `/reset` with seed=7 vs seed=42 returns different `documents[0].content`
- [ ] `/step` with `deny_claim MED` returns higher reward than `approve_claim HIGH` on `contradictory_claim`
- [ ] `/stats` after 11 episodes returns `episodes_recorded ≥ 11`, `gaming_detection_active: true`
- [ ] `/rollout?task_id=contradictory_claim&seed=42` returns a non-empty trace ending in `done: true`

### Eval Artifacts
- [ ] `reports/eval_report.json` is dated today, not 2026-04-03
- [ ] `reports/eval_report.json` has `evidence_quality > 0.0` for at least one row
- [ ] `reports/eval_report.json` has at least 2 distinct `variant_id` values across seeds
- [ ] `reports/eval_report.json` has different rewards for different tasks (not all 0.825)
- [ ] `reports/component_shift_summary.json` agrees with `reports/training_summary.json` on every common metric

### Training Artifacts
- [ ] `reports/training_summary.json` shows `decision_accuracy after > before`
- [ ] `reports/training_summary.json` shows at least 2 of 4 components improving (currently only 1)
- [ ] `docs/reward_curve.svg` has labeled axes and shows the curve going up
- [ ] `docs/component_shift.svg` shows a meaningful before/after delta (not flat)
- [ ] WandB run URL in README resolves to a real run with `eval/before/*` and `eval/after/*` keys logged

### Code & Tests
- [ ] `pytest tests/envs/test_debatefloor_rubric.py -v` passes (currently fails)
- [ ] `train/train_minimal.py` imports `FastLanguageModel` from `unsloth`
- [ ] `train/train_minimal.py` `reward_fn` calls `run_episode_via_http`
- [ ] `app/environment.py` calls `record_episode_confidence` on every terminal action
- [ ] `inference_debatefloor.py` has `STRATEGIES` entry for all 5 task IDs
- [ ] `inference_debatefloor.py` `flag_id`s in `_strategy_contradictory_claim` and `_strategy_distribution_shift_claim` are in their tasks' `expected_signals`

### YAML & Spec Compliance
- [ ] `openenv.yaml` lists all 5 task IDs (currently true)
- [ ] Every action in `openenv.yaml:action_space` is handled in `app/environment.py:_apply_action`
- [ ] `server/app.py` is a real entry point, not a one-line re-export (currently true)

### Submission Documents
- [ ] README HF Space URL is live and serving
- [ ] README WandB run URL resolves to the correct run (matches the JSON we ship)
- [ ] README Colab badge opens the correct notebook
- [ ] README "Mean reward" row matches numbers in `training_summary.json`
- [ ] README install command uses `pip install -r ...` not the broken inline list
- [ ] README links the writeup (`docs/HFBlogPost.md` — already linked)
- [ ] Trained model is pushed to HF Hub and linked from README (already linked at line 40)

---

## When to Use the HF Credits

**Not yet.** All items 1–9 above are zero-compute. They are also the items most
likely to make a judge mark you down on day-of-evaluation: a failing test, a
contradictory README, an empty `/stats`, a stale `eval_report.json`.

Burn the credits exactly once, on item 10, **after** items 1–9 are done and a
local 50-episode smoke training (T4) confirms all 4 component scores move.

The model choice (Qwen2.5-0.5B-Instruct) is correct for this submission and
should not be changed. A bigger model would invalidate the before/after delta
that the judging rubric explicitly looks for.
