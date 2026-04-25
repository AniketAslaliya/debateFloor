# DebateFloor — Pre-Evaluation Fix Plan (Live Status)

**Status:** Pre-submission hardening — seventh pass after README rewrite  
**Deadline:** April 25–26 2026 Grand Finale  
**Last validated:** April 25 2026, 19:25 IST (against current repo state + live HF Space)  
**Priority order:** FATAL → CRITICAL → HIGH → MEDIUM

> **What changed in this revision:**
> - **NEW-3 → PASS**, **CRITICAL-2 → PASS**, **FATAL-2 → PASS** (storytelling
>   half), and **NEW-6 → PASS.** Rewrote the README `Results` block, the two
>   plot captions, and the Training Pipeline install command so every number
>   shown is read directly from `reports/training_summary.json` and
>   `reports/eval_report.json` (no hand edits, no rounded estimates).
>   Sanity script verifies all 11 cited numbers match JSON; all 10 forbidden
>   hand-edited tokens (`-0.34`, `+0.83`, `~82%`, `~44%`, `41%`, `73%`,
>   `trl>=0.9.0`, plus unicode-minus variants) are gone; install command now
>   sources `requirements.txt` and `train/requirements.txt`.
>   - Training delta now shown: `0.0453 → 0.3318` (training scalar) and
>     `Decision accuracy 0.3333 → 0.6667` with honest "1 of 4 components
>     improved; `Calibration` regressed `0.3333 → 0.2`" caveat.
>   - Scripted-baseline eval table cites the per-task numbers from the
>     15-row regen: `clean_claim 0.7625 / ev_q 1.0`,
>     `contradictory_claim 0.7497 / ev_q 1.0`,
>     `distribution_shift_claim 0.3966 / ev_q 0.0` (with NEW-7 footnote),
>     `average_reward 0.6363`, `completion 100%`.
>   - The remaining open work for FATAL-2 is the **re-training half**
>     (Step 1 + Step 3) which requires HF credits; the storytelling-side
>     contradiction is fully resolved this revision.
> - **NEW-1 → PASS** and **FATAL-4 → PASS** (rev 6): eval_report regenerated,
>   5 distinct variant_ids, 10/15 evidence_quality > 0, average reward `0.6363`.
> - **NEW-2 → PASS** and **FATAL-5 → PASS** (rev 5): rubric test rewrite,
>   49/49 tests pass.
> - **FATAL-3 → PASS** (rev 4): `inference_debatefloor.py` flag_id fix,
>   contradictory_claim evidence_quality `0.0 → 1.0`.
> - **HIGH-2 → PASS** (rev 3): `record_episode_confidence` wired,
>   live `/stats` proof captured.

---

## Status Legend

- **PASS** — Implemented in code, verified against current files (and where applicable, the live Space)
- **PARTIAL** — Code change present but breaks a related contract (test, eval artifact, README, or downstream call site)
- **FAIL** — Promised fix not actually applied to the code path that runs
- **STALE** — Fix is in code but committed artifacts have not been regenerated, so judges will read old data

---

## Current Status Summary

| # | Issue | Status | Blocker for Submission? |
|---|---|---|---|
| FATAL-1 | Training loop never connects to environment | **PASS** | Resolved |
| FATAL-2 | Training evidence shows zero improvement | **PARTIAL** ⚠ | Storytelling half PASS (rev 7); re-training half pending HF credits |
| FATAL-3 | Evidence quality is 0.0 in all eval rows | **PASS** ✅ | **Resolved 25 Apr 17:50 IST** (contradictory_claim 0.0 → 1.0) |
| FATAL-4 | `variant_id` always 0 | **PASS** ✅ | **Resolved 25 Apr 19:05 IST** (5 distinct variant_ids in regenerated report) |
| FATAL-5 | Rubric is decorative; echoes env reward | **PASS** ✅ | **Resolved 25 Apr 18:20 IST** (rubric `0.29` vs env `0.428` for same step → divergence proven) |
| CRITICAL-1 | No Unsloth usage | **PASS** | Resolved |
| CRITICAL-2 | Training and eval reward use different math | **PASS** ✅ | **Resolved 25 Apr 19:25 IST** (README rewrite cites both scales by name + JSON source) |
| HIGH-1 | `coordinated_fraud` missing from `openenv.yaml` | **PASS** | Resolved |
| HIGH-2 | Anti-gaming detector disabled across sessions | **PASS** ✅ | **Resolved 25 Apr 17:25 IST** |
| HIGH-3 | `server/app.py` violates client/server separation | **PASS** | Resolved |
| HIGH-4 | Training loss 0.005 = model collapse | **PARTIAL** | No, but loss still 0.005 |
| MEDIUM-1 | reward_fn used keyword matching | **PASS** | Resolved (subsumed by FATAL-1 fix) |
| MEDIUM-2 | WandB curve caption ambiguous | **PASS** | Resolved |
| **NEW-1** | Stale `reports/eval_report.json` (3 weeks old) | **PASS** ✅ | **Resolved 25 Apr 19:05 IST** (regenerated against live HF, 15 rows) |
| **NEW-2** | `tests/envs/test_debatefloor_rubric.py` is broken | **PASS** ✅ | **Resolved 25 Apr 18:20 IST** (49/49 tests pass) |
| **NEW-3** | README results table contradicts JSON | **PASS** ✅ | **Resolved 25 Apr 19:25 IST** (every cited number now read directly from JSON) |
| **NEW-4** | `inference_debatefloor.py` missing strategies for 2 of 5 tasks | **FAIL** | Medium |
| **NEW-5** | Rubric component-name vocabulary drift | **FAIL** | Medium |
| **NEW-6** | README install command is missing deps + wrong TRL pin | **PASS** ✅ | **Resolved 25 Apr 19:25 IST** (now sources `requirements.txt` + `train/requirements.txt`) |
| **NEW-7** | `distribution_shift_claim` has no discovery path for its `expected_signals` | **FAIL** | Medium — caps that task's evidence_quality at 0.0 |

**Bottom line:** 1 of the 13 originally listed items is not fully resolved
(FATAL-2 — re-training half only; storytelling half is PASS this revision).
3 newly discovered issues remain (NEW-3 / NEW-6 are PASS; NEW-4 / NEW-5 /
NEW-7 still pending). Total estimated remaining work:
**~55 min of code fixes + one re-training run.**

---

## Table of Contents

### Originally Tracked Issues
1. [FATAL-1](#fatal-1--training-loop-never-connects-to-the-environment-pass) — Training loop never connects to env — **PASS**
2. [FATAL-2](#fatal-2--training-evidence-shows-zero-improvement-partial) — Training evidence shows zero improvement — **PARTIAL** (storytelling half PASS rev 7)
3. [FATAL-3](#fatal-3--evidence-quality-is-00-in-all-eval-rows-pass) — Evidence quality 0.0 in all eval rows — **PASS** ✅
4. [FATAL-4](#fatal-4--variant_id-is-always-0-pass) — variant_id always 0 — **PASS** ✅
5. [FATAL-5](#fatal-5--rubric-is-decorative-it-echoes-the-environments-own-reward-pass) — Rubric is decorative — **PASS** ✅
6. [CRITICAL-1](#critical-1--no-unsloth-usage-pass) — No Unsloth — **PASS**
7. [CRITICAL-2](#critical-2--training-reward-and-eval-reward-use-completely-different-math-pass) — Training vs eval reward labelling — **PASS** ✅
8. [HIGH-1](#high-1--coordinated_fraud-task-missing-from-openenvyaml-pass) — `coordinated_fraud` missing from YAML — **PASS**
9. [HIGH-2](#high-2--anti-gaming-detector-is-effectively-disabled-during-training-pass) — Anti-gaming disabled across sessions — **PASS** ✅
10. [HIGH-3](#high-3--serverapppy-violates-clientserver-separation-principle-pass) — `server/app.py` separation — **PASS**
11. [HIGH-4](#high-4--training-loss-0005-indicates-model-collapse-or-no-real-gradient-partial) — Loss 0.005 = collapse — **PARTIAL**
12. [MEDIUM-1](#medium-1--reward_fn-uses-keyword-string-matching-instead-of-env-signals-pass) — Keyword matching reward — **PASS**
13. [MEDIUM-2](#medium-2--wandb-curve-caption-ambiguous-pass) — WandB caption — **PASS**

### Newly Discovered Issues (not in original plan)
14. [NEW-1](#new-1--stale-reportseval_reportjson--md-pass) — Stale `eval_report.json` / `.md` — **PASS** ✅
15. [NEW-2](#new-2--testsenvstest_debatefloor_rubricpy-is-broken-by-the-fatal-5-fix-pass) — Broken rubric test — **PASS** ✅
16. [NEW-3](#new-3--readme-results-table-contradicts-the-actual-json-pass) — README contradicts artifacts — **PASS** ✅
17. [NEW-4](#new-4--inference_debatefloorpy-has-no-strategies-for-2-of-5-tasks-fail) — Missing inference strategies
18. [NEW-5](#new-5--rubric-component-name-vocabulary-drift-fail) — Component-name drift
19. [NEW-6](#new-6--readme-install-command-misses-deps-and-pins-too-old-trl-pass) — README install command broken — **PASS** ✅
20. [NEW-7](#new-7--distribution_shift_claim-has-no-discovery-path-for-its-expected_signals-fail) — `distribution_shift_claim` evidence_quality structurally capped at 0.0 (env code)

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

## FATAL-3 — Evidence quality is 0.0 in all eval rows (**PASS** ✅)

### Original problem
The scripted baseline raised wrong `flag_id`s, so `_evidence_total > 0` produced
zero `_evidence_hits` and the env counted them as false flags.

### What was fixed (commit shipped this revision)

**`inference_debatefloor.py` — `_strategy_contradictory_claim()`:**
- Replaced the single wrong flag (`procedure_mismatch`) with **two correct flags**:
  - `date_mismatch` (in `expected_signals`, discovered by validating DOC-10 / DOC-11)
  - `cost_inflation` (in `expected_signals`, discovered by validating DOC-12)
- Evidence text contains the keywords required by
  `app/tasks.py:get_evidence_keyword_hints()` for each flag:
  - `date_mismatch` ← contains `date`, `admission`, `mismatch`, `incident`
  - `cost_inflation` ← contains `cost`, `rate`, `2.4`, `inflation`, `overbilled`

**`inference_debatefloor.py` — `_strategy_distribution_shift_claim()`:**
- **Removed** the wrong flag (`clustered_policy_broker`). After tracing the
  env code, this task has **no discovery path for any of its 5
  expected_signals**:
  - `app/environment.py:_discover_signals_from_document()` mapping (lines
    601–620) has no entry for `distribution_shift_claim`.
  - `query_linked_claim` only special-cases `CLM-GROUP-304` from
    `coordinated_fraud` (line 413).
  - `compare_documents` `COMPARE_DOCUMENT_SIGNALS` dict in `app/tasks.py`
    has no entry for this task either.
  - Flagging anything that IS in `expected_signals` triggers the
    "raised before discovered" penalty (`+0.08` to `penalty_total`,
    `+0.02` to `exploit_penalty`) without earning an evidence hit.
- The honest behaviour is to **skip the `flag_fraud_signal` step** and
  escalate based on the cross-claim hint surfaced by `query_linked_claim`.
  This drops the penalty without losing any earnable credit.

### Verification — apples-to-apples BEFORE / AFTER on `seed=42` (live env)

```
=== contradictory_claim ===
  BEFORE: reward=0.5180  evidence_quality=0.0000  hits/total=0/1  penalty=0.1000
  AFTER : reward=0.7497  evidence_quality=1.0000  hits/total=2/2  penalty=0.0000
  delta evidence_quality: 0.0000 -> 1.0000  (+1.0000)
  delta reward          : 0.5180 -> 0.7497  (+0.2317)
  delta penalty         : 0.1000 -> 0.0000  (-0.1000)

=== distribution_shift_claim ===
  BEFORE: reward=0.2930  evidence_quality=0.0000  hits/total=0/1  penalty=0.1000
  AFTER : reward=0.3966  evidence_quality=0.0000  hits/total=0/0  penalty=0.0000
  delta evidence_quality: 0.0000 -> 0.0000  ( 0.0000) [structural — see note]
  delta reward          : 0.2930 -> 0.3966  (+0.1036)
  delta penalty         : 0.1000 -> 0.0000  (-0.1000)
```

`discovered_signals` for the AFTER `contradictory_claim` run:
`["date_mismatch", "cost_inflation", "prior_similar_claim"]` — proves both
new flags entered `_discovered_signals` via `validate_document` calls before
being flagged with grounded evidence.

### Regression check
`tests/test_calibration.py` + `tests/envs/test_insurance_claim_reward_and_exploit.py`:
**43 / 43 pass** after the fix.

### Pushes
- GitHub `origin/main`: see commit `<filled in by next push>`
- HF Space `AniketAsla/debatefloor`: redeployed via
  `huggingface_hub.create_commit()` (workaround for the HF git protocol bug).

### Remaining open item (logged separately, not a FATAL-3 regression)
The fact that `distribution_shift_claim` has no discovery path for its
`expected_signals` is a deeper env-code bug. Adding entries to
`_discover_signals_from_document` and `COMPARE_DOCUMENT_SIGNALS` for
`distribution_shift_claim` would let the strategy actually earn evidence
credit on this task. Estimated effort: ~30 minutes; tracked in
[NEW-7](#new-7--distribution_shift_claim-has-no-discovery-path-for-its-expected_signals-fail).

---

## FATAL-4 — variant_id is always 0 (**PASS** ✅)

### Original problem
Eval script did not pass `seed` in the POST body, so `build_runtime_task` always got seed=None → `variant_id = abs(seed) % 5 = 0`.

### Server-side code (already correct, no change in this revision)
- `app/main.py` forwards `body.seed` to `env.reset(...)`.
- `app/environment.py` reset path passes `seed` to `build_runtime_task`.
- `inference_debatefloor.py` sends `seed` in the JSON body of `/reset`.
- `app/tasks.py:548` `variant_id = abs(seed) % 5`.

### What was shipped this revision
Built `train/generate_eval_report.py` — a focused regenerator. (The
`pre_validation_script.py --output / --seeds / --tasks` flags suggested in
earlier revisions of this plan were never implemented in that script.) The
new tool:

- Imports `STRATEGIES` from `inference_debatefloor.py` (the canonical baseline).
- Sweeps **5 seeds** chosen to hit every variant exactly once:
  `[7, 11, 13, 19, 25]` → `variant_id ∈ {2, 1, 3, 4, 0}` = all 5 variants.
- Sweeps **3 tasks** (`clean_claim`, `contradictory_claim`,
  `distribution_shift_claim`) — the ones with shipped strategies. Adding
  the remaining 2 (`coordinated_fraud`, `identity_fraud`) is tracked under
  NEW-4.
- Produces 15 rows in both `reports/eval_report.json` and `reports/eval_report.md`.
- Runs the PLAN-prescribed invariant assertion at the end and exits non-zero
  if either FATAL-3 or FATAL-4 invariant breaks.

### Verification — live HF Space numbers (no fabrication)

PLAN's prescribed assertion script:
```
PASS: 5 distinct variant_ids: [0, 1, 2, 3, 4]
PASS: 10/15 rows with evidence_quality > 0
PASS: average_reward=0.6363
PASS: completion_rate=100.0%
PASS: generated_at=2026-04-25T13:37:28.409790+00:00
PASS: base_url=https://aniketasla-debatefloor.hf.space
PASS: total rows=15
eval_report.json passes both FATAL-3 and FATAL-4 invariants
```

| Metric | Before (stale 2026-04-03) | After (regen 2026-04-25) |
|---|---|---|
| Total rows | 6 | **15** |
| Distinct `variant_id` | `{0}` | **`{0, 1, 2, 3, 4}`** |
| Distinct rewards | 2 (`0.825`, `0.9475`) | **3 (`0.3966`, `0.7497`, `0.7625`)** |
| Rows with `evidence_quality > 0` | 0 / 6 | **10 / 15** |
| Average reward | 0.8658 | 0.6363 |
| `generated_at` | 2026-04-03T16:40:41 | **2026-04-25T13:37:28** |
| `base_url` | live HF (old project name) | live HF (current Space) |

Note on the average dropping: the new report includes
`distribution_shift_claim` (which the old report omitted) and uses the
**actual** `inference_debatefloor.py` strategies rather than fabricated
constant rewards. The `0.6363` number is what the canonical scripted
baseline genuinely scores against the live env.

How to regenerate later:
```bash
python train/generate_eval_report.py \
  --base-url https://aniketasla-debatefloor.hf.space
```

---

## FATAL-5 — Rubric is decorative; it echoes the environment's own reward (**PASS** ✅)

### Original problem
`DebateFloorRubric.forward()` summed env-derived components only → `obs.rubric_reward == obs.reward` always.

### What was fixed (`app/rubrics.py`)
- Added `_ReasoningQualityRubric` (lines 48–70): scans `action.reasoning` for evidence keywords, returns `min(1.0, hits/4.0)`. Independent of env reward.
- `DebateFloorRubric._weights` (lines 94–101) now allocates 0.20 weight to `reasoning_quality`.
- `forward()` (lines 103–109) blends env-derived components with reasoning_quality, then clamps to `[0,1]`.

### What WAS still broken (now resolved this revision)
**`tests/envs/test_debatefloor_rubric.py` was never updated**, so it:

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

A reviewer running `pytest tests/envs/test_debatefloor_rubric.py` would get a
red bar — much worse for the submission than no test at all.

### What was shipped this revision

**Replaced the test file with a 6-test suite that defends the FATAL-5 contract.**
Live numbers from `app.environment.InsuranceClaimEnvironment` (no fabrication):

| Test | obs.reward | obs.rubric_reward | divergence | reasoning_quality |
|---|---|---|---|---|
| `test_rubric_diverges_from_env_reward` (deny + "validation check") | 0.428 | 0.29 | **0.138** | 0.0 |
| `test_reasoning_quality_zero_for_empty_reasoning` | 0.428 | 0.29 | **0.138** | 0.0 |
| `test_reasoning_quality_positive_for_evidence_rich_reasoning` | 0.458 | 0.52 | **0.062** | **1.0** |
| `test_rubric_components_present_on_intermediate_steps` (validate_document) | 0.17 | 0.2625 | **0.0925** | 1.0 |

The non-zero divergence column is the **proof** that `obs.rubric_reward != obs.reward`,
which is what FATAL-5 was originally about and what the original test was
silently masking by asserting equality.

`pytest tests/envs/test_debatefloor_rubric.py -v` →
**6 passed in 12.74s.**

`pytest tests/test_calibration.py tests/envs/test_insurance_claim_reward_and_exploit.py tests/envs/test_debatefloor_rubric.py -v` →
**49 passed in 12.26s** (full DebateFloor regression).

### Test design — what each new test guards

1. `test_environment_uses_debatefloor_rubric` — env wires the right rubric class.
2. `test_rubric_components_are_exposed_on_step` — exact 8-key set is exposed
   (`fraud_detection`, `decision_accuracy`, `calibration_score`,
   `evidence_quality_score`, `efficiency_score`, `reasoning_quality`,
   `penalty`, `total`); `total` matches `obs.rubric_reward`; metadata mirror
   matches.
3. `test_rubric_diverges_from_env_reward` — strict inequality
   `obs.rubric_reward != pytest.approx(obs.reward, abs=1e-3)` for the same
   action that previously asserted equality. **This is the FATAL-5 contract
   in code form.** A regression here means the rubric has stopped being
   independent.
4. `test_reasoning_quality_zero_for_empty_reasoning` — empty/short reasoning
   forces `reasoning_quality = 0.0` (the 20-char threshold in
   `_ReasoningQualityRubric`).
5. `test_reasoning_quality_positive_for_evidence_rich_reasoning` — evidence
   keywords push `reasoning_quality` above 0; bounded at 1.0.
6. `test_rubric_components_present_on_intermediate_steps` — rubric fires on
   non-terminal actions too (regression guard for `validate_document`).

### Original (no longer needed) verbatim solution kept for reference
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

## CRITICAL-2 — Training reward and eval reward use completely different math (**PASS** ✅)

### What was fixed (earlier passes)
- `wandb.init()` config tags the run with `reward_type: env_http_reward`.
- `training_summary.json` saves both `training_reward_curve` (unbounded
  scalar) and `eval_reward_before/after` (clamped components) under
  separate keys.
- `save_training_artifacts()` plot annotation already noted the scale
  difference.

### What was still broken (until this revision)
README presented one "Mean reward" row mixing both scales (`−0.34 →
+0.83`), with neither value reproducible from any committed JSON.

### Resolution (shipped this revision — same edit as NEW-3)
The README `Results` block now has two **explicitly labelled** sections:

1. **GRPO training delta** — first row labelled
   `Training reward (live env scalar, unbounded — used for GRPO gradients)`
   citing `mean_reward_before / mean_reward_after_training`. All
   subsequent rows in this section labelled
   `(eval, clamped [0,1])` and citing `eval_reward_before/after.*`.
2. **Scripted-baseline eval** — header labelled
   `Mean reward [0,1]` and `Mean evidence_quality`, citing
   `reports/eval_report.json`. The aggregate is the clamped
   `eval_report.average_reward = 0.6363`.

A note block right under the table makes the scale separation explicit:

> Training-time reward (`0.0453 → 0.3318`) is the **raw GRPO training
> scalar** (unbounded — used for gradient stability). The four eval
> components above are the **clamped `[0,1]` per-component scores** from
> the live environment. Different numbers, different scales —
> intentionally kept separate per `openenv.yaml:never_mix=true`.

The reward-curve caption explicitly warns: "Y-axis is the unbounded
training scalar; do not compare to the clamped `[0,1]` eval components".

No further action required for CRITICAL-2.

---

## HIGH-1 — coordinated_fraud task missing from openenv.yaml (**PASS**)

### Current state — RESOLVED

`openenv.yaml` lines 34–75 list all 5 tasks: `clean_claim`, `contradictory_claim`,
`distribution_shift_claim`, `coordinated_fraud`, `identity_fraud`.

`app/tasks.py` line 509 `list_tasks_summary()` iterates the full `TASKS` dict, so
`GET /tasks` returns all 5 task IDs.

### No further action required.

---

## HIGH-2 — Anti-gaming detector is effectively disabled during training (**PASS** ✅)

### Original problem
`self._episode_history` lived on each `InsuranceClaimEnvironment` instance, but
`app/main.py` creates one env per `session_id`. With 64 concurrent GRPO sessions,
each session saw ≤2 episodes — far below `MIN_HISTORY_FOR_GAMING_DETECTION = 10`.
`/stats` permanently reported `episodes_recorded: 0` and
`gaming_detection_active: false`, contradicting the `openenv.yaml` claim and the
README "anti-gaming" innovation.

### What was fixed (commit `9f2d218`, HF Space `402ef31bbbe0`)

**`app/environment.py` (5 added / 2 removed):**
```python
# Top of file — new import
from .session_store import record_episode_confidence

# In the terminal-action branch (lines 446–453)
# HIGH-2 fix: use the global cross-session history so anti-gaming
# detection actually fires during concurrent GRPO rollouts. The
# per-instance _episode_history is kept only for per-session debug.
global_history = record_episode_confidence(conf_str)
self._calibration_score = compute_calibration_reward(
    effective_decision, conf_str, effective_ground_truth,
    global_history,
)
self._episode_history.append({"confidence": conf_str})
```

The pre-existing `app/session_store.py` already provided
`_global_confidence_history: deque(maxlen=500)`, a `Lock()`, and
`record_episode_confidence()`/`get_confidence_distribution()` — they were just
never wired in. This change wires them in.

### Verification — actual numbers from live endpoints (not invented)

| Metric | Local server | Live HF Space |
|---|---|---|
| `/stats` baseline `episodes_recorded` | 0 | 0 |
| Episodes issued (11 distinct `session_id`s) | 11 | 11 |
| `/stats` `episodes_recorded` after | **11** | **11** |
| HIGH share (4 issued / 11) | 0.364 | 0.364 |
| MED share (4 issued / 11) | 0.364 | 0.364 |
| LOW share (3 issued / 11) | 0.273 | 0.273 |
| `gaming_detection_active` | true | true |
| Cross-session probe (12th ep in new session sees prior 11) | distribution → 0.333 / 0.333 / 0.333 | — |
| Regression suite (`test_calibration.py` + `test_insurance_claim_reward_and_exploit.py`) | 43 / 43 pass | — |

Live probe command (reproducible by judges):
```bash
curl -s https://aniketasla-debatefloor.hf.space/stats | jq
```

### Pushes
| Target | Result | Commit / SHA |
|---|---|---|
| GitHub `origin/main` | `d77231c..9f2d218 main -> main` | `9f2d218` |
| HF Space `AniketAsla/debatefloor` | Build → `RUNNING` | `402ef31bbbe0` |

The HF push went through `huggingface_hub.create_commit()` because
`git push hf` hits the known HF Spaces protocol bug
(`fatal: expected 'acknowledgments'`); helper script
`push_high2_fix_to_hf.py` is left in the workspace for future redeploys.

### No further action required for HIGH-2.

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

## NEW-1 — Stale `reports/eval_report.json` + `.md` (**PASS** ✅)

### Discovery
Both files were dated **2026-04-03** (22 days before today). They contained
the exact `variant_id: 0` / `evidence_quality: 0.0` / constant `0.825 reward`
rows that FATAL-3 and FATAL-4 were supposed to fix. A judge searching the
canonical filename `eval_report.json` would have seen the broken 22-day-old
data and ignored the newer `component_eval_detailed.json`.

### Resolution (shipped this revision)
Built `train/generate_eval_report.py` (see
[FATAL-4 → What was shipped](#fatal-4--variant_id-is-always-0-pass) for full
detail) and ran it twice:

1. Against the local uvicorn dev server (smoke test) — invariants PASS.
2. Against the **live HF Space** — invariants PASS, files committed.

Both files now show:
- `generated_at: 2026-04-25T13:37:28+00:00` — fresh.
- `base_url: https://aniketasla-debatefloor.hf.space` — production.
- 15 rows × 3 tasks × 5 seeds covering all 5 variant_ids.
- Distinct variant_ids `{0, 1, 2, 3, 4}` (FATAL-4 invariant).
- 10/15 rows with `evidence_quality > 0` (FATAL-3 invariant).
- Markdown table sorted by `(task, seed)` and includes a `Variant` column
  and a `Steps` column for readability.

The PLAN-prescribed assertion script runs clean.

---

## NEW-2 — `tests/envs/test_debatefloor_rubric.py` is broken by the FATAL-5 fix (**PASS** ✅)

### Discovery
Already detailed in [FATAL-5](#fatal-5--rubric-is-decorative-it-echoes-the-environments-own-reward-pass).
The test file was not updated when the rubric was rewritten and:
- Asserted equality with env reward (the property FATAL-5 was meant to break).
- Referenced component names (`payout_accuracy`, `consistency_score`) that
  no longer exist in `app/rubrics.py`.

### Resolution (shipped this revision)
Replaced the test body with the 6-test suite documented in
[FATAL-5 → What was shipped](#fatal-5--rubric-is-decorative-it-echoes-the-environments-own-reward-pass).

`pytest tests/envs/test_debatefloor_rubric.py -v` → **6 / 6 PASS** in 12.74s.

`pytest tests/test_calibration.py tests/envs/test_insurance_claim_reward_and_exploit.py tests/envs/test_debatefloor_rubric.py -v`
→ **49 / 49 PASS** in 12.26s.

---

## NEW-3 — README results table contradicts the actual JSON (**PASS** ✅)

### Discovery
The previous README results block carried 6 fabricated headline numbers
(`−0.34 / +0.83`, `~82% / ~44%`, `41% / 73%`) and 1 fabricated plot
caption (`Calibration shifts from −0.8 to 0.0`). None of these numbers
were produced by any current code path:
- `−0.34 / +0.83` did not match any field in `training_summary.json`.
- `82% / 44%` HIGH-confidence rate was not tracked in any committed eval
  artifact (it is now technically measurable via `/stats` after the
  HIGH-2 fix, but no eval script captures before/after rates yet).
- `41% / 73%` debate-panel-convened rate was not tracked at all.
- `−0.8 → 0.0` Calibration delta came from the stale
  `component_shift_summary.json`, which itself contradicts
  `training_summary.json` (`0.3333 → 0.2`).

### Resolution (shipped this revision)

Rewrote the README `Results` block + both plot captions so every figure
shown is read directly from a committed JSON artifact, with the source
key cited inline in a new "Source key" column.

**Section 1 — GRPO training delta** (cites
`reports/training_summary.json`):
- `Training reward (live env scalar)` `0.0453 → 0.3318` —
  `mean_reward_before` / `mean_reward_after_training`.
- `Decision accuracy (eval, [0,1])` `0.3333 → 0.6667` (+100%) —
  `eval_reward_after.Decision accuracy`.
- `Calibration score (eval, [0,1])` `0.3333 → 0.2000` (regressed,
  flagged with ⚠) — `eval_reward_after.Calibration`.
- `Fraud detection (eval)` and `Evidence quality (eval)` shown as
  flat at `0.3333` (honest reporting).
- One-line caveat documents that 1 of 4 components moved with the
  current single-action prompt format.

**Section 2 — Scripted-baseline eval** (cites
`reports/eval_report.json`):
- 15 episodes (3 tasks × 5 seeds, all 5 variant_ids), generated
  `2026-04-25T13:37:28+00:00` against the live HF Space.
- Per-task means: `clean_claim 0.7625 / ev_q 1.0`,
  `contradictory_claim 0.7497 / ev_q 1.0`,
  `distribution_shift_claim 0.3966 / ev_q 0.0` (with NEW-7 footnote
  explaining the structural cap).
- Aggregate: `0.6363` mean reward, 100% completion.
- Inline regeneration command: `python train/generate_eval_report.py
  --base-url …`.

**Reward-curve caption rewritten** to cite `training_summary.json`'s
actual `0.0453 → 0.3318` and warn against comparing to the clamped
`[0,1]` eval scale.

**Component-shift caption rewritten** to show the real
`Decision 0.3333 → 0.6667` lift, the `Calibration 0.3333 → 0.2`
regression, and the two flat components, while explicitly disowning
the legacy `component_shift_summary.json` which still shows the stale
`-0.8 → -0.2` numbers (regen tracked under FATAL-2 Step 3).

### Verification — automated sanity script (no fabrication)

```
Numbers cited in README:
  [PASS] cited=0.0453 json=0.0453 in_readme=True
  [PASS] cited=0.3318 json=0.3318 in_readme=True
  [PASS] cited=0.3333 json=0.3333 in_readme=True
  [PASS] cited=0.6667 json=0.6667 in_readme=True
  [PASS] cited=0.2000 json=0.2000 in_readme=True
  [PASS] cited=0.7625 json=0.7625 in_readme=True
  [PASS] cited=0.7497 json=0.7497 in_readme=True
  [PASS] cited=0.3966 json=0.3966 in_readme=True
  [PASS] cited=1.0000 json=1.0000 in_readme=True
  [PASS] cited=0.0000 json=0.0000 in_readme=True
  [PASS] cited=0.6363 json=0.6363 in_readme=True

Forbidden ASCII tokens (must be absent):
  [PASS] '-0.34' / '+0.83' / '~82%' / '~44%' / '41%' / '73%' / 'trl>=0.9.0'

Forbidden Unicode-minus tokens (must be absent):
  [PASS] '\u22120.34' / '\u22120.8 (overconfident' / '\u22120.83'

Overall: PASS
```

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

## NEW-6 — README install command misses deps and pins too-old TRL (**PASS** ✅)

### Discovery
The previous README install line:
```bash
pip install trl>=0.9.0 transformers peft accelerate datasets wandb matplotlib
```
Issues confirmed:
1. **TRL >=0.9.0** is too old. `train/train_minimal.py` imports
   `GRPOConfig, GRPOTrainer` which were added in TRL 0.10.
   `train/requirements.txt` correctly pins `trl>=0.12.0`.
2. **Missing `unsloth`** — but `train_minimal.py` requires it (CRITICAL-1).
3. **Missing `requests`** — used by `run_episode_via_http`.
4. **Missing `openenv-core`** — needed because `train_minimal.py` imports
   `from server.calibration_grader import …` and the env server in turn
   imports `openenv.core.env_server.interfaces`.

A reviewer copy-pasting that line would get
`ImportError: cannot import name 'GRPOConfig'` and stop.

### Resolution (shipped this revision)
Replaced the install block with:

```bash
git clone https://github.com/AniketAslaliya/debateFloor.git && cd debateFloor

# Use the canonical pinned requirements files (every dep verified to
# import inside train_minimal.py and the env server).
pip install -r requirements.txt          # env server deps (FastAPI, openenv-core, ...)
pip install -r train/requirements.txt    # training deps (trl, unsloth, peft, wandb, ...)

# Optional (Colab T4): swap the pinned unsloth for the colab-new wheel
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

PYTHONPATH=. python train/train_minimal.py
```

### Verification
- `Test-Path requirements.txt` → True (116 bytes, server deps).
- `Test-Path train/requirements.txt` → True (463 bytes, training deps;
  pins `trl>=0.12.0`, `unsloth`, `requests`, `openenv-core>=0.2.3`).
- README sanity script `[PASS] 'trl>=0.9.0' present=False`.
- README sanity script `[PASS] 'pip install -r requirements.txt'` and
  `'pip install -r train/requirements.txt'` both present.

No further action required for NEW-6.

---

## NEW-7 — `distribution_shift_claim` has no discovery path for its `expected_signals` (**FAIL**)

### Discovery (during FATAL-3 fix)
While tracing why the FATAL-3 fix could not raise `evidence_quality` above
0.0 for `distribution_shift_claim`, found three independent gaps in the env
code:

1. `app/environment.py:_discover_signals_from_document()` (lines 601–620)
   has entries for `clean_claim`, `contradictory_claim`, `coordinated_fraud`,
   `identity_fraud` — **but not** `distribution_shift_claim`. Validating
   any of DOC-41 / DOC-42 / DOC-43 returns `[]`.

2. `app/environment.py:_apply_action()` `query_linked_claim` branch
   (lines 412–415) hardcodes
   `if match.get("broker_id") and claim_id == "CLM-GROUP-304"`.
   `CLM-GROUP-304` belongs to `coordinated_fraud`. None of `CLM-DIST-602`,
   `CLM-DIST-603`, `CLM-DIST-604` trigger any signal discovery.

3. `app/tasks.py:COMPARE_DOCUMENT_SIGNALS` (lines 669–686) has no entry for
   `distribution_shift_claim`, so `compare_documents` never discovers
   anything for this task either.

Result: every flag in this task's `expected_signals` is unreachable.
Flagging any of them triggers the "raised before discovered" penalty
(`+0.08 penalty_total`, `+0.02 exploit_penalty`). The honest agent move
is to skip flagging — which is what the FATAL-3 fix now does — but this
caps `evidence_quality` at 0.0 for the task in the eval table.

### Solution
Add discovery hooks symmetric to `coordinated_fraud`:

**`app/environment.py:_discover_signals_from_document()` — add:**
```python
"distribution_shift_claim": {
    "DOC-41": ["recent_policy_cluster"],     # claim form metadata flags
    "DOC-42": ["shared_repair_shop_far"],    # garage estimate exposes shop
    # DOC-43 reveals nothing direct; cross-claim only
},
```

**`app/environment.py` `query_linked_claim` branch — broaden the broker
discovery beyond `CLM-GROUP-304`:**
```python
# Already special-cased: CLM-GROUP-304 (coordinated_fraud) → clustered_policy_broker
# Add: any CLM-DIST-* with shared broker_id once 2 linked claims have been queried
if (
    match.get("broker_id") and
    claim_id.startswith("CLM-DIST-") and
    len(self._queried_claims) >= 2
):
    self._record_discovered_signals(["clustered_policy_broker"])
```

**`app/environment.py` `query_linked_claim` branch — also surface
`shared_emergency_contact` as a discovered signal (not just a hint string)
once the cross-claim contact match is detected (lines 400–410):**
```python
if len(contacts) > 1 and len(unique_contacts) == 1:
    self._record_discovered_signals(["shared_emergency_contact"])
    hint = f" Cross-claim pattern detected: shared emergency_contact={contacts[0]}."
```

**`app/tasks.py:COMPARE_DOCUMENT_SIGNALS` — add entries for
`distribution_shift_claim`** if you want `compare_documents` to also
contribute (optional; the discovery above is sufficient).

**`app/tasks.py:get_evidence_keyword_hints()` — add a `distribution_shift_claim`
sub-dict** (currently absent) with the keyword lists for each of the 5
signals so the keyword check in `flag_fraud_signal` works:
```python
"distribution_shift_claim": {
    "shared_repair_shop_far": ["repair", "shop", "fastrepair", "whitefield"],
    "shared_emergency_contact": ["contact", "phone", "emergency", "9000005555"],
    "recent_policy_cluster":   ["policy", "purchase", "days", "cluster"],
    "clustered_policy_broker": ["broker", "brk-882", "same broker"],
    "near_identical_descriptions": ["identical", "description", "narrative"],
},
```

### Verification (after fix)
Update `_strategy_distribution_shift_claim()` to validate DOC-41 and DOC-42,
query 2 linked claims, then flag `shared_emergency_contact` and
`shared_repair_shop_far`. Expected: `evidence_quality > 0` for this task in
the BEFORE/AFTER harness.

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
        inference_debatefloor.py \
        tests/envs/test_debatefloor_rubric.py README.md
git commit -m "fix: complete third-pass FATAL fixes; regenerate eval artifacts"
git push origin main
python push_high2_fix_to_hf.py   # or extend it to push the new files
```

### QW-5 — `/rollout` endpoint already exists (`app/main.py` lines 160–185)
Verify it works against the live Space:
```bash
curl -X POST "https://aniketasla-debatefloor.hf.space/rollout?task_id=contradictory_claim&seed=42" | jq
```
Should return a step-by-step trace ending in a terminal action.

### QW-6 — `/stats` reports non-zero (HIGH-2 — DONE ✅)
Already verified live:
```bash
$ curl -s https://aniketasla-debatefloor.hf.space/stats | jq
{
  "episodes_recorded": 11,
  "distribution": { "HIGH": 0.364, "MED": 0.364, "LOW": 0.273 },
  "gaming_detection_active": true
}
```
This box is now ticked.

---

## Fix Priority Order (Day-of-Evaluation, **Remaining Work Only**)

> ✅ **HIGH-2 (rev 3), FATAL-3 (rev 4), FATAL-5 / NEW-2 (rev 5),
> NEW-1 / FATAL-4 (rev 6),** and now
> **NEW-3 / CRITICAL-2 / NEW-6 (this rev) are DONE.**
> README sanity script confirms all 11 cited numbers match JSON and all 10
> forbidden hand-edited tokens are gone. List renumbered.

| # | Issue | Fix Type | Est. Time | Blocking? |
|---|-------|----------|-----------|-----------|
| 1 | **NEW-4**: Add `_strategy_coordinated_fraud` + `_strategy_identity_fraud` | code, 1 file | 30 min | Medium — `--all-tasks` errors |
| 2 | **HIGH-4 / CF-1**: Convert variance warning → `raise RuntimeError` | code, 1 file | 5 min | No — but Part 4 contract |
| 3 | **NEW-5**: Reconcile component-name vocabulary | code, 2 files | 20 min | No — but visible in artifacts |
| 4 | **NEW-7**: Add discovery hooks for `distribution_shift_claim` | code, 2 files | 30 min | Medium — caps that task's evidence at 0.0 |
| 5 | **FATAL-2 Step 1**: Re-run training with bigger settings (use HF credits) | training | 30 min on A10G | Yes — lift flat components |
| 6 | **FATAL-2 Step 3**: Regenerate `component_shift_summary.json` | output of #5 | auto | Yes — drops contradiction with `training_summary.json` |

**Total remaining time: ~1 hr 25 min of work + 1 training run.** (was 1 hr 40 min before NEW-3 / CRITICAL-2 / NEW-6 closed)

> **Recommendation:** Do items 1–8 *before* spending any HF credits.
> All 8 are zero-compute logic/text fixes. Once the pipeline is provably
> correct end-to-end (run all `pytest`, `pre_validation_script`, and the
> 11-call `/stats` check), spend the credits on item 9 with confidence.

---

## Verification Checklist (Final)

Every item below must be `true` before submitting. Tick them in order; an
earlier failure invalidates later items.

### Live Environment
- [x] `/health` returns `{"status": "healthy"}` on the live HF Space
- [ ] `/tasks` returns all 5 task IDs on the live Space
- [ ] `/reset` with seed=7 vs seed=42 returns different `documents[0].content`
- [ ] `/step` with `deny_claim MED` returns higher reward than `approve_claim HIGH` on `contradictory_claim`
- [x] `/stats` after 11 episodes returns `episodes_recorded ≥ 11`, `gaming_detection_active: true` ← **HIGH-2 verified live 25 Apr 17:25 IST**
- [ ] `/rollout?task_id=contradictory_claim&seed=42` returns a non-empty trace ending in `done: true`

### Eval Artifacts
- [x] `reports/eval_report.json` is dated today, not 2026-04-03 ← **NEW-1 fix this revision** (`generated_at: 2026-04-25T13:37:28+00:00`)
- [x] **Live env confirms `evidence_quality = 1.0` for `contradictory_claim`** (FATAL-3 fix verified seed=42; 2026-04-25 17:50 IST). Now committed in `reports/eval_report.json` (`evidence_quality = 1.0` on all 5 contradictory_claim rows).
- [x] `reports/eval_report.json` has at least 2 distinct `variant_id` values across seeds ← **FATAL-4 fix this revision** (5 distinct: `{0, 1, 2, 3, 4}`)
- [x] `reports/eval_report.json` has different rewards for different tasks (not all 0.825) ← 3 distinct rewards: `{0.3966, 0.7497, 0.7625}`
- [ ] `reports/component_shift_summary.json` agrees with `reports/training_summary.json` on every common metric

### Training Artifacts
- [x] `reports/training_summary.json` shows `decision_accuracy after > before` (0.3333 → 0.6667)
- [ ] `reports/training_summary.json` shows at least 2 of 4 components improving (currently only 1)
- [ ] `docs/reward_curve.svg` has labeled axes and shows the curve going up
- [ ] `docs/component_shift.svg` shows a meaningful before/after delta (not flat)
- [ ] WandB run URL in README resolves to a real run with `eval/before/*` and `eval/after/*` keys logged

### Code & Tests
- [x] `pytest tests/envs/test_debatefloor_rubric.py -v` → **6 / 6 PASS** ← **NEW-2 / FATAL-5 fix (this revision)**
- [x] `pytest tests/test_calibration.py tests/envs/test_insurance_claim_reward_and_exploit.py tests/envs/test_debatefloor_rubric.py -v` → **49 / 49 PASS**
- [x] `train/train_minimal.py` imports `FastLanguageModel` from `unsloth`
- [x] `train/train_minimal.py` `reward_fn` calls `run_episode_via_http`
- [x] `app/environment.py` calls `record_episode_confidence` on every terminal action ← **HIGH-2 fix (commit `9f2d218`)**
- [ ] `inference_debatefloor.py` has `STRATEGIES` entry for all 5 task IDs
- [x] `inference_debatefloor.py` `flag_id`s in `_strategy_contradictory_claim` are in `expected_signals` ← **FATAL-3 fix this revision**
- [x] `inference_debatefloor.py` `_strategy_distribution_shift_claim` no longer flags signals it cannot discover ← **FATAL-3 fix this revision**

### YAML & Spec Compliance
- [x] `openenv.yaml` lists all 5 task IDs
- [ ] Every action in `openenv.yaml:action_space` is handled in `app/environment.py:_apply_action`
- [x] `server/app.py` is a real entry point, not a one-line re-export

### Submission Documents
- [x] README HF Space URL is live and serving (`https://aniketasla-debatefloor.hf.space`, SHA `402ef31bbbe0`, stage `RUNNING`)
- [ ] README WandB run URL resolves to the correct run (matches the JSON we ship)
- [ ] README Colab badge opens the correct notebook
- [x] README "Training reward" row matches numbers in `training_summary.json` ← **NEW-3 / CRITICAL-2 fix this revision** (`0.0453 → 0.3318`, all 11 cited numbers match JSON)
- [x] README install command uses `pip install -r ...` not the broken inline list ← **NEW-6 fix this revision** (sources `requirements.txt` + `train/requirements.txt`)
- [x] README links the writeup (`docs/HFBlogPost.md` — already linked)
- [x] Trained model is pushed to HF Hub and linked from README

---

## When to Use the HF Credits

**Not yet.** Items 1–4 above are zero-compute. They are now lower-impact
than the items already closed: the visible-to-judges risks (failing test,
contradictory README, broken install, stale eval_report) are all gone.

The `/stats`-empty risk (HIGH-2), the stale eval_report risk
(NEW-1 / FATAL-4), and the README contradiction risk
(NEW-3 / CRITICAL-2 / NEW-6) have all been removed. The remaining
visible-to-judges risk is the empty-strategy `coordinated_fraud` /
`identity_fraud` rows that error if a reviewer runs
`python inference_debatefloor.py --all-tasks`.

Burn the credits exactly once, on items 5–6, **after** items 1–4 are done
and a local 50-episode smoke training (T4) confirms all 4 component scores move.

The model choice (Qwen2.5-0.5B-Instruct) is correct for this submission and
should not be changed. A bigger model would invalidate the before/after delta
that the judging rubric explicitly looks for.

---

## Change Log

| Date (IST) | Revision | Notes |
|---|---|---|
| 25 Apr 17:00 | second pass | First-round fixes audited; 6 NEW issues uncovered; HIGH-2 still FAIL |
| 25 Apr 17:30 | third pass | **HIGH-2 → PASS** (code `9f2d218`, HF `402ef31bbbe0`); priority list renumbered; live `/stats` proof captured |
| 25 Apr 17:55 | fourth pass | **FATAL-3 → PASS**: contradictory_claim evidence_quality `0.0 → 1.0` and reward `0.518 → 0.7497` (live env, seed=42). NEW-7 added: distribution_shift_claim has no env-side discovery path for its expected_signals. Priority list renumbered (10 → 10 with FATAL-3 removed and NEW-7 added). |
| 25 Apr 18:25 | fifth pass | **NEW-2 → PASS** and **FATAL-5 → PASS**: replaced `tests/envs/test_debatefloor_rubric.py` with a 6-test suite that asserts the FATAL-5 contract (`obs.rubric_reward != obs.reward`). Live divergence proof: 0.428 vs 0.29 (Δ 0.138) for the original failing call. Full DebateFloor regression: **49/49 pass**. Priority list shrinks to 9 items. |
| 25 Apr 19:10 | sixth pass | **NEW-1 → PASS** and **FATAL-4 → PASS**: built `train/generate_eval_report.py` (the previously-referenced `pre_validation_script.py --output/--seeds/--tasks` flags never existed). Regenerated `reports/eval_report.json` + `.md` against the live HF Space using `inference_debatefloor.py:STRATEGIES` × seeds `[7, 11, 13, 19, 25]` (all 5 variant_ids) × 3 tasks → 15 rows. Numbers: 5 distinct variant_ids, 3 distinct rewards (`{0.3966, 0.7497, 0.7625}`), 10/15 rows with `evidence_quality > 0`, 100% completion, average reward `0.6363`. PLAN-prescribed invariant assertion runs clean. Priority list shrinks to 8 items. |
| 25 Apr 19:25 | **seventh pass (this revision)** | **NEW-3 → PASS**, **CRITICAL-2 → PASS**, **FATAL-2 storytelling-half → PASS**, and **NEW-6 → PASS**: rewrote README `Results` block, both plot captions, and Training Pipeline install command so every cited number is read directly from `training_summary.json` / `eval_report.json` and the install command sources `requirements.txt` + `train/requirements.txt`. Sanity script verifies 11/11 cited numbers match JSON, 10/10 forbidden hand-edited tokens (`-0.34`/`+0.83`/`~82%`/`~44%`/`41%`/`73%`/`trl>=0.9.0` plus unicode-minus variants) absent, 4/4 install-command checks pass. FATAL-2 re-training half (Step 1 + Step 3) still pending HF credits. Priority list shrinks to 6 items. |
