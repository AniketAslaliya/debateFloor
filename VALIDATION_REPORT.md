# Project Validation Report — DebateFloor vs HACKATHON_CONSTRAINTS.md

**Generated:** Saturday, April 25, 2026  
**Scope:** Validate the current repo state against every rule in `HACKATHON_CONSTRAINTS.md` and verify whether the fixes promised in `PLAN.md` have actually been applied.  
**Verdict:** Submission is largely on track but **5 fixes from PLAN.md are still partially or fully unfinished**, plus 4 new gaps not covered in the plan.

---

## Legend

- PASS — implemented and verified in code
- PARTIAL — fix applied but breaks a related contract (test, eval, or doc)
- FAIL — promised in PLAN.md but not actually fixed in code
- MISSING — required by HACKATHON_CONSTRAINTS.md, not addressed anywhere

---

## Section 1 — Status of every PLAN.md fix

| # | Issue from PLAN.md | Status | Evidence |
|---|---|---|---|
| FATAL-1 | Training loop never connects to env | PASS | `train/train_minimal.py` lines 128–166 implement `run_episode_via_http()`, called from `reward_fn` lines 277–290; `_start_env_server_if_needed()` ensures the server is up before `trainer.train()`. |
| FATAL-2 | Training summary shows 0.0 improvement | PARTIAL | `reports/training_summary.json` now shows `Decision accuracy: 0.3333 → 0.6667` and a real reward curve (`mean_start: 0.0453 → mean_end: 0.3318`). However: (a) `reports/component_shift_summary.json` was not regenerated and **still shows the old 0.0/−0.8 numbers**, contradicting `training_summary.json`; (b) README still claims `−0.34 → +0.83` which appears nowhere in the JSON. |
| FATAL-3 | `evidence_quality = 0.0` in all eval rows | FAIL | `inference_debatefloor.py` line 154 still raises `flag_id="procedure_mismatch"` for `contradictory_claim`, but that task's `expected_signals` is `["date_mismatch", "cost_inflation", "signature_mismatch", "prior_similar_claim"]` (`app/tasks.py` lines 200–204). `clustered_policy_broker` on line 213 is also fired against `distribution_shift_claim` whose expected signals are `["shared_repair_shop_far", "shared_emergency_contact", …]` (lines 308). Both flags are therefore dropped → evidence_quality stays 0.0. |
| FATAL-4 | `variant_id` always 0 in eval rows | FAIL | `reports/eval_report.json` was not re-generated — it is dated `2026-04-03`, predates every fix in PLAN.md, every row still has `variant_id: 0` and identical reward 0.825. The plan said "re-run after fix"; the file is the same as before. |
| FATAL-5 | Rubric is decorative (echoes env reward) | PARTIAL | `app/rubrics.py` was rewritten to add `_ReasoningQualityRubric` (independent process signal) and a 0.20 weight. **But `tests/envs/test_debatefloor_rubric.py` was NOT updated**: line 28 still asserts `obs.rubric_reward == pytest.approx(obs.reward)` and lines 29–39 expect old component names (`payout_accuracy`, `consistency_score`) that no longer exist in the rubric. The test is now broken AND it asserts the very property the fix was supposed to invalidate. |
| CRITICAL-1 | No Unsloth usage | PASS | `train/train_minimal.py` lines 72–79 import `FastLanguageModel` from `unsloth`; lines 583–599 use `FastLanguageModel.from_pretrained` + `get_peft_model`; line 682 uses `save_pretrained_merged(..., save_method="merged_16bit")`. `train/requirements.txt` line 12 lists `unsloth`. |
| CRITICAL-2 | Training vs eval reward labels mixed | PARTIAL | `wandb.init()` config is now labelled (`reward_type: env_http_reward`), `training_summary.json` records both `training_reward_curve` and `eval_reward_before/after` separately. **However README's "Mean reward" row (`−0.34 → +0.83`) does not match either of these numbers.** That row needs to be updated to reflect the actual JSON. |
| HIGH-1 | `coordinated_fraud` missing from `openenv.yaml` | PASS | `openenv.yaml` lines 61–75 add both `coordinated_fraud` and `identity_fraud`; `list_tasks_summary()` (`app/tasks.py` line 509) iterates the full `TASKS` dict so `/tasks` returns all 5. |
| HIGH-2 | Anti-gaming detector disabled across sessions | FAIL | `app/session_store.py` and `/stats` endpoint were created (PLAN-compliant). **But `app/environment.py` line 446–451 still passes `self._episode_history` (per-instance) to `compute_calibration_reward()`, never calling `record_episode_confidence` from `session_store`.** The global store exists but no code writes to it. The `/stats` endpoint will permanently report 0 episodes recorded. |
| HIGH-3 | `server/app.py` violates client/server separation | PASS | `server/app.py` is now a real entry point with a `serve()` function and `__main__` guard; not a one-line re-export. |
| HIGH-4 | Training loss 0.005 = model collapse | PARTIAL | Episodes increased from 100 → 300, epochs from 2 → 3, num_generations set to 6 — improvements per the plan. **But `training_loss` in the latest summary is still `0.0053`** — the change of dataset alone did not solve the symptom. The reward did rise (0.045 → 0.332), so some learning happened, but loss is still in the "warning" zone the plan called out. |
| MEDIUM-1 | reward_fn used keyword matching | PASS | Reward now comes exclusively from POST `/step` (resolved by FATAL-1). Keyword scoring kept only as `_score_completion_keyword` fallback when env unreachable. |
| MEDIUM-2 | WandB curve caption ambiguous | PASS | `save_training_artifacts()` lines 515–518 add the "training scalar is unbounded" annotation; figure title and y-axis label are explicit. README has a `Note on reward scale` block. |

---

## Section 2 — HACKATHON_CONSTRAINTS.md compliance check

### Part 1 — Minimum Requirements

| Rule | Status | Evidence / Gap |
|---|---|---|
| MR-1 — Use OpenEnv (latest) | PASS | `app/environment.py` line 7 imports `Environment` from `openenv.core.env_server.interfaces`; `openenv.yaml` declares `spec_version: 1`, `name`, `type`, `runtime: fastapi`, `app: app.main:app`, `port: 7860`. |
| MR-2 — Training MUST connect via HTTP | PASS | See FATAL-1 evidence. The "kill-switch test" (turn off env → training fails) would now hold: `_wait_for_env` raises `RuntimeError` after 15 retries. |
| MR-3 — Unsloth MUST be used | PASS | See CRITICAL-1 evidence. |
| MR-4 — Training evidence shows measurable improvement | PARTIAL | Only `Decision accuracy` improves (0.33 → 0.67). `Fraud detection` and `Evidence quality` are flat (0.33 → 0.33), `Calibration` actually drops (0.33 → 0.20). Three of four components do not improve in the artifact judges will read. Required artifacts (`docs/reward_curve.svg`, `docs/component_shift.svg`, `reports/training_summary.json`) all exist. WandB run URL is in README. |
| MR-5 — Writeup linked from README | PASS | README line 42 links to `docs/HFBlogPost.md`. |
| MR-6 — Hosted on HF Space | PASS (claimed) | README links to `https://huggingface.co/spaces/AniketAsla/debatefloor`. Liveness was not verified by this audit — see Section 4 manual checks. |

### Part 3 — Architecture Rules

| Rule | Status | Evidence / Gap |
|---|---|---|
| AR-1 — Training reward and eval reward never mixed | PARTIAL | Code separates them properly. README "Mean reward" row still mixes them (see CRITICAL-2). |
| AR-2 — Rubrics independent of env reward | PARTIAL | Rubric design is correct (reasoning_quality is independent). But the existing test still asserts equality with env reward (see FATAL-5). |
| AR-3 — YAML matches code exactly | PARTIAL | All 5 tasks now in YAML. **Action-space drift:** `openenv.yaml` lists `convene_debate_panel` and `verify_provider_registration`, but `inference_debatefloor.py` only ever calls a subset; the manifest also omits no actions. **Observation-space drift:** YAML lists `discovered_signals` and `metadata`-equivalent fields, but the `InsuranceClaimObservation` model should be cross-checked field-by-field (see Section 4). |
| AR-4 — `server/` owns server logic | PARTIAL | `server/app.py` is a proper entry point now, but `app/main.py` still owns the FastAPI instance and all routes. The "minimal" Option A from PLAN.md was chosen — acceptable but borderline; the deeper Option B was not done. |
| AR-5 — Anti-gaming works across sessions | FAIL | See HIGH-2. The fix was scaffolded but never wired in. |

### Part 4 — Common Failure Modes

| Failure mode | Status | Evidence / Gap |
|---|---|---|
| CF-1 — "Looks like training" (low reward variance) | PARTIAL | `train_minimal.py` lines 293–305 log variance to WandB and warn when `variance < 0.01`. **The required hard guard `raise RuntimeError(...)` is NOT in the code** — the constraint says "raise" but the implementation only `print`s. |
| CF-2 — `evidence_quality` always 0.0 | FAIL | Same root cause as FATAL-3; the scripted agent's `flag_id`s are still wrong. |
| CF-3 — `variant_id` always 0 | UNKNOWN/FAIL | Server-side code does pass `seed` through to `build_runtime_task` correctly. The eval script (`pre_validation_script.py` / `inference_debatefloor.py`) does pass `seed` in the JSON body. **But the committed `reports/eval_report.json` is stale (dated 2026-04-03) and has not been regenerated**, so the fix cannot be verified from artifacts. |
| CF-4 — Same reward for every task | PARTIAL | New `component_eval_detailed.json` shows three distinct reward bands (clean=0.7625, contradictory=0.8113, distribution_shift=0.4001). Stale `eval_report.json` still shows constant 0.825 — must be deleted or regenerated before submission. |
| CF-5 — QLoRA save corruption | PASS | `save_pretrained_merged` is used. |

### Part 5 — Pre-Submission Checklist gaps

The following items from `HACKATHON_CONSTRAINTS.md` Part 5 are not verifiably YES today:

- [ ] `evaluate component_shift.svg` shows a *meaningful* before/after difference — current chart shows 1 of 4 components moved meaningfully.
- [ ] `reports/eval_report.json` has `evidence_quality > 0.0` for at least one row — fails (stale).
- [ ] `reports/eval_report.json` has different `variant_id` values across seeds — fails (stale).
- [ ] `pre_validation_script.py` exits with code 0 against the live Space — not run today.
- [ ] Reward variance > 0.01 per batch — only a soft warning, no hard guard.
- [ ] `decision_accuracy > 0.0` after training — true (0.6667), but other 3 components do not improve.

---

## Section 3 — Issues found that PLAN.md did not catch

### NEW-1 — Stale `reports/eval_report.json` and `reports/eval_report.md`
Both files are dated 2026-04-03 (3 weeks old) and contain the very `variant_id=0` / `evidence_quality=0.0` rows the plan was supposed to fix. They *override* the newer `reports/component_eval_detailed.json` for any reviewer who searches the canonical filename `eval_report.json`.

**Fix:** Either delete these two files or regenerate them via `pre_validation_script.py --base-url <live-space-url>` and commit.

### NEW-2 — `tests/envs/test_debatefloor_rubric.py` is broken by the FATAL-5 fix
After the rubric was made independent, the test still:
- Asserts `obs.rubric_reward == pytest.approx(obs.reward)` (the very thing the fix invalidates).
- Expects component keys `payout_accuracy` and `consistency_score` that the new rubric does not produce.

If a reviewer runs `pytest tests/envs/test_debatefloor_rubric.py` it will fail. This is much worse than no test.

**Fix:**
```python
# In tests/envs/test_debatefloor_rubric.py
assert 0.0 <= obs.rubric_reward <= 1.0
assert "reasoning_quality" in obs.rubric_components
# Independent rubric MAY differ from env reward — do not assert equality
```
And update the expected key set to match `app/rubrics.py:component_scores()`.

### NEW-3 — README results table contradicts the actual JSON
README (lines 48–54):
> Mean reward: −0.34 → +0.83  
> HIGH-confidence episodes: ~82% → ~44%  
> Debate panel convened (hard task): 41% → 73%

None of these numbers appear in `reports/training_summary.json` or `reports/component_shift_summary.json`. The actual training scalar moved 0.0453 → 0.3318 (training_summary.json line 13). HIGH-confidence rate is not measured anywhere. Debate-panel convene rate is not measured anywhere.

**Fix:** Replace the table with values that exist in committed JSON, or add the metrics to the eval pipeline so the table becomes truthful. The judging criterion "Showing Improvement in Rewards" requires verifiable evidence; right now the headline numbers are unverifiable.

### NEW-4 — `inference_debatefloor.py` and code/UI drift on tasks
`inference_debatefloor.py` defines `STRATEGIES` for only 3 tasks (`clean_claim`, `contradictory_claim`, `distribution_shift_claim`) — `coordinated_fraud` and `identity_fraud` have no scripted policy even though they are now in the YAML. Running `--all-tasks` will print `[ERROR] No strategy for task 'coordinated_fraud'`.

**Fix:** Add `_strategy_coordinated_fraud` and `_strategy_identity_fraud` with correct `flag_id`s, register them in `STRATEGIES`.

### NEW-5 — `app/rubrics.py` `component_scores()` keys ≠ those used by `_weights`
- `_weights` keys: `fraud_detection`, `decision_accuracy`, `calibration_score`, `evidence_quality_score`, `efficiency_score`, `reasoning_quality`.
- `component_scores()` returns the same six PLUS `penalty` and `total`.
- The test (`test_debatefloor_rubric.py`) expects `payout_accuracy` and `consistency_score` — totally different vocabulary.

**Fix:** Pick one canonical set of component names and propagate it to: rubric, environment-attached `rubric_components`, eval scripts, and tests.

### NEW-6 — README "Quick Start" install command is missing key deps
README line 238:
```
pip install trl>=0.9.0 transformers peft accelerate datasets wandb matplotlib
```
This omits `unsloth`, `requests` (used by training), and pins TRL to 0.9 while the train script imports `GRPOConfig` (introduced in TRL 0.10+). A reviewer running the README literally will get `ImportError`.

**Fix:** Replace with `pip install -r train/requirements.txt`.

---

## Section 4 — Recommended verification commands (run before submitting)

Run these in order; each must pass.

```bash
# 1. Manifest validates
openenv validate .

# 2. Local environment serves and is healthy
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 7860 &
sleep 5
curl -s http://localhost:7860/health     # expect {"status":"healthy", ...}
curl -s http://localhost:7860/tasks      # expect 5 task_ids

# 3. Variant IDs differ across seeds
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"contradictory_claim","seed":7}'  | jq .observation.metadata.variant_id
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"contradictory_claim","seed":42}' | jq .observation.metadata.variant_id
# Must print two DIFFERENT integers.

# 4. Reward differs for good vs bad action on contradictory_claim
#   good = deny_claim MED, bad = approve_claim HIGH
#   reward(good) - reward(bad) > 0.3 (CF-4 invariant)

# 5. Existing tests all pass — fix broken rubric test first
pytest tests/envs/test_debatefloor_rubric.py -q

# 6. Stale eval_report regenerated against live space
python pre_validation_script.py --base-url https://huggingface.co/spaces/AniketAsla/debatefloor

# 7. Anti-gaming actually records episodes
for i in 1 2 3 4 5 6 7 8 9 10 11; do
  SID=$(curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
    -d "{\"task_id\":\"clean_claim\",\"seed\":$i}" | jq -r .session_id)
  curl -s -X POST http://localhost:7860/step -H "Content-Type: application/json" \
    -d "{\"action\":{\"action_type\":\"approve_claim\",\"confidence\":\"HIGH\"},\"session_id\":\"$SID\"}" > /dev/null
done
curl -s http://localhost:7860/stats     # episodes_recorded should be ≥ 11
```

If step 7 returns `episodes_recorded: 0`, HIGH-2 is unfixed (matches Section 1).

---

## Section 5 — Prioritised fix list (smallest-risk-first)

| # | Fix | File(s) | Time | Why it matters |
|---|---|---|---|---|
| 1 | Wire `record_episode_confidence` in `environment.py` | `app/environment.py` line 451 | 10 min | Closes HIGH-2 / AR-5; makes `/stats` non-zero; unblocks judges' "innovation" claim. |
| 2 | Fix `flag_id`s in `inference_debatefloor.py` | `inference_debatefloor.py` lines 154, 213 | 15 min | Closes FATAL-3 / CF-2; unblocks `evidence_quality > 0`. |
| 3 | Update `tests/envs/test_debatefloor_rubric.py` | tests file | 15 min | Test currently fails; contradicts FATAL-5 fix. |
| 4 | Delete or regenerate `reports/eval_report.json` + `.md` | `reports/` | 10 min | Closes FATAL-4 / CF-3; removes the stale 0.825 / variant_id=0 evidence. |
| 5 | Re-write README results table to match actual JSON | `README.md` lines 48–54 | 15 min | Closes CRITICAL-2 narrative gap; protects 30% storytelling score. |
| 6 | Fix README install line to reference `train/requirements.txt` | `README.md` line 238 | 2 min | Stops reviewer reproduction error. |
| 7 | Add `_strategy_coordinated_fraud` + `_strategy_identity_fraud` | `inference_debatefloor.py` | 30 min | Closes NEW-4; aligns inference with YAML. |
| 8 | Convert `print` warning → `raise RuntimeError` for variance < 0.01 | `train/train_minimal.py` line 296 | 5 min | CF-1 hard guard required by HACKATHON_CONSTRAINTS Part 4. |
| 9 | Re-run training to lift `Fraud detection` and `Evidence quality` above 0.33 | training pipeline | 30 min on T4 | Closes MR-4 partial. Otherwise judges see 3 of 4 components flat. |
| 10 | Regenerate `component_shift_summary.json` with current numbers | `reports/` | 5 min | Removes contradiction with `training_summary.json`. |

**Total estimated time: ~2 hours of focused work.**

---

## Section 6 — One-line summary per requirement

```
MR-1 OpenEnv subclass + manifest         PASS
MR-2 Training over HTTP                  PASS
MR-3 Unsloth in training                 PASS
MR-4 Measurable improvement              PARTIAL — 1 of 4 components moves
MR-5 Writeup linked                      PASS
MR-6 HF Space hosted                     PASS (link present, liveness unverified)

AR-1 Train vs eval reward separation     PARTIAL — README mixes them
AR-2 Independent rubric                  PARTIAL — code OK, test asserts opposite
AR-3 YAML == code                        PARTIAL — tasks aligned, action/obs drift unaudited
AR-4 server/ owns server logic           PARTIAL — minimal compliance only
AR-5 Anti-gaming cross-session           FAIL  — code path never invoked

CF-1 Reward variance hard guard          PARTIAL — warn-only, no raise
CF-2 evidence_quality > 0                FAIL  — wrong flag_ids in scripted agent
CF-3 variant_id varies                   UNKNOWN — eval not re-run
CF-4 Reward differs across tasks         PARTIAL — true in new file, false in canonical eval_report.json
CF-5 QLoRA save                          PASS
```

**Bottom line:** All five `FATAL-` items had code attempts; **FATAL-3 is unfixed in code** and **FATAL-4 / FATAL-2 are unfixed in committed artifacts**. The `tests/envs/test_debatefloor_rubric.py` regression is the single highest-value cleanup — a failing test in a public repo undermines every other claim. Fix items 1–6 in the prioritised list and the submission moves from PARTIAL to FULL compliance.
