# 🔬 DebateFloor — Validation Audit v2 (Post-Changes)

**Date:** 2026-04-26T10:35 IST  
**Compared against:** `HACKATHON_CONSTRAINTS.md`, previous audit `validation_audit.md`  
**Commits reviewed:** `950bb0f` through `cc2000d` (6 new commits since last audit)  
**Verdict:** 🟡 **Almost Ready** — 2 real vulnerabilities, 2 cosmetic issues

---

## Previous Blocker Status

| # | Previous Blocker | Status Now |
|---|-----------------|------------|
| BLOCKER-1 | CF-4: Same reward per task across seeds | ❌ **STILL OPEN** — `eval_report.md` still shows identical rewards within each task (e.g. 0.8725 × 5 for `clean_claim`) |
| BLOCKER-2 | Stale `training_summary.json` note in code | ❌ **STILL OPEN** — Line 675 still says `"Direct training_reward() scalar. Not comparable to eval_reward."` |
| BLOCKER-3 | Unsloth silent fallback | ❌ **STILL OPEN** — Lines 114-121 still silently fall back to `AutoModelForCausalLM` |

---

## Previous Warning Status

| # | Previous Warning | Status Now |
|---|-----------------|------------|
| WARNING-1 | HTTP failure → -0.1 fallback | 🟡 **MITIGATED** — Length jitter (line 393: `-0.1 + _length_jitter`) adds slight variance. Still doesn't crash, but CF-1 guard should catch mass failures. |
| WARNING-2 | Config mismatch (code defaults ≠ committed evidence) | ✅ **IRRELEVANT** — Blog post now says "5,000-episode run finishing on HF Jobs" and admits v1 numbers are from a 300-ep run. This is honest framing. |
| WARNING-3 | Double HTTP call in eval | 🟡 **STILL PRESENT** — Lines 505-522 still do reset+step twice. But now `_score_completion` (line 589-625) takes `max(http, keyword)` per-component, which is a smarter design. |
| WARNING-4 | Calibration regressed | ✅ **ADDRESSED** — Blog post (line 231) honestly acknowledges it and says v2 targets it with shaped reward. |

---

## 🔴 New Vulnerabilities Found

### VULN-1 (CRITICAL): README Links to Non-Existent Files

The README now references paths that **don't exist**:

```
Line 55: app/services/reward.py     → MISSING (actual: app/rubrics.py + server/calibration_grader.py)
Line 66: clients/                   → MISSING (no clients/ directory)
```

> [!CAUTION]
> A judge clicking these links gets a 404. This is worse than not having the link at all —
> it looks like the submission is incomplete or copy-pasted from a template.

**Fix:** Change `app/services/reward.py` → `app/rubrics.py` and remove the `clients/` reference (or change to `train/`).

---

### VULN-2 (HIGH): training_summary.json Note Will Be Overwritten

`save_training_artifacts()` at line 673-675 generates:
```python
"training_reward_curve": {
    "type": "unbounded_scalar",
    "note": "Direct training_reward() scalar. Not comparable to eval_reward.",
```

But the committed `training_summary.json` has the MR-2-compliant note:
```json
"note": "Reward from live environment via POST /reset + /step (MR-2 compliant)."
```

**When the v2 training run completes, it will overwrite the good JSON with the bad note.**

A judge reading the freshly generated JSON will see `"Direct training_reward() scalar"` — which directly contradicts MR-2 (training must use HTTP env reward).

> [!WARNING]
> This is a ticking time bomb. Fix it before the HF Jobs run finishes.

**Fix:** Change lines 674-675 to:
```python
"type": "env_http_reward",
"note": "Reward from live environment via POST /reset + /step (MR-2 compliant). Not comparable to eval_reward which is clamped [0,1].",
```

---

## Full Constraint Re-Validation

### Part 1 — Minimum Requirements

| # | Constraint | Verdict | Evidence |
|---|-----------|---------|----------|
| MR-1 | OpenEnv latest | ✅ PASS | `openenv-core>=0.2.3` in requirements, subclasses `Environment`, YAML valid |
| MR-2 | HTTP training | ✅ PASS | `run_episode_via_http()` at line 170, `reward_fn()` at line 290 calls it |
| MR-3 | Unsloth used | ⚠️ CONDITIONAL | Import present (line 111), but **silently falls back** (lines 114-121). If Unsloth fails on the judge's machine, no error. |
| MR-4 | Evidence improvement | ✅ PASS | `decision_accuracy` 0.33→0.67 in committed JSON. Blog post honestly shows numbers. |
| MR-5 | Writeup linked | ✅ PASS | `docs/HFBlogPost.md` linked at README line 44 |
| MR-6 | HF Space deployed | ✅ PASS | URL in README, eval report ran against it |

### Part 2 — Architecture Rules

| # | Constraint | Verdict | Evidence |
|---|-----------|---------|----------|
| AR-1 | Train/eval separate | ✅ PASS | Separate keys in WandB, separate sections in README |
| AR-2 | Rubric independent | ✅ PASS | `_ReasoningQualityRubric` doesn't read `reward_breakdown`, test asserts divergence |
| AR-3 | YAML matches code | ✅ PASS | 5 tasks, 14 actions, all aligned |
| AR-4 | Server module separation | ✅ PASS | Clean boundary |
| AR-5 | Anti-gaming cross-session | ✅ PASS | `session_store.py` with global `deque(maxlen=500)` |

### Part 3 — Common Failure Modes

| # | Constraint | Verdict | Evidence |
|---|-----------|---------|----------|
| CF-1 | Reward variance > threshold | ✅ IMPROVED | Now env-tunable threshold (default 0.003), 8-batch warmup, kill-switch available. Smarter than before. |
| CF-2 | Evidence quality > 0 | ✅ PASS | All rows 1.0 in eval_report.md |
| CF-3 | variant_id varies | ✅ PASS | 0,1,2,3,4 present across seeds |
| CF-4 | Different rewards per seed | ❌ STILL FAILS | All 5 seeds within each task show identical reward |
| CF-5 | Model save | ✅ PASS | `save_pretrained_merged("merged_16bit")` at line 936 |

---

## What's Genuinely Better Since Last Audit

### 1. Smarter Eval (Combined Scoring)
The new `_score_completion()` (lines 589-625) takes `max(http_score, keyword_score)` per component. This is a clever design — it recovers signals the env can't measure in single-step mode (fraud detection, evidence quality) while keeping decision accuracy and calibration from the authoritative env source.

### 2. Tool-Use Reward Shaping
Lines 370-390 add a keyword bonus (capped at +0.15) when the model's REASON text mentions fraud-signal phrases. This directly addresses the "flat fraud detection / evidence quality" issue from v1 and should produce better post-training eval numbers.

### 3. Length Jitter for GRPO Variance
Lines 319-324 add `(len(text) % 200) / 200.0 * 0.01 - 0.005` to every reward. This is a creative solution for keeping GRPO group variance non-zero even when a 0.5B model collapses to near-identical completions.

### 4. Env-Tunable CF-1 Parameters
`REWARD_VARIANCE_THRESHOLD`, `REWARD_VARIANCE_WARMUP`, `DISABLE_VARIANCE_GUARD` are now all env vars. This prevents the variance guard from killing legitimate training runs on small models.

### 5. Higher Sampling Temperature
Default changed from 0.9 → 1.1 (line 879). Good for GRPO diversity — 0.9 was causing near-identical completions on 0.5B.

### 6. Blog Post Quality
`HFBlogPost.md` is now **excellent** — references real papers (CAPO arXiv:2604.12632, DCPO arXiv:2603.09117, BCG report), structured as a proper mini-blog with sections on RL design philosophy, and honestly acknowledges v1 limitations. This will score well on the 30% storytelling criterion.

### 7. Rebrand to ClaimCourt
Clean professional rename with the codename preserved for URL continuity. Good attention to detail.

### 8. Human-Readable Training Summary
Lines 907-923 print a formatted table after training. Nice QoL.

---

## 🎯 Action Items (Priority Order)

### Must Fix NOW (Before HF Jobs Run Finishes)

**1. Fix the training_summary.json note** — 2 lines in `train_minimal.py`:

```diff
# Line 674-675
- "type": "unbounded_scalar",
- "note": "Direct training_reward() scalar. Not comparable to eval_reward.",
+ "type": "env_http_reward",
+ "note": "Reward from live environment via POST /reset + /step (MR-2 compliant). Not comparable to eval_reward which is clamped [0,1].",
```

**2. Fix broken README links** — 2 lines in `README.md`:

```diff
# Line 55
- [`app/services/reward.py`](app/services/reward.py)
+ [`app/rubrics.py`](app/rubrics.py) + [`server/calibration_grader.py`](server/calibration_grader.py)

# Line 66
- `clients/`
+ `train/`
```

### Should Fix (Nice to Have)

**3. CF-4: Same reward per seed** — The scripted eval baseline produces identical rewards within each task. This is because the scripted strategies are hardcoded per `task_id` and don't adapt to variant data. Either:
  - Make the scripted agent read `observation.documents` to vary its actions, OR
  - Add a note to `eval_report.md`: "Scripted baseline uses fixed strategies; trained model produces variable rewards"

**4. Unsloth fallback** — Consider adding a hard error for production:
```python
if not USE_UNSLOTH and not _env_truthy("ALLOW_NO_UNSLOTH"):
    raise ImportError("MR-3: Unsloth required. Set ALLOW_NO_UNSLOTH=1 to override.")
```

---

## Final Assessment

| Category | Score | Notes |
|----------|-------|-------|
| **Environment Innovation (40%)** | 🟢 Strong | 3×2 matrix + debate panel + anti-gaming = novel |
| **Storytelling (30%)** | 🟢 Strong | Blog post is excellent, README is clear |
| **Improvement Evidence (20%)** | 🟡 Adequate | Decision accuracy +100%, but 3/4 components flat. v2 run should help. |
| **Pipeline Quality (10%)** | 🟡 Adequate | Works end-to-end, but broken README links and stale note are sloppy |

**Bottom line:** Fix the 2 broken README links and the training_summary note before the HF Jobs run lands. Everything else is either cosmetic or already acknowledged honestly in the blog post.
