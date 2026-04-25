# Hackathon Constraints — Non-Negotiable Rules

> This file exists because a technically impressive system was previously built
> that violated the most basic submission requirements of the OpenEnv Hackathon.
> Every rule below is derived from a real mistake that was made on this project.
>
> **Before you write any code, read this file.**
> **Before you call any task done, re-read this file.**

---

## What Went Wrong (And Why This File Exists)

Components were built in isolation — each one looked correct — but the whole system
was never validated against the judging PDF. The result:

- Training never called the environment. (FATAL)
- Unsloth was not used. (Disqualifying — minimum requirement)
- Training artifacts showed 0.0 improvement. (Main evidence criterion = failed)
- The YAML manifest disagreed with the code. (Judges read the YAML first)
- The rubric just echoed the environment's own reward. (Architecturally wrong)

None of these were bugs. They were omissions. Each task was completed as asked,
but nobody stepped back and asked: *"Does the whole thing actually satisfy the requirements?"*

That is what this file is for.

---

## Part 1 — Minimum Requirements (Non-Negotiable)

These come directly from the judging PDF. If ANY of these are missing,
the submission is at serious disadvantage. Verify all of them before considering
any feature complete.

### MR-1 — Use OpenEnv (latest release)

- The environment MUST subclass `Environment` from `openenv.core.env_server.interfaces`
- `reset()`, `step()`, `state` must follow the Gym-style API exactly
- `openenv.yaml` MUST exist at repo root with valid `spec_version`, `name`, `type`, `runtime`, `app`, `port`
- The environment MUST be deployable as a Hugging Face Space

**Your check:** Before finishing any environment work, run:
```bash
openenv validate .
```
If the CLI is not available, manually verify openenv.yaml has all required fields.

---

### MR-2 — The training script MUST connect to the live environment via HTTP

**This is the most important rule in this file.**

The training loop MUST:
1. Call `/reset` on the running environment server
2. Submit actions via `/step`
3. Read `reward` from the `/step` HTTP response
4. Pass that reward to the GRPO trainer

**What you must NEVER do:**
- Generate episodes from Python code and build a static `Dataset`
- Compute reward by string-matching the model's completion text
- Call internal Python functions (like `training_reward()`) directly from the training loop
- Skip the HTTP server entirely

**The test:** Can you run the training script with the environment server turned OFF?
If yes → the training is not connected to the environment. This is wrong.

**Template rollout function you must always use:**
```python
def run_episode_via_http(task_id: str, model, tok, base_url: str) -> float:
    r = requests.post(f"{base_url}/reset", json={"task_id": task_id, "seed": random.randint(0, 9999)})
    session_id = r.json()["session_id"]
    # generate completion from model
    # parse decision + confidence
    # submit via /step
    step_r = requests.post(f"{base_url}/step", json={"action": action, "session_id": session_id})
    return float(step_r.json()["reward"])
```

---

### MR-3 — Unsloth MUST be used in the training script

The hackathon stack is: **TRL + Unsloth + OpenEnv**. Unsloth is not optional.

```python
# This import MUST appear in train_minimal.py
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=512,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)
```

**Saving:** NEVER do `model.save_pretrained()` on a QLoRA model directly.
Always use `model.save_pretrained_merged(..., save_method="merged_16bit")`.

`train/requirements.txt` MUST include `unsloth`.

---

### MR-4 — Training evidence MUST show measurable improvement

The judges look for:
- A reward curve that goes UP over training steps
- Before-vs-after numbers on at least one metric
- Both saved to the repo as committed files (not just in a Colab cell or a deleted WandB run)

**Your check before finishing training work:**
- Open `reports/training_summary.json`
- Check `component_shift.after` values
- If `decision_accuracy` is 0.0 after training → training is broken, do not commit

**Required files that MUST exist and be non-trivial:**
- `reports/training_summary.json` — with real before/after component scores
- `docs/reward_curve.svg` — with labeled axes, reward going up
- `docs/component_shift.svg` — before/after bar chart
- WandB run URL that resolves to a real run

---

### MR-5 — A short writeup MUST exist and be linked from README

Either:
- A mini-blog post on Hugging Face (preferred)
- OR a YouTube video < 2 minutes
- OR a short slide deck

The README MUST have a direct link to whichever one exists.
Do not consider the submission complete until this link is in the README.

---

### MR-6 — The environment MUST be hosted on a Hugging Face Space

- The Space URL must be in the README
- The Space must respond to `/health` with `{ "status": "healthy" }`
- `openenv.yaml` must be present in the Space repo

---

## Part 2 — Judging Criteria Weights (Optimize For These)

| Criterion | Weight | What it actually means |
|-----------|--------|------------------------|
| Environment Innovation | 40% | Is it novel? Does it test agent behavior in a new way? |
| Storytelling | 30% | Can a non-technical person understand the README and demo? |
| Showing Improvement in Rewards | 20% | Quantitative before/after evidence that the model learned |
| Reward & Training Pipeline | 10% | Is reward coherent? Does the pipeline produce real improvement? |

### On the 40% criterion — Innovation

Judges have seen chess, snake, tic-tac-toe, and grid-world clones.
To score well on innovation, the environment must:
- Test something an LLM currently cannot do well
- Exist in an underexplored RL/LLM training domain
- Have a reward function that captures something hard to measure cleverly

When proposing or building any feature, ask yourself:
*"Does this make the environment more novel, or is it just adding complexity?"*

### On the 30% criterion — Storytelling

Storytelling is judged on the README and the demo, not the code.

Always ensure:
- README answers: what capability gap? what does the agent do? what changed after training?
- A reviewer can read the README in 3–5 minutes and want to try the environment
- The demo shows: baseline attempt → reward output → trained attempt → improvement

### On the 20% criterion — Showing Improvement

This criterion fails silently. You can build something that "looks like training"
but produces 0.0 improvement because the reward function has a bug.

**Always run this sanity check after any training change:**
```python
# Does the reward function return different values for good vs bad actions?
good_reward = reward_fn(["DECISION: deny_claim\nCONFIDENCE: MED\nREASON: date mismatch found"], ...)
bad_reward = reward_fn(["DECISION: approve_claim\nCONFIDENCE: HIGH\nREASON: looks fine"], ...)
assert good_reward[0] > bad_reward[0], f"Reward function is broken: {good_reward} vs {bad_reward}"
```

If this assertion fails, training will produce 0.0 improvement no matter how long it runs.

---

## Part 3 — Architecture Rules You Must Never Violate

### AR-1 — Training reward and evaluation reward must never be mixed

- Training reward: simple unbounded scalar, optimized for gradient stability
- Evaluation reward: multi-component, clamped to [0, 1], used for reporting only
- These are DIFFERENT NUMBERS. Never log one as if it were the other.
- In WandB, log them as separate keys: `train/reward` and `eval/reward`
- In README, label which is which

### AR-2 — Rubrics must be independent of the environment's own reward

The rubric must NOT just read `observation.reward_breakdown` and re-weight it.
That is not a rubric — it is a re-labeling.

A rubric must evaluate something the environment reward does not already measure.
Valid examples:
- Reasoning quality (does `action.reasoning` cite specific evidence?)
- Step diversity (is the agent exploring or looping?)
- Format compliance (does the output follow the required schema?)

**Test:** If `obs.rubric_reward == obs.reward` for every possible observation,
the rubric is decorative and must be rewritten.

### AR-3 — The YAML manifest must match the code exactly

`openenv.yaml` is what judges read first. It must be the source of truth.

- Every task in `app/tasks.py` MUST appear in the `tasks:` section of `openenv.yaml`
- Every action in the `action_space:` section MUST be handled in `_apply_action()`
- Every field in `observation_space:` MUST exist in the Observation model

After adding any task or action to code, update `openenv.yaml` in the same commit.

### AR-4 — The server module must own server logic

`server/app.py` must NOT be a one-line re-export from `app/`.
The `server/` module is the deployment boundary. Business logic lives in `app/`.
Clients must never import from `server/` internals.

### AR-5 — Anti-gaming must work across sessions, not per-session

If concurrent sessions are supported (`SUPPORTS_CONCURRENT_SESSIONS: true`),
any cross-episode detection (anti-gaming, confidence tracking) must use a
shared, thread-safe store — not instance variables on the environment object.

A per-instance counter that resets every session is not anti-gaming.
It is security theater.

---

## Part 4 — Common Failure Modes to Watch For

### CF-1 — The "looks like training" failure

Symptom: training script runs without errors, loss decreases, reward curve exists.
Actual state: reward function returns near-constant values → advantage is ~0 → no learning.

**Check:** reward variance per batch must be > 0.01. If variance is near zero, GRPO learns nothing.

```python
import statistics
variance = statistics.variance(batch_rewards)
if variance < 0.01:
    raise RuntimeError(f"Reward variance too low ({variance:.4f}). Training will not converge.")
```

### CF-2 — The "evidence quality is always 0.0" failure

Symptom: eval report shows all rows with `evidence_quality: 0.0`.
Cause: the scripted agent raises fraud flags with wrong `flag_id` values (not in `expected_signals`),
or raises flags before calling the investigative actions that discover them.

**Rule:** `flag_fraud_signal` must only be called AFTER `validate_document` or equivalent.
The flag_id must exactly match an entry in `expected_signals` for the current task.

**Check after any change to the inference script:**
```python
assert any(row["evidence_quality"] > 0 for row in eval_report["rows"]), \
    "Evidence quality is 0.0 for all tasks — scripted agent has wrong flag_ids"
```

### CF-3 — The "variant_id always 0" failure

Symptom: eval report shows `variant_id: 0` for all seeds.
Cause: the eval script is not passing `seed` in the POST body.

**Rule:** seed MUST be in the JSON body of `/reset`, not a query parameter.
```python
# CORRECT
requests.post(f"{base_url}/reset", json={"task_id": task_id, "seed": seed})

# WRONG — seed will be ignored
requests.get(f"{base_url}/reset?seed={seed}")
```

### CF-4 — The "same reward for every task" failure

Symptom: eval report shows the same reward (e.g. 0.825) for all tasks and seeds.
Cause: the agent always takes the same scripted actions regardless of task_id.

**Rule:** Each task must produce meaningfully different rewards for different strategies.
The reward delta between a good strategy and a bad strategy must be > 0.3.

### CF-5 — The "model save corruption" failure

Symptom: trained model loads but produces worse outputs than the base model.
Cause: QLoRA adapters were merged naively into a 4-bit base model.

**Rule:** Always use `model.save_pretrained_merged(..., save_method="merged_16bit")`.
Test inference immediately after saving — do not leave this until submission day.

---

## Part 5 — Pre-Submission Validation Checklist

You must be able to answer YES to every question before the submission is complete.

### Environment
- [ ] Does `openenv validate .` pass without errors?
- [ ] Does `/health` return `{ "status": "healthy" }` on the live Space?
- [ ] Does `/reset` with seed=7 return different document content than seed=42?
- [ ] Does `/step` with a correct action return higher reward than an incorrect action?
- [ ] Are all tasks in `app/tasks.py` listed in `openenv.yaml`?
- [ ] Are all actions in `_apply_action()` listed in `openenv.yaml` action_space?

### Training
- [ ] Does the training script call `/reset` and `/step` HTTP endpoints?
- [ ] Does the training script import from `unsloth`?
- [ ] Does `reports/training_summary.json` show `decision_accuracy > 0.0` after training?
- [ ] Is reward variance > 0.01 per batch during training?
- [ ] Is the model saved using `save_pretrained_merged`?
- [ ] Does the WandB run URL in README resolve to a real run?

### Evidence
- [ ] Does `reports/eval_report.json` have `evidence_quality > 0.0` for at least one row?
- [ ] Does `reports/eval_report.json` have different `variant_id` values across seeds?
- [ ] Does `docs/reward_curve.svg` have labeled axes and a curve that goes up?
- [ ] Does `docs/component_shift.svg` show a meaningful before/after difference?

### Submission artifacts
- [ ] Is the HF Space URL in the README?
- [ ] Is the WandB run URL in the README?
- [ ] Is the Colab notebook badge in the README?
- [ ] Is the writeup (blog/video/slides) linked from the README?
- [ ] Does `pre_validation_script.py` exit with code 0 against the live Space?
- [ ] Is the trained model pushed to HF Hub and linked from the README?

---

## Part 6 — Questions to Ask Before Building Anything New

Before writing any code for a new feature, answer these:

1. **Does this feature connect to the live environment, or does it bypass it?**
   If it bypasses the environment, it should not exist.

2. **Does this feature produce evidence that judges can verify?**
   If the evidence only lives in a Colab cell or an uncommitted local file, it does not count.

3. **Does this feature change what `openenv.yaml` must say?**
   If yes, update the YAML in the same commit as the code change.

4. **Does this feature affect the reward surface?**
   If yes, run the reward variance sanity check and the before/after comparison.

5. **Does this feature appear in the README?**
   If it is important enough to build, it is important enough to explain.
