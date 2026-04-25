# DebateFloor — Pre-Evaluation Fix Plan

**Status:** Pre-submission hardening  
**Deadline:** April 25–26 2026 Grand Finale  
**Priority order:** FATAL → CRITICAL → HIGH → MEDIUM

Each issue below states: what a Meta judge will see, why it fails, and the exact fix.

---

## Table of Contents

1. [FATAL-1] Training loop never connects to the environment
2. [FATAL-2] Training evidence shows zero improvement (training_summary.json)
3. [FATAL-3] Evidence quality is 0.0 in all eval rows
4. [FATAL-4] variant_id is always 0 — procedural generation appears broken
5. [FATAL-5] Rubric is decorative — it echoes the environment's own reward
6. [CRITICAL-1] No Unsloth usage — violates minimum submission requirements
7. [CRITICAL-2] Training reward and eval reward use completely different math
8. [HIGH-1] coordinated_fraud task missing from openenv.yaml
9. [HIGH-2] Anti-gaming detector is effectively disabled during training
10. [HIGH-3] server/app.py violates client/server separation principle
11. [HIGH-4] Training loss (0.005) indicates model collapse or no real gradient
12. [MEDIUM-1] reward_fn uses keyword string matching instead of env signals
13. [MEDIUM-2] WandB curve caption says "training signal, not evaluation score" — clarify or remove the confusion
14. [Quick wins] README, plots, final checklist

---

## FATAL-1 — Training loop never connects to the environment

### What the judge sees
`train/train_minimal.py` generates episodes from `server.claim_generator` directly in Python, builds a static `Dataset`, and passes it to `GRPOTrainer`. The `/reset` and `/step` HTTP endpoints are **never called** during training. The judging criteria explicitly states:

> "Your training loop should connect to your environment (not a static dataset)."

This is the single biggest disqualifying issue.

### Why it fails
GRPO training against a static dataset teaches the model to pattern-match pre-baked strings. It does not interact with the actual environment that judges will pull and evaluate. The trained model will behave completely differently in the live environment versus during training, because the two reward surfaces measure different things.

### Fix

**Step 1 — Write a rollout function that calls the live environment HTTP API**

In `train/train_minimal.py`, replace the static dataset approach with a rollout function:

```python
import requests

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

def run_episode_via_http(prompt_text: str, model, tok, task_id: str) -> float:
    """Run one full episode against the live environment. Returns final reward."""
    # 1. Reset
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": random.randint(0, 9999)}, timeout=15)
    r.raise_for_status()
    data = r.json()
    session_id = data["session_id"]

    # 2. Generate action from model
    completion = _generate_completion(model, tok, prompt_text)

    # 3. Parse decision + confidence from completion
    parsed = _extract_completion_fields(completion)
    if not parsed["decision"] or not parsed["confidence"]:
        return -0.2  # format penalty

    # 4. Submit terminal action to environment
    action = {
        "action_type": parsed["decision"],
        "confidence": parsed["confidence"],
        "parameters": {"reason": parsed["reason"]},
        "reasoning": parsed["reason"],
    }
    step_r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action, "session_id": session_id},
        timeout=15,
    )
    step_r.raise_for_status()
    step_data = step_r.json()
    return float(step_data["reward"])
```

**Step 2 — Rewrite reward_fn to use the live environment**

```python
def reward_fn(completions, prompts, task_ids, **kwargs):
    """
    GRPO reward function that calls the live environment for each completion.
    Each completion is one rollout; reward comes from /step response.
    """
    rewards = []
    for completion_list, prompt, task_id in zip(completions, prompts, task_ids):
        text = completion_list[0].get("content", "") if isinstance(completion_list, list) else completion_list
        reward = run_episode_via_http(prompt, model_ref, tok_ref, task_id)
        rewards.append(reward)
    return rewards
```

**Step 3 — Start the environment server before training**

Add a startup check to `main()`:

```python
def _wait_for_env(base_url: str, retries: int = 10) -> None:
    for i in range(retries):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"Environment ready at {base_url}")
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"Environment not reachable at {base_url} after {retries} retries")

# In main(), before training:
_wait_for_env(ENV_BASE_URL)
```

**Step 4 — Update the dataset to include task_id per row**

```python
rows = []
for ep in training_episodes:
    row = make_row(ep, tok)
    row["task_id"] = ep.task_id   # <- pass task_id so reward_fn can call the right endpoint
    rows.append(row)
dataset = Dataset.from_list(rows)
```

**Step 5 — In the Colab notebook, start the server in background before training cell**

Add a cell before training:
```python
import subprocess, time
server_proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"],
    cwd="/content/debateFloor"
)
time.sleep(8)  # wait for startup
import requests
assert requests.get("http://localhost:7860/health").json()["status"] == "healthy"
print("Environment server running.")
```

---

## FATAL-2 — Training evidence shows zero improvement

### What the judge sees
`reports/training_summary.json` contains:
```json
"component_shift": {
  "before": { "Decision accuracy": 0.0, "Fraud detection": 0.0, "Evidence quality": 0.5, "Calibration": -0.8 },
  "after":  { "Decision accuracy": 0.0, "Fraud detection": 0.0, "Evidence quality": 0.0, "Calibration": 0.0 }
}
```

Decision accuracy and Fraud detection are 0.0 both before and after. Evidence quality went down. Calibration went to 0.0 (not an improvement — it means the model stopped committing to any answer). The README claims "−0.34 → +0.83 mean reward" but this number does not appear in the JSON file judges will open.

### Why it fails
This is the primary evidence artifact judges look at to verify the "Showing Improvement in Rewards" criterion (20% of score). Showing 0.0 → 0.0 on every meaningful metric is worse than having no evidence at all.

### Fix

**Step 1 — Re-run training after FATAL-1 is fixed.** The component scores will be non-zero once the reward function actually interacts with the environment.

**Step 2 — Verify the eval harness measures what it claims**

In `evaluate_component_shift()`, the scoring function `_score_completion()` checks:
```python
fraud_hits = sum(1 for signal in expected if signal.replace("_", " ") in completion_lc)
```
This is keyword matching, not environment scoring. Fix it to call the environment and use `reward_breakdown` fields directly from the `/step` response instead of re-scoring locally. See FATAL-1 fix above.

**Step 3 — After re-running, ensure the mean reward in README matches the JSON**

In `save_training_artifacts()`, add the mean reward to the summary JSON:
```python
rewards = [r.get("reward") or r.get("rewards/mean") for r in log_history if r.get("reward") or r.get("rewards/mean")]
summary["mean_reward_before"] = float(before_components.get("Decision accuracy", 0.0))  # use a real metric
summary["mean_reward_after_training"] = float(sum(rewards) / len(rewards)) if rewards else 0.0
summary["mean_reward_final"] = summary["mean_reward_after_training"]
```

**Step 4 — Commit the updated reports/ artifacts to the repo.** Judges pull the repo, not just the Space. The JSON must reflect real training numbers.

---

## FATAL-3 — Evidence quality is 0.0 in all eval rows

### What the judge sees
Every row in `reports/eval_report.json`:
```json
{ "task_id": "clean_claim", "seed": 7, "reward": 0.825, "evidence_quality": 0.0 }
```

`evidence_quality` is 0.0 for every single task and seed. This is the column that measures whether the agent grounded fraud signals with actual evidence text. A weighted component at 14% that is always 0 means 14 points are left on the table every episode.

### Why it fails
The `evidence_quality_score` is only non-zero when `self._evidence_total > 0`, which only happens when `flag_fraud_signal` is called with evidence matching the keyword hints in `get_evidence_keyword_hints()`. The scripted eval agent in `inference_debatefloor.py` raises the wrong `flag_id` (`procedure_mismatch`) which is not in the task's `expected_signals`, so `self._evidence_hits` never increments.

### Fix

**Step 1 — Fix the scripted baseline in `inference_debatefloor.py`**

In `_strategy_contradictory_claim()`, the flag_id must match an expected_signal for the task. Check `app/tasks.py` — the `contradictory_claim` task expects `["date_mismatch", "cost_inflation", "signature_mismatch"]`. Fix:

```python
# OLD (wrong flag_id — not in expected_signals)
actions.append({
    "action_type": "flag_fraud_signal",
    "parameters": {
        "flag_id": "procedure_mismatch",
        "evidence": "...",
    },
})

# NEW (correct flag_id with evidence keyword that matches get_evidence_keyword_hints)
actions.append({
    "action_type": "flag_fraud_signal",
    "parameters": {
        "flag_id": "date_mismatch",
        "evidence": "Claim form records incident date 2026-02-20 but hospital admission on 2026-02-17 — date mismatch confirmed.",
    },
    "reasoning": "Document contradiction is a strong fraud indicator.",
})
```

**Step 2 — Validate get_evidence_keyword_hints covers the flags being raised**

In `app/tasks.py`, `get_evidence_keyword_hints()` must return keywords that appear in the evidence strings your agent writes. Audit the mapping and ensure each expected signal has at least 2–3 keywords a real agent would naturally write.

**Step 3 — Re-run `pre_validation_script.py` and `inference_debatefloor.py` after the fix**

After the fix, re-generate `reports/eval_report.json`. Evidence quality should be non-zero for `contradictory_claim` and `coordinated_fraud` tasks.

---

## FATAL-4 — variant_id is always 0

### What the judge sees
```json
{ "task_id": "clean_claim", "seed": 7, "variant_id": 0 }
{ "task_id": "clean_claim", "seed": 17, "variant_id": 0 }
```

Both seed=7 and seed=17 return variant_id=0. With `variant_id = abs(seed) % 5`, seed=7 gives 7%5=2 and seed=17 gives 17%5=2. Both should be variant 2, not 0. This means the eval script is not passing the seed through to `build_runtime_task`.

### Why it fails
The openenv.yaml claims "500+ unique episodes via seed variation." Judges will test this by calling `/reset` with different seeds and checking that episode content varies. If variant_id is always 0, the procedural generation claim is false.

### Fix

**Step 1 — Audit how seed flows through `/reset` to `build_runtime_task`**

In `app/main.py`, the `reset` endpoint receives `body.seed`. Trace it to `env.reset(task_id=body.task_id, seed=body.seed, ...)`. In `app/environment.py`, `reset()` passes `seed` to `build_runtime_task(selected_task, seed=seed)`. This looks correct in code.

**The likely bug:** the eval script generating `eval_report.json` calls `/reset` without passing `seed` in the body, or it passes seed as a query param instead of body. Check your eval script and ensure seed is in the POST body:

```python
# In your eval script — ensure seed is in the JSON body, not a query param
requests.post(f"{base_url}/reset", json={"task_id": task_id, "seed": seed}, timeout=15)
```

**Step 2 — Verify variant changes are observable in the response**

After reset with seed=7, the `observation.incident` or `observation.documents` content should differ from seed=42. Add a quick test:

```python
r1 = requests.post(f"{base}/reset", json={"task_id": "clean_claim", "seed": 7}).json()
r2 = requests.post(f"{base}/reset", json={"task_id": "clean_claim", "seed": 42}).json()
assert r1["observation"]["reward_breakdown"]["payout_accuracy"] != r2["observation"]["reward_breakdown"]["payout_accuracy"] \
    or r1["observation"]["documents"] != r2["observation"]["documents"], "Variants must differ"
```

**Step 3 — Update eval_report.json after fix**

Re-run the eval script and commit the updated report. variant_id must show different values across seeds.

---

## FATAL-5 — Rubric is decorative; it echoes the environment's own reward

### What the judge sees
In `tests/envs/test_debatefloor_rubric.py`:
```python
assert obs.rubric_reward == pytest.approx(obs.reward)
```

The test asserts they are equal. In `app/rubrics.py`, `DebateFloorRubric.forward()` reads fields from `observation.reward_breakdown` — the same fields the environment already computed. The rubric has no independent logic; it just re-weights what's already there.

### Why it fails
OpenEnv rubrics are supposed to provide an **independent evaluation signal** — a separate judgment layer that can disagree with or supplement the environment's reward. A rubric that always equals the environment reward provides no additional information to the training infrastructure and is architecturally wrong.

### Fix

**Step 1 — Give the rubric an independent evaluation role**

The rubric should evaluate the **reasoning quality** of the action independently of the environment's outcome. Implement a lightweight independent check — for example, the rubric checks whether the agent's `action.reasoning` field contains structured evidence:

```python
class _ReasoningQualityRubric(Rubric):
    """
    Independent of environment reward.
    Scores whether the agent's reasoning text references specific evidence.
    This fires on every step, giving a dense process signal the env reward doesn't have.
    """
    EVIDENCE_KEYWORDS = [
        "date", "mismatch", "document", "inconsistency", "signal", "evidence",
        "policy", "hospital", "bill", "procedure", "claim", "fraud", "verified",
    ]

    def forward(self, action: Any, observation: Any) -> float:
        reasoning = getattr(action, "reasoning", "") or ""
        if len(reasoning) < 20:
            return 0.0
        reasoning_lc = reasoning.lower()
        hits = sum(1 for kw in self.EVIDENCE_KEYWORDS if kw in reasoning_lc)
        return min(1.0, hits / 4.0)   # 4 keywords = full score; independent of env outcome
```

**Step 2 — Update DebateFloorRubric to compose env-based + independent signals**

```python
class DebateFloorRubric(Rubric):
    def __init__(self):
        super().__init__()
        # Env-derived components (as before)
        self.fraud_detection = _RewardFieldRubric("fraud_detection_score")
        self.decision_accuracy = _RewardFieldRubric("decision_accuracy")
        self.calibration_score = _RewardFieldRubric("calibration_score")
        # New: independent rubric — can disagree with env reward
        self.reasoning_quality = _ReasoningQualityRubric()

        self._weights = {
            "fraud_detection":   0.30,
            "decision_accuracy": 0.25,
            "calibration_score": 0.25,
            "reasoning_quality": 0.20,   # <- independent signal
        }
```

**Step 3 — Update the test to reflect the rubric is now independent**

```python
# OLD (asserts equality — wrong)
assert obs.rubric_reward == pytest.approx(obs.reward)

# NEW (asserts rubric ran and produced a value, not that it equals env reward)
assert 0.0 <= obs.rubric_reward <= 1.0
assert obs.rubric_components["reasoning_quality"] >= 0.0
# Rubric and env reward should NOT always be equal
```

---

## CRITICAL-1 — No Unsloth usage

### What the judge sees
`train/train_minimal.py` docstring says "no Unsloth" right in the title. The judging criteria says:

> "A working training script using **Unsloth** or HF TRL"

The hackathon stack explicitly features Unsloth for memory efficiency and it is named in the requirements document. `train/requirements.txt` does not list `unsloth`.

### Fix

**Step 1 — Add Unsloth to `train/requirements.txt`**

```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
```

Or for stable release:
```
unsloth>=2024.8
```

**Step 2 — Replace model loading in `train_minimal.py` with Unsloth's `FastLanguageModel`**

```python
# OLD
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map="auto")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# NEW
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=512,
    dtype=None,          # auto-detect
    load_in_4bit=True,   # QLoRA on T4
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

**Step 3 — Fix model save to use Unsloth's correct merged save path**

The hackathon guide explicitly warns: "Do not upcast a 4-bit model to 16-bit and then merge the LoRA weights naively."

```python
# OLD (potentially corrupts QLoRA weights)
model.save_pretrained("./debatefloor_checkpoint")

# NEW (Unsloth's safe merge path)
model.save_pretrained_merged(
    "./debatefloor_checkpoint",
    tok,
    save_method="merged_16bit",   # safe merge
)
```

**Step 4 — Update `GRPOConfig` for Unsloth compatibility**

Remove `fp16` / `bf16` flags — Unsloth handles this internally. Set `gradient_checkpointing=True`.

**Step 5 — Update notebook Cell 1 install to include Unsloth**

```python
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q trl>=0.12.0 transformers peft accelerate datasets wandb matplotlib requests
```

---

## CRITICAL-2 — Training reward and eval reward use completely different math

### What the judge sees
`openenv.yaml` states `never_mix: true` and calls this "CRITICAL — compound rewards break GRPO." Yet:

- Training uses `training_reward()` from `calibration_grader.py` — an unbounded scalar: `−0.05 step penalty + 1.0/−0.5 decision + 0.3*flags + 0.5*calibration`
- The environment observation returns `reward_breakdown.total` — a `[0, 1]` clamped weighted sum
- These two numbers are presented interchangeably in the README and WandB curves

Judges will ask: "Which reward is the curve showing?" and there is no clear answer.

### Fix

**Step 1 — Decide on one reward definition for training and be explicit**

The training scalar from `training_reward()` is correct for GRPO (unbounded is fine). The env reward (0–1) is correct for evaluation. Keep both. Make the distinction explicit everywhere.

**Step 2 — Label the WandB run clearly**

In `wandb.init()`:
```python
wandb.init(
    ...
    config={
        "reward_type": "training_scalar_unbounded",
        "reward_range": "[-1.5, 2.2] approximately — not clamped",
        "eval_reward_type": "six_component_clamped_0_1",
    }
)
```

**Step 3 — Fix README to clarify the distinction**

Update the results table caption:
```
Note: "Mean reward" in the training curve is the raw GRPO training scalar 
(unbounded, used for gradient stability). The evaluation reward in the 
before/after table is the six-component score clamped to [0.0, 1.0]. 
These are different numbers measuring different things — this is intentional 
per openenv.yaml:never_mix=true.
```

**Step 4 — Ensure `save_training_artifacts()` saves both reward types separately**

```python
summary["training_reward_curve"] = {
    "type": "unbounded_scalar",
    "values": [r["reward"] for r in log_history if "reward" in r],
}
summary["eval_reward_before"] = before_components
summary["eval_reward_after"] = after_components
```

---

## HIGH-1 — coordinated_fraud task missing from openenv.yaml

### What the judge sees
`openenv.yaml` defines exactly 3 tasks: `clean_claim`, `contradictory_claim`, `distribution_shift_claim`. But `app/tasks.py` has `coordinated_fraud` as a full `TaskDefinition` in the `TASKS` dict, and `reports/eval_report.json` tests it. Judges pull the YAML to understand what your environment does — it is the manifest.

### Fix

Add `coordinated_fraud` to the `tasks:` section in `openenv.yaml`:

```yaml
tasks:
  - id: clean_claim
    difficulty: easy
    max_steps: 10
    objective: >-
      Validate a legitimate insurance claim. All documents are in order.
      Correct decision: approve_claim with HIGH confidence.

  - id: contradictory_claim
    difficulty: medium
    max_steps: 18
    objective: >-
      Detect fraud signals in a claim with contradictory documents.
      Correct decision: deny_claim with MED confidence.

  - id: coordinated_fraud
    difficulty: hard
    max_steps: 22
    objective: >-
      Investigate a coordinated fraud ring. Multiple linked claims share
      emergency contact and broker. Agent must query_linked_claim to discover
      cross-claim signals. Correct decision: escalate_to_human or request_investigation.

  - id: distribution_shift_claim
    difficulty: hard
    max_steps: 28
    objective: >-
      Investigate a phantom provider or distribution-shifted claim.
      Agent must verify_provider_registration and query historical data.
      Correct decision: escalate_to_human with LOW confidence.

  - id: identity_fraud
    difficulty: medium
    max_steps: 20
    objective: >-
      Detect identity fraud. Agent must verify_identity to reveal mismatch.
      Correct decision: deny_claim with MED confidence.
```

Also verify `list_tasks_summary()` in `app/tasks.py` returns all tasks from the `TASKS` dict — currently it may only list the 3 in the YAML. Check the `/tasks` endpoint returns all 5 task IDs.

---

## HIGH-2 — Anti-gaming detector is effectively disabled during training

### What the judge sees
`self._episode_history` accumulates within one `InsuranceClaimEnvironment` instance. But `app/main.py` creates one environment per `session_id`. A GRPO run spawning 64 concurrent sessions means each session only ever sees 1–2 episodes — far below the `MIN_HISTORY_FOR_GAMING_DETECTION = 10` threshold. The anti-gaming system never fires during training.

### Fix

**Step 1 — Move episode history to a shared, process-level store**

In `app/main.py`, add a global confidence history store:

```python
from collections import deque
from threading import Lock

_global_confidence_history: deque = deque(maxlen=500)  # last 500 episodes, all sessions
_confidence_history_lock = Lock()

def record_episode_confidence(confidence: str) -> list[dict]:
    """Thread-safe append. Returns snapshot for gaming detection."""
    with _confidence_history_lock:
        _global_confidence_history.append({"confidence": confidence})
        return list(_global_confidence_history)
```

**Step 2 — Pass the global history to `calibration_reward()` instead of per-instance history**

In `app/environment.py`, after a terminal action:

```python
# OLD
self._calibration_score = compute_calibration_reward(
    effective_decision, conf_str, effective_ground_truth,
    self._episode_history,   # <- only this session's history
)

# NEW
from app.main import record_episode_confidence
global_history = record_episode_confidence(conf_str)
self._calibration_score = compute_calibration_reward(
    effective_decision, conf_str, effective_ground_truth,
    global_history,   # <- all sessions' history
)
```

**Note:** Avoid circular import — move `record_episode_confidence` to a separate `app/session_store.py` module imported by both `main.py` and `environment.py`.

**Step 3 — Add a `/stats` endpoint so judges can verify anti-gaming is active**

```python
@app.get("/stats")
def stats() -> dict:
    with _confidence_history_lock:
        history = list(_global_confidence_history)
    total = len(history)
    if total == 0:
        return {"episodes_recorded": 0, "confidence_distribution": {}}
    return {
        "episodes_recorded": total,
        "confidence_distribution": {
            "HIGH": sum(1 for e in history if e["confidence"] == "HIGH") / total,
            "MED":  sum(1 for e in history if e["confidence"] == "MED")  / total,
            "LOW":  sum(1 for e in history if e["confidence"] == "LOW")  / total,
        },
        "gaming_detected": total >= 10,
    }
```

---

## HIGH-3 — server/app.py violates client/server separation

### What the judge sees
`server/app.py` contains exactly one line:
```python
from app.main import app
```

The `server/` module is supposed to be the server boundary. Having it just re-export from `app/` means any code importing `server.app` is actually importing `app.main`. This violates the OpenEnv architecture principle:

> "Clients should never import server internals."

### Fix

**Option A (minimal) — Make server/app.py a proper entry point**

```python
# server/app.py
"""
Server entry point for DebateFloor environment.
All business logic lives in app/. This module is the deployment boundary.
"""
import uvicorn
from app.main import app  # noqa: F401 — re-exported for uvicorn discovery

__all__ = ["app"]


def serve(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run("server.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    serve()
```

**Option B (proper) — Move FastAPI app creation to server/app.py**

Move the `app = FastAPI(...)` instantiation and all route definitions from `app/main.py` into `server/app.py`. Keep `app/` as pure business logic (environment, models, tasks, rubrics). This is the architecturally correct separation.

**Update Dockerfile CMD** to reflect whichever option you choose:
```dockerfile
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

---

## HIGH-4 — Training loss of 0.005 indicates model collapse

### What the judge sees
`training_summary.json` shows `training_loss: 0.005647`. For a 0.5B model doing GRPO over 100 episodes / 2 epochs with batch size 2, normal loss is in the 0.5–2.0 range. A loss of 0.005 means either the model memorised the tiny dataset in epoch 1 and the gradient went to zero, or the reward function produced near-constant rewards (zero gradient signal).

### Fix

**Step 1 — Increase dataset size before re-running**

Change `EPISODES = 100` to at least `EPISODES = 300`. 100 episodes with batch_size=2 and 2 epochs is only 100 gradient steps — far too few for meaningful GRPO learning.

**Step 2 — Ensure reward variance is non-zero**

GRPO learns from reward differences within a group. If all 4 completions for a prompt get the same reward (e.g., all −0.05 because the model always outputs garbage), the advantage is zero and no learning happens. Add reward variance logging:

```python
# In reward_fn, before returning rewards:
import statistics
if len(rewards) > 1:
    variance = statistics.variance(rewards)
    if variance < 0.01:
        print(f"WARNING: Low reward variance ({variance:.4f}) — GRPO gradient may be near zero")
wandb.log({"reward_variance": variance}) if USE_WANDB else None
```

**Step 3 — Increase num_generations from 4 to 8**

More generations per prompt = more reward variance = stronger GRPO gradient:
```python
args = GRPOConfig(
    ...
    num_generations=8,   # was 4
    ...
)
```

**Step 4 — After re-running with the env-connected reward (FATAL-1 fix), loss should normalise.** The root cause is that `training_reward()` was returning −0.05 (step penalty with `done=False`) for every step because the static dataset approach never set `done=True` for most rows.

---

## MEDIUM-1 — reward_fn uses keyword matching instead of environment signals

### What the judge sees
In `train_minimal.py`:
```python
legit = sum(1 for s in sigs if s.replace("_", " ") in text.lower())
r = training_reward(decision, confidence, gt, legit, step_num=1, done=True)
```

`legit` counts how many fraud signal names appear anywhere in the completion text. The actual environment only awards fraud detection credit when `validate_document` or `query_linked_claim` is called first, then `flag_fraud_signal` with grounded evidence. These are completely different checks.

### Fix

This is fully resolved by FATAL-1 (connecting to the live environment). Once reward comes from `/step`, `legit` is no longer computed from text matching — it comes from the environment's `reward_breakdown.fraud_detection_score` directly.

If you still want a simple scalar reward for GRPO training (recommended for stability), extract it from the environment response:

```python
def reward_fn(completions, prompts, task_ids, **kwargs):
    rewards = []
    for completion_list, prompt, task_id in zip(completions, prompts, task_ids):
        text = ...  # get completion text
        env_reward = run_episode_via_http(prompt, model_ref, tok_ref, task_id)
        rewards.append(env_reward)
    return rewards
```

No more keyword matching needed.

---

## MEDIUM-2 — WandB curve caption is ambiguous

### What the judge sees
README says:
> "Note on reward scale: Training reward is an unbounded shaped scalar for gradient stability. Evaluation reward is clamped to [0.0, 1.0]. The curve shows the training signal, not the evaluation score."

This note is buried after the plots. Judges looking at the WandB run first will see values outside [0, 1] and be confused.

### Fix

**Step 1 — In the WandB run, log both signals separately with clear names**

```python
wandb.log({
    "train/reward_scalar": training_reward_value,       # unbounded
    "eval/reward_clamped": eval_reward_clamped,         # [0, 1]
    "eval/decision_accuracy": decision_accuracy,
    "eval/calibration_score": calibration_score,
})
```

**Step 2 — Add axis labels to the reward curve SVG**

In `save_training_artifacts()`, the matplotlib plot already labels axes. Verify the saved `docs/reward_curve.svg` has:
- x-axis: "Training step"
- y-axis: "Training reward (unbounded scalar)"
- Title: "DebateFloor GRPO training — training reward per step"

Add a text annotation:
```python
ax1.annotate(
    "Note: training scalar is unbounded.\nSee eval table for [0,1] clamped scores.",
    xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9, color='gray'
)
```

---

## Quick Wins — Do these last, they take < 30 minutes total

### QW-1 — Run pre_validation_script.py against the live Space before eval

```bash
python pre_validation_script.py --base-url https://huggingface.co/spaces/AniketAsla/debatefloor
```

All checks must be green. Fix any failures before the evaluation day.

### QW-2 — Ensure `/tasks` returns all 5 task IDs

```python
r = requests.get("https://your-space.hf.space/tasks").json()
task_ids = {t["task_id"] for t in r["tasks"]}
assert task_ids == {"clean_claim", "contradictory_claim", "coordinated_fraud", "identity_fraud", "distribution_shift_claim"}
```

### QW-3 — Add Colab badge to README pointing to the notebook

Judges need a one-click path to re-run training. Add:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AniketAslaliya/debateFloor/blob/main/train/train_debatefloor.ipynb)
```

### QW-4 — Commit the updated reports/ to the repo

After re-running all evals and training:
```bash
git add reports/ docs/reward_curve.svg docs/component_shift.svg
git commit -m "fix: update training artifacts with env-connected GRPO results"
git push
```

### QW-5 — Verify the HF Space link in README resolves to a running environment

The README links to `https://huggingface.co/spaces/AniketAsla/debatefloor`. Hit `/health` from a fresh browser. If the Space is sleeping, judges will see a cold-start delay. Pin the Space (Settings → Pin Space) to keep it warm.

### QW-6 — Add a `/rollout` endpoint for judges to test the full agent loop

This is a bonus that impresses. One endpoint that runs a full scripted episode and returns the step-by-step trace:
```python
@app.post("/rollout")
def rollout(task_id: str = "contradictory_claim", seed: int = 42) -> dict:
    """Runs a scripted demo episode. Returns full trace for judges."""
    ...
```

---

## Fix Priority Order (Day-of-Evaluation)

| Order | Issue | Time estimate | Blocking? |
|-------|-------|---------------|-----------|
| 1 | FATAL-1: Connect training to env | 2–3 hours | Yes — everything else depends on real training |
| 2 | CRITICAL-1: Add Unsloth | 30 min | Yes — minimum requirement |
| 3 | FATAL-3: Fix evidence_quality (wrong flag_id) | 20 min | Yes — eval report |
| 4 | FATAL-4: Fix variant_id in eval script | 20 min | Yes — procedural generation claim |
| 5 | HIGH-1: Add coordinated_fraud to openenv.yaml | 10 min | Yes — YAML is what judges read |
| 6 | Re-run training with FATAL-1+CRITICAL-1 fixes | 15–30 min (T4) | Yes — need real artifacts |
| 7 | FATAL-2: Commit updated reports/ JSON | 5 min | Yes |
| 8 | FATAL-5: Make rubric independent | 45 min | No — but visible in code review |
| 9 | HIGH-2: Global anti-gaming store | 30 min | No — but listed as innovation |
| 10 | HIGH-3: Fix server/app.py | 15 min | No — architectural |
| 11 | CRITICAL-2: Label reward types clearly | 20 min | No — but affects WandB clarity |
| 12 | MEDIUM-2: Fix WandB axis labels | 10 min | No |
| 13 | Quick wins QW-1 through QW-6 | 30 min total | No |

**Total estimated time: 6–8 hours of focused work.**

---

## Verification Checklist

Before submitting, every item below must be true:

- [ ] `/health` returns `{ "status": "healthy" }` on the live Space
- [ ] `/reset` with seed=7 and seed=42 return different `documents[0].content` values
- [ ] `/step` with a valid action returns `reward` that changes based on action quality
- [ ] `reports/eval_report.json` has `evidence_quality > 0.0` for at least one task row
- [ ] `reports/eval_report.json` has `variant_id` values other than 0
- [ ] `reports/training_summary.json` has `decision_accuracy > 0.0` in the `after` block
- [ ] `train/train_minimal.py` imports and uses `FastLanguageModel` from Unsloth
- [ ] `train/train_minimal.py` calls `/reset` and `/step` during training (not static dataset only)
- [ ] `openenv.yaml` lists all 5 task IDs
- [ ] `server/app.py` is more than a one-line re-export
- [ ] The Colab notebook runs end-to-end without errors on a T4 runtime
- [ ] WandB run URL in README resolves to a real run with a reward curve that goes up
- [ ] README Colab badge opens the correct notebook
- [ ] `pre_validation_script.py` exits with code 0 against the live Space
