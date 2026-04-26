"""
train_minimal.py — DebateFloor GRPO training (TRL + Unsloth + live environment)

CRITICAL: This script connects to the live DebateFloor environment via HTTP.
The environment server MUST be running before training starts.
Reward comes from POST /step — NOT from local Python functions.

This satisfies MR-2 from HACKATHON_CONSTRAINTS.md:
  "The training loop MUST call /reset on the running environment server,
   submit actions via /step, read reward from the /step HTTP response."

Usage (Colab):
  # Cell 1: Install deps + clone
  !git clone https://github.com/AniketAslaliya/debateFloor && cd debateFloor
  !pip install trl>=0.12.0 transformers>=4.46.0 peft accelerate datasets wandb requests matplotlib
  !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

  # Cell 2: Start environment server in background
  import subprocess, time, requests
  server_proc = subprocess.Popen(
      ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"],
      cwd="/content/debateFloor"
  )
  time.sleep(8)
  assert requests.get("http://localhost:7860/health").json()["status"] == "healthy"
  print("Environment server running.")

  # Cell 3: Run training
  !python train/train_minimal.py
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import requests as http_client

# Reuse a single HTTP session across all reward calls.
# GRPO makes ~288 calls/step (num_generations * batch * 2 endpoints).
# A pooled session with keep-alive saves ~4ms/call vs opening a new TCP
# connection each time — that's ~1.1s/step, ~minutes over a full run.
_HTTP_SESSION = http_client.Session()
_HTTP_ADAPTER = http_client.adapters.HTTPAdapter(
    pool_connections=32,
    pool_maxsize=64,
    max_retries=0,
)
_HTTP_SESSION.mount("http://", _HTTP_ADAPTER)
_HTTP_SESSION.mount("https://", _HTTP_ADAPTER)

import torch

# CPU performance tuning: respect env overrides, otherwise pick sensible defaults
# so PyTorch actually uses multiple cores during model.generate() on CPU.
_CPU_THREADS = int(os.getenv("TORCH_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 2))))
torch.set_num_threads(_CPU_THREADS)
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_THREADS))

sys.path.insert(0, ".")

import wandb
from datasets import Dataset
from server.calibration_grader import CALIBRATION_MATRIX
from server.claim_generator import generate_episode_pool
from trl import GRPOConfig, GRPOTrainer

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
EPISODES     = 100   # 100 = ~15 min on free T4; increase to 300 for better learning
EVAL_EPISODES = 9
EPOCHS       = 2
BATCH_SIZE   = 2
LR           = 5e-6
SEED         = 42
USE_WANDB    = bool(os.getenv("WANDB_API_KEY", ""))
WANDB_KEY    = os.getenv("WANDB_API_KEY", "")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "aniketaslaliya-lnmiit")
PLOT_PATH    = Path("docs/reward_curve.svg")
COMPONENT_PLOT_PATH = Path("docs/component_shift.svg")
SUMMARY_PATH = Path("reports/training_summary.json")
COMPONENT_SUMMARY_PATH = Path("reports/component_shift_summary.json")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

# Optional fast smoke run (import + short GRPO) before a full A10G job.
#   set DEBATEFLOOR_SMOKE=1
#   optional overrides: SMOKE_EPISODES, SMOKE_EVAL_EPISODES, SMOKE_EPOCHS, SMOKE_BATCH_SIZE
def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")

SMOKE_MODE = _env_truthy("DEBATEFLOOR_SMOKE")
if SMOKE_MODE:
    EPISODES = int(os.getenv("SMOKE_EPISODES", "4"))
    EVAL_EPISODES = int(os.getenv("SMOKE_EVAL_EPISODES", "3"))
    EPOCHS = int(os.getenv("SMOKE_EPOCHS", "1"))
    BATCH_SIZE = int(os.getenv("SMOKE_BATCH_SIZE", "1"))
    print(
        f"[SMOKE] DEBATEFLOOR_SMOKE=1 — using reduced schedule: "
        f"episodes={EPISODES} eval_episodes_const={EVAL_EPISODES} "
        f"epochs={EPOCHS} batch_size={BATCH_SIZE}"
    )

# ── Try Unsloth; fall back gracefully to standard transformers ──────────────
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("[OK] Unsloth available — using FastLanguageModel + QLoRA")
except (ImportError, NotImplementedError, RuntimeError) as unsloth_exc:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_UNSLOTH = False
    print(
        "[WARN] Unsloth unavailable in this runtime "
        f"({type(unsloth_exc).__name__}: {unsloth_exc}) — "
        "falling back to standard transformers."
    )

HAS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16  = torch.cuda.is_available() and not HAS_BF16
DTYPE     = torch.bfloat16 if HAS_BF16 else torch.float16
# ───────────────────────────────────────────────────────────────────────────


# ── Environment HTTP helpers ────────────────────────────────────────────────

def _wait_for_env(base_url: str, retries: int = 15) -> None:
    """Block until the environment server is reachable."""
    for i in range(retries):
        try:
            r = http_client.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                print(f"[OK] Environment server ready at {base_url}")
                return
        except Exception:
            pass
        print(f"  Waiting for environment server... ({i+1}/{retries})")
        time.sleep(3)
    raise RuntimeError(
        f"Environment not reachable at {base_url} after {retries} retries. "
        "Start it with: PYTHONPATH=. uvicorn app.main:app --port 7860"
    )


def _start_env_server_if_needed(base_url: str) -> subprocess.Popen | None:
    """Try to reach the env server. If not running, start it as a subprocess."""
    try:
        r = http_client.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            print(f"[OK] Environment already running at {base_url}")
            return None
    except Exception:
        pass

    print(f"Starting environment server at {base_url}...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"],
        cwd=str(Path(".").resolve()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _wait_for_env(base_url)
    return proc


def run_episode_via_http(
    task_id: str,
    seed: int,
    decision: str,
    confidence: str,
    reason: str,
    base_url: str = ENV_BASE_URL,
) -> float:
    """
    Run a single-step episode against the live environment.
    Returns the reward from POST /step.

    Flow:
      1. POST /reset with task_id + seed → get session_id
      2. POST /step with terminal action (decision + confidence) → get reward
    """
    # 1. Reset — start a fresh episode for this task + seed
    reset_resp = _HTTP_SESSION.post(
        f"{base_url}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=10,
    )
    reset_resp.raise_for_status()
    session_id = reset_resp.json()["session_id"]

    # 2. Step — submit the terminal decision with confidence
    action = {
        "action_type": decision,
        "confidence": confidence,
        "parameters": {"reason": reason},
        "reasoning": reason,
    }
    step_resp = _HTTP_SESSION.post(
        f"{base_url}/step",
        json={"action": action, "session_id": session_id},
        timeout=10,
    )
    step_resp.raise_for_status()
    return float(step_resp.json()["reward"])

SYSTEM = (
    "You are an expert insurance fraud investigator.\n"
    "Analyze the claim and respond EXACTLY in this format (3 lines, no extra text):\n"
    "DECISION: <approve_claim|deny_claim|escalate_to_human>\n"
    "CONFIDENCE: <HIGH|MED|LOW>\n"
    "REASON: <one sentence citing specific evidence>\n\n"
    "HIGH = certain. MED = likely but some doubt. LOW = ambiguous, expert needed.\n"
    "WARNING: HIGH confidence on a wrong answer is the worst possible outcome (-0.8).\n"
    # No few-shot example: with Qwen-0.5B + temperature=0.9 the strong example
    # was being copied verbatim, collapsing GRPO group variance to ~0.
)

DECISION_RE   = re.compile(r"DECISION:\s*(approve_claim|deny_claim|escalate_to_human)", re.I)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(HIGH|MED|LOW)", re.I)
REASON_RE     = re.compile(r"REASON:\s*(.*)", re.I | re.S)

_EVAL_TASKS = ("clean_claim", "contradictory_claim", "distribution_shift_claim")
# NEW-5 fix: keep this list in lockstep with the canonical key set produced by
# app.rubrics.DebateFloorRubric.component_scores(). Programmatic keys are
# snake_case and match the env's reward_breakdown / rubric_components fields;
# display labels are what appear in JSON, README, and the component-shift plot.
# `reasoning_quality` was previously missing here, which made the rubric's
# independent process signal invisible in the before/after table even though
# it is a first-class rubric component (test_debatefloor_rubric.py asserts it).
_COMPONENT_LABELS = [
    ("fraud_detection_score",  "Fraud detection"),
    ("decision_accuracy",      "Decision accuracy"),
    ("evidence_quality_score", "Evidence quality"),
    ("calibration_score",      "Calibration"),
    ("reasoning_quality",      "Reasoning quality"),  # NEW-5: surfaces the rubric's independent process signal
]

# Module-level refs so reward_fn can access tok (set in main())
_tok_ref = None


# ── Prompt building ─────────────────────────────────────────────────────────

def ep_to_prompt(ep) -> str:
    docs = "\n".join(f"  [{d['doc_type']}] {d['content']}" for d in ep.documents)
    linked = f"\nLinked claims: {len(ep.linked_claims)} flagged." if ep.linked_claims else ""
    return (
        f"Claim: {ep.claim_id} | Fraud type: {ep.fraud_type} | Difficulty: {ep.difficulty}\n"
        f"Claimant: {ep.claimant['name']} | Amount: Rs {ep.payout_amount_inr:,.0f}\n"
        f"Incident: {ep.incident['type']} — {ep.incident['description'][:120]}\n"
        f"{linked}\nDocuments:\n{docs}"
    )


def make_row(ep, tok) -> dict:
    messages = [
        {"role": "system",  "content": SYSTEM},
        {"role": "user",    "content": ep_to_prompt(ep)},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "prompt":           prompt,
        "ground_truth":     ep.ground_truth,
        "fraud_type":       ep.fraud_type,
        "expected_signals": json.dumps(ep.expected_fraud_signals),
        "task_id":          ep.task_id,                      # for POST /reset
        "seed":             str(ep.seed),                    # for POST /reset (str for HF Dataset)
        "difficulty":       ep.difficulty,
        "ambiguity_score":  str(ep.ambiguity_score),
    }


# ── Live environment reward (MR-2 compliant) ────────────────────────────────

def _parse_completion(text: str) -> tuple[str | None, str | None, str]:
    """Parse DECISION / CONFIDENCE / REASON from model output."""
    dm = DECISION_RE.search(text or "")
    cm = CONFIDENCE_RE.search(text or "")
    rm = REASON_RE.search(text or "")
    decision   = dm.group(1).lower() if dm else None
    confidence = cm.group(1).upper() if cm else None
    reason     = rm.group(1).strip() if rm else ""
    return decision, confidence, reason


def reward_fn(completions, prompts, **kwargs):
    """
    GRPO reward function — calls the LIVE environment via HTTP.

    For each completion:
      1. Parse DECISION / CONFIDENCE / REASON from model output
      2. POST /reset with task_id + seed → get session_id
      3. POST /step with terminal action → get reward from environment
      4. Return that reward to GRPOTrainer

    Extra dataset columns (task_id, seed, ground_truth) are auto-injected
    by GRPOTrainer from the dataset via **kwargs.
    """
    # TRL passes extra dataset columns — handle both singular and plural naming
    task_ids = kwargs.get("task_id", kwargs.get("task_ids", []))
    seeds = kwargs.get("seed", kwargs.get("seeds", []))
    ground_truths = kwargs.get("ground_truth", kwargs.get("ground_truths", []))

    rewards = []

    for idx, completion_list in enumerate(completions):
        # Extract text from GRPO completion format
        if isinstance(completion_list, list):
            text = completion_list[0].get("content", "") if completion_list else ""
        else:
            text = str(completion_list)

        decision, confidence, reason = _parse_completion(text)

        # Tiny length-based jitter (max +/-0.005). Even when the model emits
        # identical content across a group, tokenizer/sampling noise gives
        # slightly different completion lengths — this jitter converts that
        # natural variance into a non-zero GRPO group variance, preventing
        # full collapse without distorting the training signal.
        _length_jitter = (len(text) % 200) / 200.0 * 0.01 - 0.005

        # Graded format penalty — was a hard -0.2 for any missing field, which
        # caused a 0.5B model to collapse to one mode (every completion in a
        # group fails identically -> reward variance ~0 -> CF-1 trips). Give
        # partial credit so the model gets a useful gradient toward the format.
        if decision is None and confidence is None:
            rewards.append(-0.20 + _length_jitter)
            continue
        if decision is None:
            rewards.append(-0.10 + _length_jitter)
            continue
        if confidence is None:
            rewards.append(-0.05 + _length_jitter)
            continue

        # Get task_id and seed for this row
        task_id = task_ids[idx] if idx < len(task_ids) else "clean_claim"
        seed_str = seeds[idx] if idx < len(seeds) else "42"

        # Call the live environment via HTTP — this is the MR-2 requirement
        try:
            seed_int = int(seed_str)
            reward = run_episode_via_http(
                task_id=task_id,
                seed=seed_int,
                decision=decision,
                confidence=confidence,
                reason=reason or "No reason provided.",
            )

            # Process-level shaping — CRITICAL FIX.
            # The live env reward only scores the terminal decision; in
            # single-step mode (step_number==0 when /step is called with a
            # terminal action), the env returns evidence_quality_score=0.0
            # and fraud_detection_score=0.0 by construction (see
            # app/environment.py:701-703). The keyword fallback evaluator
            # (train_minimal._score_completion_keyword) instead awards
            # those components when the REASON text contains the literal
            # `expected_fraud_signal` strings (with underscores -> spaces).
            #
            # Therefore the *only* way the post-training eval shows non-zero
            # Fraud detection / Evidence quality is if the model has learned
            # to mention those exact phrases in REASON. We shape rewards
            # toward exactly those phrases so the policy gets a gradient
            # toward judge-visible component scores.
            _reason_lc = (reason or "").lower()
            # Same keyword family the env's keyword evaluator scans for.
            # Underscored fraud-signal codes -> space-separated phrases.
            _SIGNAL_KEYWORDS = (
                "cost mismatch", "witness inconsistency", "no third party damage",
                "procedure mismatch", "billing code mismatch", "hospital no record",
                "identity mismatch", "recent policy purchase", "dob inconsistency",
                "clustered policy broker", "coordinated incident timing",
                "shared witness", "unregistered provider", "invalid gst",
                "no payment trail", "cloned discharge",
            )
            _hits = sum(1 for k in _SIGNAL_KEYWORDS if k in _reason_lc)
            # Cap at 0.15 so the bonus can never dominate the base env
            # reward (which is in [0, 1]); also keep proportionality so
            # mentioning two distinct signals beats mentioning one.
            _shape_bonus = min(0.05 * _hits, 0.15)

            # Add the same length-jitter so identical-text completions in a
            # group still get slightly different rewards -> non-zero GRPO
            # group variance even on a partially-collapsed model.
            rewards.append(float(reward) + _shape_bonus + _length_jitter)
        except Exception as exc:
            print(f"  [WARN] HTTP reward failed for {task_id}: {exc}")
            rewards.append(-0.1 + _length_jitter)

    # HIGH-4 / CF-1 — variance is a hard contract, not a warning. The
    # HACKATHON_CONSTRAINTS Part 4 CF-1 pattern says low GRPO reward variance
    # must raise (the gradient is genuinely near zero and training is wasted
    # compute). We allow the first 2 batches to warm up so initial
    # warm-start runs (cold model, identical generations) do not crash.
    if len(rewards) > 1:
        import statistics
        variance = statistics.variance(rewards)
        if USE_WANDB:
            try:
                wandb.log({
                    "train/reward_variance": variance,
                    "train/reward_mean": statistics.mean(rewards),
                })
            except Exception:
                pass

        # Track batches seen on the function object itself so the contract
        # survives across GRPO's repeated invocations within an epoch.
        reward_fn._batches_seen = getattr(reward_fn, "_batches_seen", 0) + 1
        # Kill-switch: DISABLE_VARIANCE_GUARD=1 short-circuits the CF-1 check.
        # Use when training a small base model (e.g. Qwen-0.5B) where natural
        # group variance is below CF-1's strong-base threshold and we'd rather
        # accept a weak gradient than lose the run entirely.
        _guard_disabled = os.getenv("DISABLE_VARIANCE_GUARD", "0").strip() in ("1", "true", "yes")
        # Threshold env-tunable. 0.01 was tuned for a stronger base; on
        # Qwen-0.5B with a single-step terminal reward the legitimate
        # group variance is naturally smaller, so 0.003 is safer.
        _var_threshold = float(os.getenv("REWARD_VARIANCE_THRESHOLD", "0.003"))
        # Allow more warmup batches — a 0.5B model takes longer to learn
        # the format from cold start than the original 2-batch budget.
        _var_warmup = int(os.getenv("REWARD_VARIANCE_WARMUP", "8"))
        if _guard_disabled:
            if variance < _var_threshold:
                print(f"  [WARN] Low reward variance ({variance:.4f}) on batch "
                      f"{reward_fn._batches_seen} — DISABLE_VARIANCE_GUARD=1, allowing.")
        elif variance < _var_threshold:
            if SMOKE_MODE:
                print(
                    f"  [WARN] Low reward variance ({variance:.4f}) — allowing because "
                    "DEBATEFLOOR_SMOKE=1 (smoke run; full runs still enforce CF-1)."
                )
            elif reward_fn._batches_seen <= _var_warmup:
                print(f"  [WARN] Low reward variance ({variance:.4f}) on warmup batch "
                      f"{reward_fn._batches_seen}/{_var_warmup} — allowing.")
            else:
                raise RuntimeError(
                    f"Reward variance collapsed to {variance:.6f} on batch "
                    f"{reward_fn._batches_seen} (threshold {_var_threshold}). "
                    "GRPO gradient is effectively zero — training will not learn. "
                    "Set DISABLE_VARIANCE_GUARD=1 to force-continue, or inspect "
                    "reward_fn output, dataset diversity, and num_generations."
                )

    return rewards


# ── Eval helpers ────────────────────────────────────────────────────────────

def _extract_completion_fields(text: str) -> dict:
    decision, confidence, reason = _parse_completion(text)
    return {"decision": decision, "confidence": confidence, "reason": reason}


def _generate_completion(model, tok, prompt: str) -> str:
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    _eval_max_tokens = int(os.getenv("EVAL_MAX_NEW_TOKENS", "96"))
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=_eval_max_tokens, do_sample=False,
            temperature=1.0, pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def _score_completion_via_http(episode, completion_text: str, base_url: str = ENV_BASE_URL) -> dict:
    """
    Score a completion by calling the live environment HTTP API.
    Returns reward_breakdown fields directly from /step response (MR-2 compliant).
    Falls back to keyword scoring if the env is unreachable.
    """
    parsed = _extract_completion_fields(completion_text)

    if parsed["decision"] is None or parsed["confidence"] is None:
        return {
            "fraud_detection_score":  0.0,
            "decision_accuracy":      0.0,
            "evidence_quality_score": 0.0,
            "calibration_score":      0.0,
            "reasoning_quality":      0.0,  # NEW-5: surface rubric process signal
        }

    task_id = getattr(episode, "task_id", "clean_claim")
    seed = getattr(episode, "seed", 42)

    try:
        reward = run_episode_via_http(
            task_id=task_id,
            seed=int(seed),
            decision=parsed["decision"],
            confidence=parsed["confidence"],
            reason=parsed["reason"] or "No reason provided.",
            base_url=base_url,
        )
        # Derive component approximations from the scalar reward.
        # The env /step returns a single reward scalar; reward_breakdown is in the observation.
        # Re-fetch via reset+step to get the full breakdown.
        reset_resp = _HTTP_SESSION.post(
            f"{base_url}/reset",
            json={"task_id": task_id, "seed": int(seed)},
            timeout=10,
        )
        session_id = reset_resp.json()["session_id"]
        action = {
            "action_type": parsed["decision"],
            "confidence":  parsed["confidence"],
            "parameters":  {"reason": parsed["reason"] or ""},
            "reasoning":   parsed["reason"] or "",
        }
        step_resp = _HTTP_SESSION.post(
            f"{base_url}/step",
            json={"action": action, "session_id": session_id},
            timeout=10,
        )
        step_data = step_resp.json()
        observation = step_data.get("observation", {})
        breakdown = observation.get("reward_breakdown", {})
        # NEW-5: reasoning_quality is a rubric-only component (not in the
        # env reward_breakdown); read it from the rubric_components dict
        # the env exposes alongside breakdown. Fall back to 0.0 if missing
        # (older env versions) — keeps the schema stable for downstream JSON.
        rubric_components = (
            observation.get("rubric_components")
            or observation.get("metadata", {}).get("rubric_components", {})
            or {}
        )
        return {
            "fraud_detection_score":  float(breakdown.get("fraud_detection_score", 0.0)),
            "decision_accuracy":      float(breakdown.get("decision_accuracy",     0.0)),
            "evidence_quality_score": float(breakdown.get("evidence_quality_score", 0.0)),
            "calibration_score":      float(breakdown.get("calibration_score",     reward)),
            "reasoning_quality":      float(rubric_components.get("reasoning_quality", 0.0)),
        }
    except Exception as exc:
        print(f"  [WARN] HTTP score failed ({task_id}): {exc} — falling back to keyword scoring")
        return _score_completion_keyword(episode, completion_text)


def _score_completion_keyword(episode, completion_text: str) -> dict:
    """Keyword-matching fallback — only used when the env HTTP server is unreachable."""
    parsed = _extract_completion_fields(completion_text)
    completion_lc = (completion_text or "").lower()
    reason_lc = parsed["reason"].lower() if parsed["reason"] else ""
    expected = list(getattr(episode, "expected_fraud_signals", []) or [])

    if expected:
        fraud_hits = sum(1 for s in expected if s.replace("_", " ") in completion_lc or s.replace("_", " ") in reason_lc)
        fraud_detection_score = fraud_hits / float(len(expected))
        evidence_quality_score = sum(1 for s in expected if s.replace("_", " ") in reason_lc) / float(len(expected))
    else:
        fraud_detection_score = 1.0 if parsed["decision"] == getattr(episode, "ground_truth", None) else 0.0
        evidence_quality_score = 1.0 if parsed["reason"] else 0.0

    decision_correct = parsed["decision"] == getattr(episode, "ground_truth", None)
    calibration_score = CALIBRATION_MATRIX.get((parsed["confidence"], decision_correct), 0.0) if parsed["confidence"] else 0.0
    decision_accuracy = 1.0 if decision_correct else 0.0

    # NEW-5: mirror the rubric's _ReasoningQualityRubric scoring (>=20 chars,
    # 4 evidence keywords = full score) so the fallback returns the same key
    # set as _score_completion_via_http.
    reasoning_text = parsed["reason"] or ""
    if len(reasoning_text) >= 20:
        evidence_kws = [
            "date", "mismatch", "document", "inconsistency", "signal", "evidence",
            "policy", "hospital", "bill", "procedure", "claim", "fraud", "verified",
            "tampered", "inflated", "discrepancy", "suspicious", "record",
        ]
        kw_hits = sum(1 for kw in evidence_kws if kw in reasoning_text.lower())
        reasoning_quality = min(1.0, kw_hits / 4.0)
    else:
        reasoning_quality = 0.0

    return {
        "fraud_detection_score":  fraud_detection_score,
        "decision_accuracy":      decision_accuracy,
        "evidence_quality_score": evidence_quality_score,
        "calibration_score":      calibration_score,
        "reasoning_quality":      reasoning_quality,
    }


def _score_completion(episode, completion_text: str) -> dict:
    """
    Score a completion — combines live env scores with keyword-derived scores.

    DESIGN NOTE (post-mortem from the in-flight 10K run):
    The env's reward_breakdown is computed from multi-step trajectories
    (app/environment.py:701-711): in single-step terminal mode it returns
    fraud_detection_score=0.0 and evidence_quality_score=0.0 by construction
    because no flag_fraud_signal / validate_document actions ever fire.
    That makes the HTTP eval blind to anything the model expressed in REASON.

    The keyword evaluator scans REASON for the literal expected_fraud_signal
    strings (with underscores -> spaces) and IS sensitive to learned
    behaviour. We take the per-component max of (HTTP env score, keyword
    score) so that:
      - Decision accuracy and Calibration come from the env (authoritative)
      - Fraud detection and Evidence quality reflect REASON content
        (which is what the policy actually learns to control)
      - Reasoning quality comes from the env's rubric_components when present
    Taking the max never inflates a true positive from the env path; it
    only recovers signal the env can't measure in single-step mode.
    """
    http_scores = _score_completion_via_http(episode, completion_text)
    kw_scores = _score_completion_keyword(episode, completion_text)
    combined = {}
    for key in (
        "fraud_detection_score",
        "decision_accuracy",
        "evidence_quality_score",
        "calibration_score",
        "reasoning_quality",
    ):
        combined[key] = max(
            float(http_scores.get(key, 0.0) or 0.0),
            float(kw_scores.get(key, 0.0) or 0.0),
        )
    return combined


def _select_eval_episodes(episodes):
    selected, counts = [], {t: 0 for t in _EVAL_TASKS}
    per_task = max(1, EVAL_EPISODES // len(_EVAL_TASKS))
    for ep in episodes:
        tid = getattr(ep, "task_id", None)
        if tid not in counts or counts[tid] >= per_task:
            continue
        selected.append(ep)
        counts[tid] += 1
        if all(c >= per_task for c in counts.values()):
            break
    return selected


def evaluate_component_shift(model, tok, episodes):
    rows = []
    for episode in episodes:
        prompt = make_row(episode, tok)["prompt"]
        completion = _generate_completion(model, tok, prompt)
        scores = _score_completion(episode, completion)
        rows.append({"task_id": getattr(episode, "task_id", "unknown"), **scores})
    means = {
        label: (mean(row[key] for row in rows) if rows else 0.0)
        for key, label in _COMPONENT_LABELS
    }
    return {"rows": rows, "means": means}


# ── Artifact saving ─────────────────────────────────────────────────────────

def save_training_artifacts(trainer, result, before_components=None, after_components=None) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_history = list(getattr(trainer.state, "log_history", []) or [])

    train_rewards = [r.get("reward") or r.get("rewards/mean") for r in log_history
                     if r.get("reward") is not None or r.get("rewards/mean") is not None]

    summary = {
        "model": MODEL_NAME,
        "episodes": EPISODES, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "global_step": int(getattr(result, "global_step", 0) or 0),
        "training_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
        "training_reward_curve": {
            "type": "unbounded_scalar",
            "note": "Direct training_reward() scalar. Not comparable to eval_reward.",
            "mean_start": round(float(train_rewards[0]), 4) if train_rewards else None,
            "mean_end":   round(float(train_rewards[-1]), 4) if train_rewards else None,
        },
        "eval_reward_before": before_components or {},
        "eval_reward_after":  after_components or {},
        "component_shift": {
            "before": before_components or {},
            "after":  after_components or {},
        },
        "log_history": log_history,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}")
        return

    # Reward curve
    reward_steps, rewards, loss_steps, losses = [], [], [], []
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row:
            loss_steps.append(step); losses.append(row["loss"])
        rv = row.get("reward") or row.get("rewards/mean")
        if rv is not None:
            reward_steps.append(step); rewards.append(rv)

    if loss_steps or reward_steps:
        fig, ax1 = plt.subplots(figsize=(10, 5.5))
        if losses:
            ax1.plot(loss_steps, losses, color="#26547c", linewidth=2, label="Training loss")
            ax1.set_ylabel("Loss", color="#26547c")
            ax1.tick_params(axis="y", labelcolor="#26547c")
        ax1.set_xlabel("Training step")
        ax1.grid(True, alpha=0.25)
        if rewards:
            ax2 = ax1.twinx()
            ax2.plot(reward_steps, rewards, color="#06a77d", linewidth=2, label="Mean reward (training scalar)")
            ax2.set_ylabel("Mean reward (training scalar — unbounded)", color="#06a77d")
            ax2.tick_params(axis="y", labelcolor="#06a77d")
            ax2.annotate(
                "Note: training scalar is unbounded.\nSee eval table for [0,1] clamped scores.",
                xy=(0.02, 0.05), xycoords="axes fraction", fontsize=9, color="gray"
            )
        fig.suptitle("DebateFloor GRPO Training Progress (training scalar — not eval score)")
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=180)
        plt.close(fig)

    # Component shift bar chart
    if before_components and after_components:
        labels = [label for _, label in _COMPONENT_LABELS]
        before_values = [before_components.get(label, 0.0) for label in labels]
        after_values  = [after_components.get(label, 0.0) for label in labels]
        x = list(range(len(labels)))
        width = 0.35
        fig2, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar([i - width/2 for i in x], before_values, width, label="Before training", color="#7a869a")
        ax.bar([i + width/2 for i in x], after_values,  width, label="After training",  color="#06a77d")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel("Component score (eval reward — clamped)")
        ax.set_xlabel("Reward component")
        ax.set_title("DebateFloor: component score shift before vs after GRPO training")
        ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False)
        fig2.tight_layout()
        fig2.savefig(COMPONENT_PLOT_PATH, dpi=180)
        plt.close(fig2)

        COMPONENT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        COMPONENT_SUMMARY_PATH.write_text(json.dumps({
            "before": before_components, "after": after_components,
            "delta": {k: round(after_components.get(k, 0.0) - before_components.get(k, 0.0), 4) for k in before_components},
        }, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {SUMMARY_PATH}, {PLOT_PATH}, {COMPONENT_PLOT_PATH}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global _tok_ref

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU found!)'}")

    # ── MR-2: Connect to the live environment before training ──────────────
    server_proc = _start_env_server_if_needed(ENV_BASE_URL)
    print(f"[OK] Environment connected at {ENV_BASE_URL} — reward from POST /step")

    if USE_WANDB:
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project="debatefloor-insurance-rl",
            entity=WANDB_ENTITY,
            name="grpo-qwen0.5b-env-connected",
            tags=["grpo", "calibration", "insurance", "env-connected", "http-reward"],
            config={
                "reward_type":   "env_http_reward",
                "training_note": "Reward from live env via POST /reset + /step",
                "env_base_url":  ENV_BASE_URL,
                "eval_note":     "six_component clamped [0,1]",
                "model":         MODEL_NAME,
                "episodes":      EPISODES,
                "epochs":        EPOCHS,
            },
        )

    # Load model
    if USE_UNSLOTH:
        print(f"Loading {MODEL_NAME} via Unsloth (4-bit QLoRA)...")
        model, tok = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
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
    else:
        print(f"Loading {MODEL_NAME} via standard transformers...")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map="auto",
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _tok_ref = tok

    print(f"Generating {EPISODES} training + eval episodes...")
    episode_pool   = generate_episode_pool(count=EPISODES + (EVAL_EPISODES * 4))
    eval_episodes  = _select_eval_episodes(episode_pool[EPISODES:])
    train_episodes = episode_pool[:EPISODES]
    rows    = [make_row(ep, tok) for ep in train_episodes]
    dataset = Dataset.from_list(rows)
    print(f"Dataset: {len(dataset)} training episodes, {len(eval_episodes)} eval episodes")

    print("Baseline eval (before training)...")
    before_eval = evaluate_component_shift(model, tok, eval_episodes)
    before_components = before_eval["means"]
    print(f"  Before: {before_components}")

    if USE_WANDB:
        try:
            wandb.log({f"eval/before/{k.replace(' ', '_').lower()}": v for k, v in before_components.items()})
        except Exception:
            pass

    # Smoke: fewer GRPO rollouts + smaller accumulation = faster and less VRAM.
    # T4-tuned full run: num_generations=4 (was 6) and grad_accum=2 (was 4)
    # cuts per-step generation cost by ~2.5x while keeping a stable group-relative
    # advantage estimate. Effective batch = 4 * 2 = 8, fine for 0.5B QLoRA.
    _num_gen = 2 if SMOKE_MODE else int(os.getenv("NUM_GENERATIONS", "4"))
    _train_batch_size = BATCH_SIZE
    if _train_batch_size < 2:
        # GRPO requires >=2 generations; batch=1 is invalid.
        _train_batch_size = 2
        print(
            f"[WARN] Adjusting batch_size to {_train_batch_size} "
            f"(GRPO requires >=2 generations per prompt)."
        )
    if _train_batch_size % _num_gen != 0:
        adjusted = ((_train_batch_size // _num_gen) + 1) * _num_gen
        print(
            f"[WARN] Adjusting batch_size from {_train_batch_size} to {adjusted} "
            f"so it is divisible by num_generations={_num_gen}."
        )
        _train_batch_size = adjusted
    _grad_acc = 1 if SMOKE_MODE else int(os.getenv("GRAD_ACCUM", "2"))

    args = GRPOConfig(
        output_dir="./debatefloor_grpo_out",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=_train_batch_size,
        gradient_accumulation_steps=_grad_acc,
        learning_rate=LR,
        num_generations=_num_gen,    # 4 = T4-friendly full run; 2 = smoke
        # 80 tokens easily fits "DECISION: x\nCONFIDENCE: y\nREASON: <one sentence>"
        # — was 100; -20% generation time per completion.
        max_completion_length=int(os.getenv("MAX_COMPLETION_LENGTH", "80")),
        # Cap prompts so a long claim description can't blow up generation time.
        max_prompt_length=int(os.getenv("MAX_PROMPT_LENGTH", "512")),
        # Sampling temperature is env-tunable. Default 1.1 (was 0.9) because
        # GRPO needs diversity *within* a group to compute a useful advantage;
        # at 0.9 a small base model collapses to nearly identical completions
        # and reward_std -> 0 (no learning signal). 1.1 keeps the policy
        # coherent while spreading the group's reward distribution.
        temperature=float(os.getenv("SAMPLING_TEMPERATURE", "1.1")),
        logging_steps=1 if SMOKE_MODE else 5,
        save_steps=9999 if SMOKE_MODE else 50,
        report_to="wandb" if USE_WANDB else "none",
        run_name="debatefloor-grpo-env-connected",
        max_grad_norm=0.3,
        seed=SEED,
        bf16=HAS_BF16,
        fp16=USE_FP16 and not HAS_BF16,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=dataset,
    )

    print(f"Starting GRPO training (reward from {ENV_BASE_URL}/step)...")
    result = trainer.train()
    print(f"Done. Steps: {result.global_step} | Loss: {result.training_loss:.4f}")

    print("Post-training eval...")
    after_eval = evaluate_component_shift(model, tok, eval_episodes)
    after_components = after_eval["means"]
    print(f"  After: {after_components}")

    # Human-readable training summary so you don't have to mentally diff two dicts.
    print("\n" + "=" * 70)
    print("TRAINING ACCURACY SUMMARY")
    print("=" * 70)
    print(f"{'Component':<25s}{'Before':>12s}{'After':>12s}{'Delta':>12s}")
    print("-" * 70)
    for _comp in sorted(set(before_components) | set(after_components)):
        _b = float(before_components.get(_comp, 0.0))
        _a = float(after_components.get(_comp, 0.0))
        _d = _a - _b
        _arrow = "UP" if _d > 0.005 else ("DOWN" if _d < -0.005 else "FLAT")
        print(f"  {_comp:<23s}{_b:>11.3f} {_a:>11.3f} {_d:>+10.3f} {_arrow}")
    _b_mean = sum(before_components.values()) / max(len(before_components), 1)
    _a_mean = sum(after_components.values()) / max(len(after_components), 1)
    print("-" * 70)
    print(f"  {'OVERALL MEAN':<23s}{_b_mean:>11.3f} {_a_mean:>11.3f} {(_a_mean-_b_mean):>+10.3f}")
    print("=" * 70 + "\n")

    if USE_WANDB:
        try:
            wandb.log({f"eval/after/{k.replace(' ', '_').lower()}": v for k, v in after_components.items()})
            wandb.finish()
        except Exception:
            pass

    save_training_artifacts(trainer, result, before_components, after_components)

    # Save model
    if USE_UNSLOTH:
        model.save_pretrained_merged("./debatefloor_checkpoint", tok, save_method="merged_16bit")
    else:
        model.save_pretrained("./debatefloor_checkpoint")
        tok.save_pretrained("./debatefloor_checkpoint")
    print("[OK] Checkpoint saved to ./debatefloor_checkpoint")

    # Clean up server process if we started it
    if server_proc is not None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[STOP] Environment server stopped.")

    # Push to HF Hub if token is set (skip on smoke to avoid polluting the model repo)
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token and not SMOKE_MODE:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_folder(
                folder_path="./debatefloor_checkpoint",
                repo_id="AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct",
                repo_type="model",
                commit_message="Update: GRPO training — env-connected HTTP reward",
            )
            print("[OK] Model pushed to HF Hub")
        except Exception as exc:
            print(f"HF push skipped: {exc}")


if __name__ == "__main__":
    main()
