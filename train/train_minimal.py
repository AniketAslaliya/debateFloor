"""
train_minimal.py — DebateFloor GRPO training (Unsloth + live environment)

Key changes from original:
  - Uses Unsloth FastLanguageModel (CRITICAL-1)
  - reward_fn calls live /reset + /step HTTP endpoints (FATAL-1)
  - Environment server started in background before training (FATAL-1)
  - num_generations=8 for better reward variance (HIGH-4)
  - Separate wandb logging for train vs eval reward (CRITICAL-2)

Usage (Colab):
  !git clone https://github.com/AniketAslaliya/debateFloor && cd debateFloor
  !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install trl>=0.12.0 transformers peft accelerate datasets wandb requests matplotlib
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

import requests
import torch

sys.path.insert(0, ".")

import wandb
from datasets import Dataset
from server.calibration_grader import CALIBRATION_MATRIX, training_reward
from server.claim_generator import generate_episode_pool
from trl import GRPOConfig, GRPOTrainer

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
EPISODES     = 300   # was 100 — increased for meaningful GRPO learning (HIGH-4)
EVAL_EPISODES = 9
EPOCHS       = 2
BATCH_SIZE   = 2
LR           = 5e-6
SEED         = 42
USE_WANDB    = bool(os.getenv("WANDB_API_KEY", ""))
WANDB_KEY    = os.getenv("WANDB_API_KEY", "")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "aniketaslaliya-lnmiit")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
PLOT_PATH    = Path("docs/reward_curve.svg")
COMPONENT_PLOT_PATH = Path("docs/component_shift.svg")
SUMMARY_PATH = Path("reports/training_summary.json")
COMPONENT_SUMMARY_PATH = Path("reports/component_shift_summary.json")

# Try Unsloth first; fall back gracefully if not installed (CRITICAL-1)
try:
    from unsloth import FastLanguageModel
    USE_UNSLOTH = True
    print("✅ Unsloth available — using FastLanguageModel + QLoRA")
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    USE_UNSLOTH = False
    print("⚠️  Unsloth not found — falling back to standard transformers")

HAS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16  = torch.cuda.is_available() and not HAS_BF16
DTYPE     = torch.bfloat16 if HAS_BF16 else torch.float16
# ───────────────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are an expert insurance fraud investigator.\n"
    "Analyze the claim and respond EXACTLY in this format:\n"
    "DECISION: <approve_claim|deny_claim|escalate_to_human>\n"
    "CONFIDENCE: <HIGH|MED|LOW>\n"
    "REASON: <one sentence citing specific evidence>\n\n"
    "HIGH = certain. MED = likely but some doubt. LOW = ambiguous, expert needed.\n"
    "WARNING: HIGH confidence on a wrong answer is the worst possible outcome (-0.8)."
)

DECISION_RE   = re.compile(r"DECISION:\s*(approve_claim|deny_claim|escalate_to_human)", re.I)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(HIGH|MED|LOW)", re.I)
REASON_RE     = re.compile(r"REASON:\s*(.*)", re.I | re.S)

_EVAL_TASKS = ("clean_claim", "contradictory_claim", "distribution_shift_claim")
_COMPONENT_LABELS = [
    ("fraud_detection_score", "Fraud detection"),
    ("decision_accuracy",     "Decision accuracy"),
    ("evidence_quality_score","Evidence quality"),
    ("calibration_score",     "Calibration"),
]

# Module-level refs so reward_fn can access model/tok (set in main())
_model_ref = None
_tok_ref   = None


# ── Environment startup ─────────────────────────────────────────────────────

def _start_env_server() -> subprocess.Popen | None:
    """Start the environment server as a background process. Returns Popen or None."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "0.0.0.0", "--port", "7860", "--log-level", "error"],
            cwd=os.getcwd(),
        )
        print(f"Started environment server (PID={proc.pid}). Waiting for startup...")
        return proc
    except Exception as e:
        print(f"Could not start server automatically: {e}")
        return None


def _wait_for_env(base_url: str = ENV_BASE_URL, retries: int = 15) -> None:
    """Block until the environment /health responds 200."""
    for i in range(retries):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"✅ Environment ready at {base_url}")
                return
        except Exception:
            pass
        print(f"  Waiting for environment... ({i+1}/{retries})")
        time.sleep(4)
    raise RuntimeError(f"Environment not reachable at {base_url} after {retries} retries")


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
        "task_id":          ep.task_id,   # FATAL-1: needed for /reset call
    }


# ── Live environment reward (FATAL-1 fix) ──────────────────────────────────

def _generate_completion_text(tok, prompt: str) -> str:
    """Generate text from the current _model_ref."""
    global _model_ref, _tok_ref
    device = next(_model_ref.parameters()).device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = _model_ref.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def run_episode_via_http(prompt: str, task_id: str) -> float:
    """Run one episode against the live /reset + /step HTTP API. Returns reward."""
    try:
        # 1. Reset
        reset_r = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "seed": random.randint(0, 9999)},
            timeout=15,
        )
        reset_r.raise_for_status()
        session_id = reset_r.json()["session_id"]

        # 2. Generate
        text = _generate_completion_text(_tok_ref, prompt)

        # 3. Parse
        dm = DECISION_RE.search(text)
        cm = CONFIDENCE_RE.search(text)
        rm = REASON_RE.search(text)
        if not dm or not cm:
            return -0.2  # format penalty

        action = {
            "action_type": dm.group(1).lower(),
            "confidence":  cm.group(1).upper(),
            "reason":      (rm.group(1).strip() if rm else ""),
            "reasoning":   (rm.group(1).strip() if rm else ""),
        }

        # 4. Step
        step_r = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action, "session_id": session_id},
            timeout=15,
        )
        step_r.raise_for_status()
        return float(step_r.json().get("reward", -0.1))

    except Exception as exc:
        print(f"HTTP rollout error: {exc}")
        return -0.1


def reward_fn(completions, prompts, task_ids, **kwargs):
    """
    GRPO reward function — calls the live environment for each completion. (FATAL-1)
    Each completion is one rollout; reward comes from /step response.
    """
    rewards = []
    for completion_list, prompt, task_id in zip(completions, prompts, task_ids):
        text = completion_list[0].get("content", "") if isinstance(completion_list, list) else str(completion_list)
        reward = run_episode_via_http(prompt, task_id)
        rewards.append(reward)

    # Log reward variance (HIGH-4 — detect zero-gradient situations)
    if len(rewards) > 1:
        import statistics
        variance = statistics.variance(rewards)
        if variance < 0.01:
            print(f"  ⚠️  Low reward variance ({variance:.4f}) — GRPO gradient may be near zero")
        if USE_WANDB:
            wandb.log({"train/reward_variance": variance})

    return rewards


# ── Eval helpers ────────────────────────────────────────────────────────────

def _extract_completion_fields(text: str) -> dict:
    dm = DECISION_RE.search(text or "")
    cm = CONFIDENCE_RE.search(text or "")
    rm = REASON_RE.search(text or "")
    return {
        "decision":   dm.group(1).lower() if dm else None,
        "confidence": cm.group(1).upper() if cm else None,
        "reason":     (rm.group(1).strip() if rm else ""),
    }


def _generate_completion(model, tok, prompt: str) -> str:
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=96, do_sample=False,
            temperature=0.0, pad_token_id=tok.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def _score_completion(episode, completion_text: str) -> dict:
    parsed = _extract_completion_fields(completion_text)
    completion_lc = (completion_text or "").lower()
    reason_lc = parsed["reason"].lower()
    expected = list(getattr(episode, "expected_fraud_signals", []) or [])

    if expected:
        fraud_hits = sum(1 for s in expected if s.replace("_", " ") in completion_lc or s.replace("_", " ") in reason_lc)
        fraud_detection_score = fraud_hits / float(len(expected))
        evidence_quality_score = sum(1 for s in expected if s.replace("_", " ") in reason_lc) / float(len(expected))
    else:
        fraud_detection_score = 1.0 if parsed["decision"] == getattr(episode, "ground_truth", None) else 0.0
        evidence_quality_score = 1.0 if parsed["reason"] else 0.0

    decision_correct = parsed["decision"] == getattr(episode, "ground_truth", None)
    calibration_score = CALIBRATION_MATRIX.get((parsed["confidence"], decision_correct), 0.0)
    decision_accuracy = 1.0 if decision_correct else 0.0

    return {
        "fraud_detection_score": fraud_detection_score,
        "decision_accuracy":     decision_accuracy,
        "evidence_quality_score": evidence_quality_score,
        "calibration_score":     calibration_score,
    }


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

    # Extract reward values from log history
    train_rewards = [r.get("reward") or r.get("rewards/mean") for r in log_history
                     if r.get("reward") is not None or r.get("rewards/mean") is not None]

    summary = {
        "model": MODEL_NAME,
        "episodes": EPISODES, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "global_step": int(getattr(result, "global_step", 0) or 0),
        "training_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
        # CRITICAL-2: explicitly separate train scalar from eval [0,1] score
        "training_reward_curve": {
            "type": "unbounded_scalar",
            "note": "Used for GRPO gradient stability only. Not comparable to eval_reward.",
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

    # Reward curve (MEDIUM-2: proper axis labels + annotation)
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

    print(f"✅ Saved: {SUMMARY_PATH}, {PLOT_PATH}, {COMPONENT_PLOT_PATH}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global _model_ref, _tok_ref

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    if USE_WANDB:
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project="debatefloor-insurance-rl",
            entity=WANDB_ENTITY,
            name="grpo-qwen0.5b-live-env",
            tags=["grpo", "calibration", "insurance", "openenv", "unsloth"],
            config={
                "reward_type": "live_environment_http",          # CRITICAL-2
                "training_reward_note": "unbounded scalar from /step API",
                "eval_reward_note": "six_component clamped [0,1]",
                "never_mix": True,
            },
        )

    # Start environment server (FATAL-1)
    server_proc = _start_env_server()
    _wait_for_env(ENV_BASE_URL)

    # Load model with Unsloth if available (CRITICAL-1)
    if USE_UNSLOTH:
        print(f"Loading {MODEL_NAME} via Unsloth (4-bit QLoRA)...")
        model, tok = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=512,
            dtype=None,        # auto-detect
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
        print(f"Loading {MODEL_NAME} via transformers...")
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, device_map="auto")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Set module-level refs for reward_fn (FATAL-1)
    _model_ref = model
    _tok_ref   = tok

    print(f"Generating {EPISODES} training + eval episodes...")
    episode_pool   = generate_episode_pool(count=EPISODES + (EVAL_EPISODES * 4))
    eval_episodes  = _select_eval_episodes(episode_pool[EPISODES:])
    train_episodes = episode_pool[:EPISODES]
    rows    = [make_row(ep, tok) for ep in train_episodes]
    dataset = Dataset.from_list(rows)
    print(f"Dataset: {len(dataset)} training episodes, {len(eval_episodes)} eval episodes")

    print("Baseline eval...")
    before_eval = evaluate_component_shift(model, tok, eval_episodes)
    before_components = before_eval["means"]
    print(f"  Before: {before_components}")

    # Log baseline to WandB (CRITICAL-2: separate eval metrics)
    if USE_WANDB:
        wandb.log({f"eval/before/{k.replace(' ', '_').lower()}": v for k, v in before_components.items()})

    args = GRPOConfig(
        output_dir="./debatefloor_grpo_out",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        num_generations=8,          # HIGH-4: was 4, more variance = stronger GRPO gradient
        max_completion_length=100,
        temperature=0.9,
        logging_steps=5,
        save_steps=50,
        report_to="wandb" if USE_WANDB else "none",
        run_name="debatefloor-grpo-live-env",
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

    print("Starting GRPO training against live environment...")
    result = trainer.train()
    print(f"Done. Steps: {result.global_step} | Loss: {result.training_loss:.4f}")

    print("Post-training eval...")
    after_eval = evaluate_component_shift(model, tok, eval_episodes)
    after_components = after_eval["means"]
    print(f"  After: {after_components}")

    if USE_WANDB:
        wandb.log({f"eval/after/{k.replace(' ', '_').lower()}": v for k, v in after_components.items()})
        wandb.finish()

    save_training_artifacts(trainer, result, before_components, after_components)

    # Save model (CRITICAL-1: use Unsloth safe merge if available)
    if USE_UNSLOTH:
        model.save_pretrained_merged("./debatefloor_checkpoint", tok, save_method="merged_16bit")
    else:
        model.save_pretrained("./debatefloor_checkpoint")
        tok.save_pretrained("./debatefloor_checkpoint")
    print("✅ Checkpoint saved to ./debatefloor_checkpoint")

    # Push to HF Hub if token is set
    hf_token = os.getenv("HF_TOKEN", "")
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_folder(
                folder_path="./debatefloor_checkpoint",
                repo_id="AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct",
                repo_type="model",
                commit_message="Update: GRPO training with live environment connection",
            )
            print("✅ Model pushed to HF Hub")
        except Exception as exc:
            print(f"HF push skipped: {exc}")

    # Stop environment server
    if server_proc:
        server_proc.terminate()
        print("Environment server stopped.")


if __name__ == "__main__":
    main()
