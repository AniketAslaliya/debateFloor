"""
train_minimal.py — DebateFloor minimal GRPO training (no Unsloth)

Runs on free Colab T4 in ~15 minutes.
Produces a real WandB reward curve using pure TRL GRPOTrainer.
Also saves a before-vs-after component score plot from a held-out eval sweep.

Environment variables:
  WANDB_API_KEY  Set to your WandB key to enable public training logs.
  WANDB_ENTITY   Your WandB username/org (default: 'aniketaslaliya').
                 Public project: wandb.ai/{WANDB_ENTITY}/debatefloor-insurance-rl

Usage (Colab):
  !git clone https://github.com/AniketAslaliya/debateFloor && cd debateFloor
  !pip install trl>=0.9.0 transformers peft accelerate datasets wandb requests
  !python train/train_minimal.py

Or with WandB:
  import os; os.environ['WANDB_API_KEY'] = 'your_key'
  !python train/train_minimal.py
"""

import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, ".")

import wandb
from datasets import Dataset
from server.calibration_grader import CALIBRATION_MATRIX, training_reward
from server.claim_generator import generate_episode_pool
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"  # tiny — runs on T4 in 15 min
EPISODES    = 100
EVAL_EPISODES = 9
EPOCHS      = 2
BATCH_SIZE  = 2
LR          = 5e-6
SEED        = 42
USE_WANDB   = bool(os.getenv("WANDB_API_KEY", ""))
WANDB_KEY   = os.getenv("WANDB_API_KEY", "")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "aniketaslaliya-lnmiit")  # Set to your WandB username
PLOT_PATH   = Path("docs/reward_curve.svg")
COMPONENT_PLOT_PATH = Path("docs/component_shift.svg")
SUMMARY_PATH = Path("reports/training_summary.json")
COMPONENT_SUMMARY_PATH = Path("reports/component_shift_summary.json")

# T4 supports float16 but NOT bfloat16 for AMP grad scaling
HAS_BF16    = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_FP16    = torch.cuda.is_available() and not HAS_BF16
DTYPE       = torch.bfloat16 if HAS_BF16 else torch.float16
# ────────────────────────────────────────────────────────────────────

SYSTEM = (
    "You are an expert insurance fraud investigator.\n"
    "Analyze the claim and respond EXACTLY in this format:\n"
    "DECISION: <approve_claim|deny_claim|escalate_to_human>\n"
    "CONFIDENCE: <HIGH|MED|LOW>\n"
    "REASON: <one sentence>\n\n"
    "HIGH = certain. MED = likely but some doubt. LOW = ambiguous, expert needed.\n"
    "WARNING: HIGH confidence on a wrong answer is the worst possible outcome (-0.8)."
)

DECISION_RE   = re.compile(r"DECISION:\s*(approve_claim|deny_claim|escalate_to_human)", re.I)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(HIGH|MED|LOW)", re.I)
REASON_RE = re.compile(r"REASON:\s*(.*)", re.I | re.S)

_EVAL_TASKS = ("clean_claim", "contradictory_claim", "distribution_shift_claim")
_COMPONENT_LABELS = [
    ("fraud_detection_score", "Fraud detection"),
    ("decision_accuracy", "Decision accuracy"),
    ("evidence_quality_score", "Evidence quality"),
    ("calibration_score", "Calibration"),
]


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
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": ep_to_prompt(ep)},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "prompt":           prompt,
        "ground_truth":     ep.ground_truth,
        "fraud_type":       ep.fraud_type,
        "expected_signals": json.dumps(ep.expected_fraud_signals),
    }


def reward_fn(completions, ground_truth, expected_signals, **kwargs):
    rewards = []
    for text, gt, sigs_json in zip(completions, ground_truth, expected_signals):
        if isinstance(text, list):
            text = text[0].get("content", "") if text else ""
        dm = DECISION_RE.search(text)
        cm = CONFIDENCE_RE.search(text)
        if not dm or not cm:
            rewards.append(-0.2)
            continue
        decision   = dm.group(1).lower()
        confidence = cm.group(1).upper()
        sigs       = json.loads(sigs_json) if sigs_json else []
        legit      = sum(1 for s in sigs if s.replace("_", " ") in text.lower())
        r = training_reward(decision, confidence, gt, legit, step_num=1, done=True)
        rewards.append(r)
    return rewards


def _select_eval_episodes(episodes):
    selected = []
    counts = {task_id: 0 for task_id in _EVAL_TASKS}
    per_task = max(1, EVAL_EPISODES // len(_EVAL_TASKS))

    for episode in episodes:
        task_id = getattr(episode, "task_id", None)
        if task_id not in counts:
            continue
        if counts[task_id] >= per_task:
            continue
        selected.append(episode)
        counts[task_id] += 1
        if all(count >= per_task for count in counts.values()):
            break

    return selected


def _extract_completion_fields(text: str) -> dict:
    decision_match = DECISION_RE.search(text or "")
    confidence_match = CONFIDENCE_RE.search(text or "")
    reason_match = REASON_RE.search(text or "")
    return {
        "decision": decision_match.group(1).lower() if decision_match else None,
        "confidence": confidence_match.group(1).upper() if confidence_match else None,
        "reason": (reason_match.group(1).strip() if reason_match else ""),
    }


def _score_completion(episode, completion_text: str) -> dict:
    parsed = _extract_completion_fields(completion_text)
    completion_lc = (completion_text or "").lower()
    reason_lc = parsed["reason"].lower()
    expected = list(getattr(episode, "expected_fraud_signals", []) or [])

    if expected:
        fraud_hits = sum(
            1
            for signal in expected
            if signal.replace("_", " ") in completion_lc or signal.replace("_", " ") in reason_lc
        )
        fraud_detection_score = fraud_hits / float(len(expected))
        evidence_quality_score = sum(
            1
            for signal in expected
            if signal.replace("_", " ") in reason_lc
        ) / float(len(expected))
    else:
        fraud_detection_score = 1.0 if parsed["decision"] == getattr(episode, "ground_truth", None) else 0.0
        evidence_quality_score = 1.0 if parsed["reason"] else 0.0

    decision_correct = parsed["decision"] == getattr(episode, "ground_truth", None)
    calibration_score = CALIBRATION_MATRIX.get((parsed["confidence"], decision_correct), 0.0)
    decision_accuracy = 1.0 if decision_correct else 0.0

    return {
        "fraud_detection_score": fraud_detection_score,
        "decision_accuracy": decision_accuracy,
        "evidence_quality_score": evidence_quality_score,
        "calibration_score": calibration_score,
    }


def _generate_completion(model, tok, prompt: str) -> str:
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated = output_ids[0][prompt_length:]
    return tok.decode(generated, skip_special_tokens=True)


def evaluate_component_shift(model, tok, episodes):
    rows = []
    for episode in episodes:
        prompt = make_row(episode, tok)["prompt"]
        completion = _generate_completion(model, tok, prompt)
        scores = _score_completion(episode, completion)
        rows.append({
            "task_id": getattr(episode, "task_id", "unknown"),
            "fraud_type": getattr(episode, "fraud_type", "unknown"),
            "completion": completion,
            **scores,
        })

    means = {
        label: (mean(row[key] for row in rows) if rows else 0.0)
        for key, label in _COMPONENT_LABELS
    }
    return {"rows": rows, "means": means}


def save_training_artifacts(trainer, result, before_components=None, after_components=None) -> None:
    """Save local training evidence for the README and submission review."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPONENT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPONENT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_history = list(getattr(trainer.state, "log_history", []) or [])
    summary = {
        "model": MODEL_NAME,
        "episodes": EPISODES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "global_step": int(getattr(result, "global_step", 0) or 0),
        "training_loss": float(getattr(result, "training_loss", 0.0) or 0.0),
        "log_history": log_history,
        "component_shift": {
            "before": before_components or {},
            "after": after_components or {},
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot generation: matplotlib unavailable ({exc})")
        return

    loss_steps = []
    losses = []
    reward_steps = []
    rewards = []
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row:
            loss_steps.append(step)
            losses.append(row["loss"])
        reward_value = row.get("reward")
        if reward_value is None:
            reward_value = row.get("rewards/mean")
        if reward_value is not None:
            reward_steps.append(step)
            rewards.append(reward_value)

    if not loss_steps and not reward_steps:
        print(f"Saved training summary to {SUMMARY_PATH}; no plottable log rows found.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    if losses:
        ax1.plot(loss_steps, losses, color="#26547c", linewidth=2, label="Training loss")
        ax1.set_ylabel("Loss", color="#26547c")
        ax1.tick_params(axis="y", labelcolor="#26547c")
    ax1.set_xlabel("Training step")
    ax1.grid(True, alpha=0.25)

    if rewards:
        ax2 = ax1.twinx()
        ax2.plot(reward_steps, rewards, color="#06a77d", linewidth=2, label="Mean reward")
        ax2.set_ylabel("Mean reward", color="#06a77d")
        ax2.tick_params(axis="y", labelcolor="#06a77d")

    fig.suptitle("DebateFloor GRPO Training Progress")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    if before_components and after_components:
        labels = [label for _, label in _COMPONENT_LABELS]
        before_values = [before_components.get(label, 0.0) for label in labels]
        after_values = [after_components.get(label, 0.0) for label in labels]

        x_positions = list(range(len(labels)))
        width = 0.35
        fig2, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar([x - width / 2 for x in x_positions], before_values, width=width, label="Before training", color="#7a869a")
        ax.bar([x + width / 2 for x in x_positions], after_values, width=width, label="After training", color="#06a77d")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Component score")
        ax.set_xlabel("Reward component")
        ax.set_title("DebateFloor component score shift before vs after training")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig2.tight_layout()
        fig2.savefig(COMPONENT_PLOT_PATH, dpi=180)
        plt.close(fig2)

        COMPONENT_SUMMARY_PATH.write_text(
            json.dumps(
                {
                    "before": before_components,
                    "after": after_components,
                    "delta": {
                        label: round(after_components.get(label, 0.0) - before_components.get(label, 0.0), 4)
                        for label in before_components
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Saved training summary to {SUMMARY_PATH}")
    print(f"Saved reward curve to {PLOT_PATH}")
    if before_components and after_components:
        print(f"Saved component shift plot to {COMPONENT_PLOT_PATH}")
        print(f"Saved component shift summary to {COMPONENT_SUMMARY_PATH}")


def main():
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"dtype={DTYPE} | fp16={USE_FP16} | bf16={HAS_BF16}")

    if USE_WANDB:
        wandb.login(key=WANDB_KEY)
        wandb.init(
            project="debatefloor-insurance-rl",
            entity=WANDB_ENTITY,
            name="grpo-qwen0.5b",
            notes="DebateFloor GRPO training: calibrated confidence via 3x2 matrix reward. "
                  "See github.com/AniketAslaliya/debateFloor for environment details.",
            tags=["grpo", "calibration", "insurance", "openenv"],
        )

    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Force explicit dtype — never "auto" on T4 (auto picks bfloat16, breaks fp16 AMP)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )

    print(f"Generating {EPISODES} training episodes plus held-out eval episodes...")
    episode_pool = generate_episode_pool(count=EPISODES + (EVAL_EPISODES * 4))
    eval_episodes = _select_eval_episodes(episode_pool[EPISODES:])
    training_episodes = episode_pool[:EPISODES]
    rows = [make_row(ep, tok) for ep in training_episodes]
    dataset = Dataset.from_list(rows)
    print(f"Dataset ready: {len(dataset)} episodes")

    print(f"Evaluating baseline component scores on {len(eval_episodes)} held-out episodes...")
    before_eval = evaluate_component_shift(model, tok, eval_episodes)
    before_components = before_eval["means"]

    args = GRPOConfig(
        output_dir="./debatefloor_grpo_out",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        num_generations=4,
        max_completion_length=100,
        temperature=0.9,
        logging_steps=5,
        save_steps=50,
        report_to="wandb" if USE_WANDB else "none",
        run_name="debatefloor-grpo-qwen0.5b",
        max_grad_norm=0.3,
        seed=SEED,
        bf16=HAS_BF16,
        fp16=USE_FP16,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    result = trainer.train()
    print(f"Done. Steps: {result.global_step} | Loss: {result.training_loss:.4f}")

    print("Evaluating trained component scores on the same held-out episodes...")
    after_eval = evaluate_component_shift(model, tok, eval_episodes)
    after_components = after_eval["means"]

    if USE_WANDB:
        wandb.finish()
        print("WandB run complete.")

    save_training_artifacts(trainer, result, before_components, after_components)

    model.save_pretrained("./debatefloor_checkpoint")
    tok.save_pretrained("./debatefloor_checkpoint")
    print("Checkpoint saved to ./debatefloor_checkpoint")


if __name__ == "__main__":
    main()
