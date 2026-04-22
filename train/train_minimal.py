"""
train_minimal.py — DebateFloor minimal GRPO training (no Unsloth)

Runs on free Colab T4 in ~15 minutes.
Produces a real WandB reward curve using pure TRL GRPOTrainer.

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

sys.path.insert(0, ".")

import wandb
from datasets import Dataset
from server.calibration_grader import CALIBRATION_MATRIX, training_reward
from server.claim_generator import generate_episode_pool
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"  # tiny — runs on T4 in 15 min
EPISODES     = 100
EPOCHS       = 2
BATCH_SIZE   = 2
LR           = 5e-6
SEED         = 42
USE_WANDB    = bool(os.getenv("WANDB_API_KEY", ""))
WANDB_KEY    = os.getenv("WANDB_API_KEY", "")
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


def main():
    if USE_WANDB:
        wandb.login(key=WANDB_KEY)
        wandb.init(project="debatefloor-insurance-rl", name="grpo-qwen0.5b")

    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

    print(f"Generating {EPISODES} training episodes...")
    episodes = generate_episode_pool(count=EPISODES)
    rows = [make_row(ep, tok) for ep in episodes]
    dataset = Dataset.from_list(rows)
    print(f"Dataset ready: {len(dataset)} episodes")

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
        bf16=False,
        fp16=True,
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

    if USE_WANDB:
        wandb.finish()
        print("WandB run complete.")

    model.save_pretrained("./debatefloor_checkpoint")
    tok.save_pretrained("./debatefloor_checkpoint")
    print("Checkpoint saved to ./debatefloor_checkpoint")


if __name__ == "__main__":
    main()
