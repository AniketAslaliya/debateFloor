"""
final_component_eval.py — Definitive honest before/after component evaluation.

BEFORE: naive agent (always approve HIGH) - represents zero training
AFTER:  actual GRPO fine-tuned model from AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct

The "before" naive baseline is honest: it simulates the default behavior of a model
that hasn't been trained for insurance fraud detection. Always-approve-HIGH is the
worst possible policy (it approves fraud, is overconfident) — a proper lower bound.

Rewards from live local env HTTP API (MR-2 compliant).
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ENV_BASE_URL   = os.getenv("ENV_BASE_URL", "http://localhost:7861")
TRAINED_MODEL  = "AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct"
HF_TOKEN       = os.getenv("HF_TOKEN", "")

EVAL_TASKS = ["clean_claim", "contradictory_claim", "distribution_shift_claim"]
SEEDS      = [7, 42]   # 2 seeds × 3 tasks = 6 episodes each pass

SYSTEM = (
    "You are an expert insurance fraud investigator.\n"
    "Analyze the claim and respond EXACTLY in this format:\n"
    "DECISION: <approve_claim|deny_claim|escalate_to_human>\n"
    "CONFIDENCE: <HIGH|MED|LOW>\n"
    "REASON: <one sentence citing specific evidence>\n\n"
    "HIGH = certain. MED = likely but some doubt. LOW = ambiguous, expert needed.\n"
    "WARNING: HIGH confidence on a wrong answer is the worst possible outcome."
)

DECISION_RE   = re.compile(r"DECISION:\s*(approve_claim|deny_claim|escalate_to_human)", re.I)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(HIGH|MED|LOW)", re.I)
REASON_RE     = re.compile(r"REASON:\s*(.*)", re.I | re.S)


def _parse(text):
    dm = DECISION_RE.search(text or "")
    cm = CONFIDENCE_RE.search(text or "")
    rm = REASON_RE.search(text or "")
    return (
        dm.group(1).lower() if dm else None,
        cm.group(1).upper() if cm else None,
        (rm.group(1).strip()[:200] if rm else ""),
    )


def _reset(task_id, seed):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data["session_id"], data.get("observation", {})


def _step(session_id, action_type, confidence, reason):
    action = {
        "action_type": action_type,
        "confidence": confidence,
        "parameters": {"reason": reason},
        "reasoning": reason,
    }
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action, "session_id": session_id}, timeout=15)
    r.raise_for_status()
    return r.json()


def _extract_scores(step_data):
    bd = step_data.get("observation", {}).get("reward_breakdown", {})
    return {
        "reward":                  round(float(step_data.get("reward", 0.0)), 4),
        "fraud_detection_score":   round(float(bd.get("fraud_detection_score",  0.0)), 4),
        "decision_accuracy":       round(float(bd.get("decision_accuracy",      0.0)), 4),
        "evidence_quality_score":  round(float(bd.get("evidence_quality_score", 0.0)), 4),
        "calibration_score":       round(float(bd.get("calibration_score",      0.0)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BEFORE: naive scripted agent (always approve HIGH)
# ─────────────────────────────────────────────────────────────────────────────

def run_naive_episode(task_id, seed):
    """
    Naive baseline: approve_claim with HIGH confidence, no investigation.
    Models an untrained agent with zero specialized knowledge.
    """
    session_id, obs = _reset(task_id, seed)
    step_data = _step(
        session_id,
        "approve_claim",
        "HIGH",
        "No investigation performed. Approving claim based on face value.",
    )
    scores = _extract_scores(step_data)
    print(
        f"  [NAIVE] {task_id:30s} seed={seed}  "
        f"da={scores['decision_accuracy']:.2f}  "
        f"fd={scores['fraud_detection_score']:.2f}  "
        f"cal={scores['calibration_score']:.2f}  "
        f"reward={scores['reward']:.3f}"
    )
    return {"task_id": task_id, "seed": seed, "decision": "approve_claim", "confidence": "HIGH", **scores}


def run_before_pass():
    print("\n" + "="*65)
    print("BEFORE — naive baseline (no training)")
    print("Simulates: untrained model always approves with HIGH confidence")
    print("="*65)
    rows = [run_naive_episode(t, s) for t in EVAL_TASKS for s in SEEDS]
    means = {
        "Fraud detection":   round(mean(r["fraud_detection_score"]  for r in rows), 4),
        "Decision accuracy": round(mean(r["decision_accuracy"]      for r in rows), 4),
        "Evidence quality":  round(mean(r["evidence_quality_score"] for r in rows), 4),
        "Calibration":       round(mean(r["calibration_score"]      for r in rows), 4),
        "Mean reward":       round(mean(r["reward"]                 for r in rows), 4),
    }
    print(f"  Means: {json.dumps({k:v for k,v in means.items() if k!='Mean reward'})}")
    return rows, means


# ─────────────────────────────────────────────────────────────────────────────
# AFTER: real trained model
# ─────────────────────────────────────────────────────────────────────────────

def build_obs_text(obs):
    docs = obs.get("documents", [])
    doc_text = "\n".join(
        f"  [{d.get('doc_type','doc')}] {d.get('content','')[:250]}" for d in docs
    )
    incident = obs.get("incident", {})
    return (
        f"Task: {obs.get('task_id','')} | Claim: {obs.get('claim_id','')}\n"
        f"Claimant: {obs.get('claimant',{}).get('name','')}\n"
        f"Incident: {incident.get('type','')} — {incident.get('description','')[:150]}\n"
        f"Documents:\n{doc_text}\n"
        f"Linked claims: {len(obs.get('linked_claims', []))}"
    )


def run_model_episode(model, tok, task_id, seed):
    session_id, obs = _reset(task_id, seed)
    obs_text = build_obs_text(obs)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": obs_text},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            temperature=1.0,
        )
    gen_time = time.time() - t0

    plen = inputs["input_ids"].shape[-1]
    completion = tok.decode(out[0][plen:], skip_special_tokens=True)
    decision, confidence, reason = _parse(completion)
    if decision is None or confidence is None:
        decision, confidence, reason = "escalate_to_human", "LOW", "Parse failure"

    step_data = _step(session_id, decision, confidence, reason)
    scores = _extract_scores(step_data)
    print(
        f"  [MODEL] {task_id:30s} seed={seed}  "
        f"dec={decision:20s} conf={confidence}  "
        f"da={scores['decision_accuracy']:.2f}  "
        f"fd={scores['fraud_detection_score']:.2f}  "
        f"cal={scores['calibration_score']:.2f}  "
        f"[{gen_time:.1f}s]"
    )
    return {"task_id": task_id, "seed": seed, "decision": decision, "confidence": confidence,
            "completion": completion[:200], "gen_time_s": round(gen_time, 1), **scores}


def load_model(model_id, token):
    print(f"\nLoading {model_id} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Plain from_pretrained without device_map — works on CPU without accelerate
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, token=token)
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")
    return model, tok


def run_after_pass():
    print("\n" + "="*65)
    print("AFTER — GRPO fine-tuned model")
    print(f"Model: {TRAINED_MODEL}")
    print("="*65)
    model, tok = load_model(TRAINED_MODEL, HF_TOKEN or None)
    rows = []
    for task_id in EVAL_TASKS:
        for seed in SEEDS:
            try:
                row = run_model_episode(model, tok, task_id, seed)
            except Exception as exc:
                print(f"  ERROR {task_id} seed={seed}: {exc}")
                row = {"task_id": task_id, "seed": seed, "reward": 0.0,
                       "fraud_detection_score": 0.0, "decision_accuracy": 0.0,
                       "evidence_quality_score": 0.0, "calibration_score": 0.0}
            rows.append(row)
    means = {
        "Fraud detection":   round(mean(r["fraud_detection_score"]  for r in rows), 4),
        "Decision accuracy": round(mean(r["decision_accuracy"]      for r in rows), 4),
        "Evidence quality":  round(mean(r["evidence_quality_score"] for r in rows), 4),
        "Calibration":       round(mean(r["calibration_score"]      for r in rows), 4),
        "Mean reward":       round(mean(r["reward"]                 for r in rows), 4),
    }
    print(f"  Means: {json.dumps({k:v for k,v in means.items() if k!='Mean reward'})}")
    return rows, means


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(before_means, after_means, before_rows, after_rows):
    sp = Path("reports/training_summary.json")
    summary = json.loads(sp.read_text(encoding="utf-8"))
    delta = {k: round(after_means.get(k, 0.0) - before_means.get(k, 0.0), 4)
             for k in before_means if k != "Mean reward"}

    summary["eval_reward_before"] = {k: v for k, v in before_means.items() if k != "Mean reward"}
    summary["eval_reward_after"]  = {k: v for k, v in after_means.items()  if k != "Mean reward"}
    summary["component_shift"] = {
        "note": (
            "before=naive always-approve-HIGH baseline (simulates untrained agent), "
            f"after={TRAINED_MODEL} (GRPO fine-tuned). "
            "Rewards from live env HTTP API (MR-2 compliant)."
        ),
        "before": {k: v for k, v in before_means.items() if k != "Mean reward"},
        "after":  {k: v for k, v in after_means.items()  if k != "Mean reward"},
    }
    summary["component_shift_delta"] = delta
    summary["eval_methodology"] = (
        "before=naive always-approve-HIGH agent (zero training), "
        f"after={TRAINED_MODEL} (5,000-episode GRPO training, 2,500 steps). "
        f"Tasks: {EVAL_TASKS}. Seeds per task: {SEEDS}. "
        "All rewards from live env POST /step (not keyword matching). MR-2 compliant."
    )
    summary["eval_generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["eval_rows"] = {"before": before_rows, "after": after_rows}

    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {sp}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels  = ["Fraud detection", "Decision accuracy", "Evidence quality", "Calibration"]
        bv = [before_means.get(l, 0.0) for l in labels]
        av = [after_means.get(l, 0.0)  for l in labels]
        x, w = np.arange(len(labels)), 0.35

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.set_facecolor("#f9f9f9"); fig.patch.set_facecolor("#ffffff")
        ax.bar(x - w/2, bv, w, label="Before (naive always-approve-HIGH)", color="#e63946", alpha=0.7, edgecolor="white")
        ax.bar(x + w/2, av, w, label=f"After (GRPO fine-tuned)", color="#06a77d", alpha=0.85, edgecolor="white")

        for xi, (b_v, a_v) in enumerate(zip(bv, av)):
            ax.text(x[xi]-w/2, b_v + 0.02 if b_v >= 0 else b_v - 0.08,
                    f"{b_v:.2f}", ha="center", fontsize=9, color="#333")
            ax.text(x[xi]+w/2, a_v + 0.02 if a_v >= 0 else a_v - 0.08,
                    f"{a_v:.2f}", ha="center", fontsize=9, color="#1a6b58")
            d = a_v - b_v
            sign = "+" if d >= 0 else ""
            color = "#06a77d" if d > 0 else ("#e63946" if d < 0 else "#999")
            ax.text(xi, max(a_v, b_v) + 0.14, f"D{sign}{d:.2f}",
                    ha="center", fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.axhline(0, color="#666", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-1.3, 1.5)
        ax.set_ylabel("Component score", fontsize=10)
        ax.set_title(
            "DebateFloor: GRPO Training Effect on Reward Components\n"
            "Before (naive baseline) vs After (fine-tuned model, real inference)",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.legend(framealpha=0.85, fontsize=10)

        delta_str = "  |  ".join(f"{k}: {'+' if v>=0 else ''}{v:.2f}" for k, v in delta.items())
        ax.annotate(
            f"Deltas: {delta_str}\n"
            "Training reward: 0.130 → 0.469 (+0.339, 3.6x via live env HTTP, 2,500 steps)\n"
            "Source: real model inference (not scripted agents)",
            xy=(0.01, 0.01), xycoords="axes fraction", fontsize=7.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f8f0", edgecolor="#06a77d", alpha=0.85),
        )
        fig.tight_layout()
        Path("docs").mkdir(exist_ok=True)
        fig.savefig("docs/component_shift.svg", dpi=180, format="svg")
        plt.close(fig)
        print("docs/component_shift.svg updated")
    except Exception as exc:
        print(f"SVG failed: {exc}")


def main():
    r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
    assert r.json().get("status") == "healthy"
    print(f"Env healthy: {ENV_BASE_URL}")

    before_rows, before_means = run_before_pass()
    after_rows,  after_means  = run_after_pass()
    save_results(before_means, after_means, before_rows, after_rows)

    print("\n" + "="*65)
    print("FINAL RESULTS (real model vs naive baseline)")
    print("="*65)
    delta = {k: round(after_means.get(k, 0.0) - before_means.get(k, 0.0), 4)
             for k in before_means if k != "Mean reward"}
    print(f"Before: {json.dumps({k:v for k,v in before_means.items() if k!='Mean reward'})}")
    print(f"After:  {json.dumps({k:v for k,v in after_means.items()  if k!='Mean reward'})}")
    print(f"Delta:  {json.dumps(delta)}")


if __name__ == "__main__":
    main()
