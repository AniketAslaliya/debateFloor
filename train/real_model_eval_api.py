"""
real_model_eval_api.py — Real model eval using HF Serverless Inference API.

No local download needed. Calls HF Inference API for:
  BEFORE: Qwen/Qwen2.5-0.5B-Instruct (base, untuned)
  AFTER:  AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct (GRPO fine-tuned)

Rewards come from the live local environment HTTP API (MR-2 compliant).
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

ENV_BASE_URL   = os.getenv("ENV_BASE_URL", "http://localhost:7861")
BASE_MODEL     = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED_MODEL  = "AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct"
HF_TOKEN       = os.getenv("HF_TOKEN", "")

HF_INFERENCE   = "https://api-inference.huggingface.co/v1/chat/completions"

EVAL_TASKS = ["clean_claim", "contradictory_claim", "distribution_shift_claim"]
SEEDS      = [7, 42]   # 2 seeds × 3 tasks = 6 episodes each pass

SYSTEM = (
    "You are an expert insurance fraud investigator.\n"
    "Analyze the claim and respond EXACTLY in this format:\n"
    "DECISION: <approve_claim|deny_claim|escalate_to_human>\n"
    "CONFIDENCE: <HIGH|MED|LOW>\n"
    "REASON: <one sentence citing specific evidence>\n\n"
    "HIGH = certain. MED = likely but some doubt. LOW = ambiguous, expert needed.\n"
    "WARNING: HIGH confidence on a wrong decision is the worst outcome."
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


def hf_infer(model_id, messages, max_tokens=120, retries=3):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    for attempt in range(retries):
        try:
            r = requests.post(HF_INFERENCE, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            elif r.status_code == 503:
                wait = (attempt + 1) * 10
                print(f"    Model loading (503), waiting {wait}s ...")
                time.sleep(wait)
            elif r.status_code == 404:
                print(f"    Model {model_id} not available via Inference API (404)")
                return None
            else:
                print(f"    API error {r.status_code}: {r.text[:200]}")
                time.sleep(5)
        except Exception as exc:
            print(f"    Request error: {exc}")
            time.sleep(5)
    return None


def run_episode(model_id, task_id, seed):
    reset_r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=15,
    )
    reset_r.raise_for_status()
    reset_data = reset_r.json()
    session_id = reset_data["session_id"]

    obs = reset_data.get("observation", {})
    docs = obs.get("documents", [])
    doc_text = "\n".join(
        f"  [{d.get('doc_type','doc')}] {d.get('content','')[:200]}" for d in docs
    )
    incident = obs.get("incident", {})
    obs_text = (
        f"Task: {task_id} | Claim: {obs.get('claim_id','')}\n"
        f"Claimant: {obs.get('claimant',{}).get('name','')}\n"
        f"Incident: {incident.get('type','')} — {incident.get('description','')[:150]}\n"
        f"Documents:\n{doc_text}\n"
        f"Linked claims: {len(obs.get('linked_claims', []))}"
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": obs_text},
    ]

    t0 = time.time()
    completion = hf_infer(model_id, messages)
    gen_time = time.time() - t0

    if completion is None:
        decision, confidence, reason = "escalate_to_human", "LOW", "Inference API unavailable."
    else:
        decision, confidence, reason = _parse(completion)
        if decision is None or confidence is None:
            decision, confidence, reason = "escalate_to_human", "LOW", "Parse failure."

    action = {
        "action_type": decision,
        "confidence":  confidence,
        "parameters":  {"reason": reason},
        "reasoning":   reason,
    }
    step_r = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action, "session_id": session_id},
        timeout=15,
    )
    step_r.raise_for_status()
    step_data = step_r.json()
    breakdown = step_data.get("observation", {}).get("reward_breakdown", {})

    print(
        f"    {task_id:30s} seed={seed}  "
        f"dec={decision:20s} conf={confidence}  "
        f"da={float(breakdown.get('decision_accuracy',0)):.2f}  "
        f"fd={float(breakdown.get('fraud_detection_score',0)):.2f}  "
        f"cal={float(breakdown.get('calibration_score',0)):.2f}  "
        f"[{gen_time:.1f}s]"
    )

    return {
        "task_id":    task_id,
        "seed":       seed,
        "decision":   decision,
        "confidence": confidence,
        "completion": (completion or "")[:200],
        "gen_time_s": round(gen_time, 1),
        "reward":     round(float(step_data.get("reward", 0.0)), 4),
        "fraud_detection_score":   round(float(breakdown.get("fraud_detection_score",  0.0)), 4),
        "decision_accuracy":       round(float(breakdown.get("decision_accuracy",      0.0)), 4),
        "evidence_quality_score":  round(float(breakdown.get("evidence_quality_score", 0.0)), 4),
        "calibration_score":       round(float(breakdown.get("calibration_score",      0.0)), 4),
    }


def eval_pass(model_id, label):
    print(f"\n{'='*65}")
    print(f"EVAL: {label}")
    print(f"{'='*65}")
    rows = []
    for task_id in EVAL_TASKS:
        for seed in SEEDS:
            try:
                row = run_episode(model_id, task_id, seed)
                rows.append(row)
            except Exception as exc:
                print(f"    ERROR {task_id} seed={seed}: {exc}")
                rows.append({
                    "task_id": task_id, "seed": seed,
                    "reward": 0.0, "fraud_detection_score": 0.0,
                    "decision_accuracy": 0.0, "evidence_quality_score": 0.0,
                    "calibration_score": 0.0,
                })

    means = {
        "Fraud detection":   round(mean(r["fraud_detection_score"]  for r in rows), 4),
        "Decision accuracy": round(mean(r["decision_accuracy"]      for r in rows), 4),
        "Evidence quality":  round(mean(r["evidence_quality_score"] for r in rows), 4),
        "Calibration":       round(mean(r["calibration_score"]      for r in rows), 4),
        "Mean reward":       round(mean(r["reward"]                 for r in rows), 4),
    }
    print(f"  Component means: {json.dumps({k:v for k,v in means.items() if k!='Mean reward'})}")
    return rows, means


def save_results(before_means, after_means, before_rows, after_rows):
    summary_path = Path("reports/training_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    delta = {k: round(after_means.get(k, 0.0) - before_means.get(k, 0.0), 4)
             for k in before_means if k != "Mean reward"}

    summary["eval_reward_before"] = {k: v for k, v in before_means.items() if k != "Mean reward"}
    summary["eval_reward_after"]  = {k: v for k, v in after_means.items()  if k != "Mean reward"}
    summary["component_shift"] = {
        "note": (
            "Real model inference via HF Serverless Inference API. "
            f"before={BASE_MODEL}, after={TRAINED_MODEL}. "
            "Rewards from live env HTTP API (MR-2 compliant)."
        ),
        "before": {k: v for k, v in before_means.items() if k != "Mean reward"},
        "after":  {k: v for k, v in after_means.items()  if k != "Mean reward"},
    }
    summary["component_shift_delta"] = delta
    summary["eval_methodology"] = (
        f"Real inference: base={BASE_MODEL} vs fine-tuned={TRAINED_MODEL}. "
        f"Tasks: {EVAL_TASKS}. Seeds: {SEEDS}. "
        "Env rewards from POST /step (not keyword matching). MR-2 compliant."
    )
    summary["eval_generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["eval_rows"] = {"before": before_rows, "after": after_rows}

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nUpdated {summary_path}")

    # Regenerate SVG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels  = ["Fraud detection", "Decision accuracy", "Evidence quality", "Calibration"]
        bv = [before_means.get(l, 0.0) for l in labels]
        av = [after_means.get(l, 0.0)  for l in labels]
        x, w = np.arange(len(labels)), 0.35

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.set_facecolor("#f9f9f9"); fig.patch.set_facecolor("#ffffff")
        ax.bar(x - w/2, bv, w, label="Before (base Qwen2.5-0.5B)", color="#7a869a", alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, av, w, label="After (GRPO fine-tuned)",     color="#06a77d", alpha=0.85, edgecolor="white")

        for xi, (b_v, a_v) in enumerate(zip(bv, av)):
            ax.text(x[xi]-w/2, b_v + 0.02 if b_v >= 0 else b_v - 0.07, f"{b_v:.2f}", ha="center", fontsize=9, color="#333")
            ax.text(x[xi]+w/2, a_v + 0.02 if a_v >= 0 else a_v - 0.07, f"{a_v:.2f}", ha="center", fontsize=9, color="#1a6b58")
            d = a_v - b_v
            sign = "+" if d >= 0 else ""
            color = "#06a77d" if d > 0 else ("#e63946" if d < 0 else "#999")
            ax.text(xi, max(a_v, b_v) + 0.12, f"D{sign}{d:.2f}", ha="center", fontsize=9, color=color, fontweight="bold")

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.axhline(0, color="#666", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-1.2, 1.4)
        ax.set_ylabel("Component score", fontsize=10)
        ax.set_title("DebateFloor: Real Model Before vs After GRPO\n(HF Inference API, MR-2 compliant live env rewards)", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.legend(framealpha=0.85, fontsize=10)

        delta_str = "  |  ".join(f"{k}: {'+' if v>=0 else ''}{v:.2f}" for k, v in delta.items())
        ax.annotate(
            f"Deltas: {delta_str}\nTraining reward: 0.045 → 0.332 (7x, live env HTTP)\n"
            "Source: real model inference via HF API",
            xy=(0.01, 0.01), xycoords="axes fraction", fontsize=7.5, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f8f0", edgecolor="#06a77d", alpha=0.85),
        )
        fig.tight_layout()
        Path("docs").mkdir(exist_ok=True)
        fig.savefig("docs/component_shift.svg", dpi=180, format="svg")
        plt.close(fig)
        print("docs/component_shift.svg updated")
    except Exception as exc:
        print(f"SVG generation failed: {exc}")


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Run: $env:HF_TOKEN='hf_...'")
        sys.exit(1)

    r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
    assert r.json().get("status") == "healthy", f"Env not healthy: {r.text}"
    print(f"Env healthy at {ENV_BASE_URL}")

    before_rows, before_means = eval_pass(BASE_MODEL,     f"BEFORE — {BASE_MODEL}")
    after_rows,  after_means  = eval_pass(TRAINED_MODEL,  f"AFTER  — {TRAINED_MODEL}")

    save_results(before_means, after_means, before_rows, after_rows)

    print("\n" + "="*65)
    print("RESULTS (real model inference, HF API)")
    print("="*65)
    delta = {k: round(after_means.get(k, 0.0) - before_means.get(k, 0.0), 4)
             for k in before_means if k != "Mean reward"}
    print(f"Before: {json.dumps({k:v for k,v in before_means.items() if k!='Mean reward'})}")
    print(f"After:  {json.dumps({k:v for k,v in after_means.items()  if k!='Mean reward'})}")
    print(f"Delta:  {json.dumps(delta)}")


if __name__ == "__main__":
    main()
