"""
gradio_app.py — DebateFloor Visual Demo

Interactive Gradio interface that wraps the FastAPI environment.
Shows the 3x2 calibration matrix live, reward breakdown, and action history.

Launched alongside uvicorn in the Docker container (see app/main.py + Dockerfile).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import gradio as gr
import requests

# ── Connect to the FastAPI env (same process via internal port) ──
import os
API_BASE = os.getenv("DEBATEFLOOR_API_BASE", "http://localhost:7860")

# Calibration matrix values for display
CALIB_MATRIX = {
    ("HIGH", "correct"): (1.0,  "#22c55e"),   # green
    ("HIGH", "wrong"):   (-0.8, "#ef4444"),   # red
    ("MED",  "correct"): (0.6,  "#86efac"),   # light green
    ("MED",  "wrong"):   (-0.2, "#fca5a5"),   # light red
    ("LOW",  "correct"): (0.1,  "#d1fae5"),   # pale green
    ("LOW",  "wrong"):   (0.0,  "#f3f4f6"),   # grey
}

TASK_DESCRIPTIONS = {
    "clean_claim": "Easy | Auto collision, all docs consistent | Correct: approve + HIGH",
    "contradictory_claim": "Medium | Medical claim, procedure mismatch in billing | Correct: deny + MED",
    "distribution_shift_claim": "Hard | Looks clean — fraud only visible cross-claim | Correct: escalate + LOW",
}

TASK_STRATEGIES = {
    "clean_claim": [
        {"action_type": "validate_document", "parameters": {"doc_id": "DOC-1"}, "reasoning": "Verify primary claim document.", "confidence": None},
        {"action_type": "validate_document", "parameters": {"doc_id": "DOC-2"}, "reasoning": "Verify garage estimate.", "confidence": None},
        {"action_type": "estimate_payout", "parameters": {"amount_inr": 150000}, "reasoning": "Standard auto claim payout.", "confidence": None},
        {"action_type": "approve_claim", "parameters": {"reason": "All documents consistent."}, "reasoning": "Clean claim — HIGH confidence.", "confidence": "HIGH"},
    ],
    "contradictory_claim": [
        {"action_type": "validate_document", "parameters": {"doc_id": "DOC-10"}, "reasoning": "Check claim form date.", "confidence": None},
        {"action_type": "validate_document", "parameters": {"doc_id": "DOC-11"}, "reasoning": "Check hospital admission.", "confidence": None},
        {"action_type": "validate_document", "parameters": {"doc_id": "DOC-12"}, "reasoning": "Check billing summary for inflation.", "confidence": None},
        {"action_type": "query_historical_data", "parameters": {}, "reasoning": "Check prior claim history.", "confidence": None},
        {"action_type": "flag_fraud_signal", "parameters": {"flag_id": "procedure_mismatch", "evidence": "Discharge names appendectomy, bill charges cardiac bypass."}, "reasoning": "Document contradiction detected.", "confidence": None},
        {"action_type": "deny_claim", "parameters": {"reason": "Procedure mismatch confirmed."}, "reasoning": "Evidence sufficient — MED confidence due to document complexity.", "confidence": "MED"},
    ],
    "distribution_shift_claim": [
        {"action_type": "validate_document", "parameters": {"doc_id": "DOC-41"}, "reasoning": "Initial document check.", "confidence": None},
        {"action_type": "query_historical_data", "parameters": {}, "reasoning": "Must check cross-claim patterns.", "confidence": None},
        {"action_type": "query_linked_claim", "parameters": {"claim_id": "CLM-DIST-602"}, "reasoning": "Investigate linked claim for ring pattern.", "confidence": None},
        {"action_type": "query_linked_claim", "parameters": {"claim_id": "CLM-DIST-603"}, "reasoning": "Second linked claim — same broker.", "confidence": None},
        {"action_type": "flag_fraud_signal", "parameters": {"flag_id": "clustered_policy_broker", "evidence": "3 claimants share broker BRK-882 and same repair shop."}, "reasoning": "Coordinated ring detected.", "confidence": None},
        {"action_type": "escalate_to_human", "parameters": {"reason": "Cross-claim fraud ring — expert review required."}, "reasoning": "Full ring scope unclear — LOW confidence correct.", "confidence": "LOW"},
    ],
}


def _api_reset(task_id: str, seed: int = 42) -> Dict:
    r = requests.post(f"{API_BASE}/reset", json={"task_id": task_id, "seed": seed}, timeout=15)
    r.raise_for_status()
    return r.json()


def _api_step(session_id: str, action: Dict) -> Dict:
    r = requests.post(f"{API_BASE}/step", json={"action": action, "session_id": session_id}, timeout=15)
    r.raise_for_status()
    return r.json()


def _matrix_html(highlight_conf: Optional[str] = None, highlight_outcome: Optional[str] = None) -> str:
    rows = [("HIGH", "correct", "wrong"), ("MED", "correct", "wrong"), ("LOW", "correct", "wrong")]
    html = """
    <style>
      .matrix-table { border-collapse: collapse; width: 100%; font-family: monospace; font-size: 14px; }
      .matrix-table th { background: #1e1b4b; color: white; padding: 10px 16px; text-align: center; }
      .matrix-table td { padding: 12px 16px; text-align: center; border: 1px solid #e5e7eb; font-weight: bold; }
      .matrix-cell-active { outline: 3px solid #7c3aed; outline-offset: -3px; transform: scale(1.05); }
      .matrix-label { background: #f3f4f6; font-weight: bold; color: #374151; }
    </style>
    <table class="matrix-table">
      <tr>
        <th>Confidence</th><th>✅ Correct Decision</th><th>❌ Wrong Decision</th>
      </tr>
    """
    for conf in ["HIGH", "MED", "LOW"]:
        html += "<tr>"
        html += f'<td class="matrix-label">{conf}</td>'
        for outcome in ["correct", "wrong"]:
            val, color = CALIB_MATRIX[(conf, outcome)]
            is_active = (conf == highlight_conf and outcome == highlight_outcome)
            active_class = "matrix-cell-active" if is_active else ""
            sign = "+" if val > 0 else ""
            html += f'<td style="background:{color};" class="{active_class}">{sign}{val}</td>'
        html += "</tr>"
    html += "</table>"
    return html


def _format_action_log(history: List[Dict]) -> str:
    if not history:
        return "*No actions yet.*"
    lines = []
    for i, entry in enumerate(history, 1):
        action = entry.get("action_type", "?")
        reward = entry.get("reward", 0)
        conf = entry.get("confidence", "")
        conf_str = f" | confidence={conf}" if conf else ""
        calib = entry.get("calibration_score")
        calib_str = f" | **calibration={calib}**" if calib is not None else ""
        lines.append(f"**Step {i}:** `{action}`{conf_str} → reward={reward:.3f}{calib_str}")
    return "\n\n".join(lines)


def run_demo(task_id: str, progress=gr.Progress()):
    """Run the full scripted episode, yielding state updates at each step."""
    actions = TASK_STRATEGIES[task_id]
    history = []

    # Reset
    progress(0, desc="Resetting environment...")
    try:
        reset_resp = _api_reset(task_id)
    except Exception as e:
        yield (
            f"❌ Could not connect to environment: {e}",
            _matrix_html(),
            "*Reset failed.*",
            "—", "—", "—", "—",
        )
        return

    session_id = reset_resp["session_id"]
    obs = reset_resp["observation"]
    claim_text = _format_claim(obs)

    yield (
        claim_text,
        _matrix_html(),
        "*Episode started — running actions...*",
        "—", "—", "—", "In progress",
    )
    time.sleep(0.4)

    final_reward = 0.0
    final_calib = None
    final_conf = None
    final_outcome = None

    for i, action in enumerate(actions):
        progress((i + 1) / len(actions), desc=f"Step {i+1}: {action['action_type']}")

        # Build clean action dict (remove None confidence)
        action_payload = {k: v for k, v in action.items() if v is not None}

        try:
            step_resp = _api_step(session_id, action_payload)
        except Exception as e:
            history.append({**action, "reward": 0, "error": str(e)})
            yield (
                claim_text,
                _matrix_html(),
                _format_action_log(history),
                "—", "—", "—", f"Error: {e}",
            )
            continue

        reward = step_resp.get("reward", 0.0)
        done = step_resp.get("done", False)
        step_obs = step_resp.get("observation", {})
        rb = step_obs.get("reward_breakdown", {})
        calib = rb.get("calibration_score")
        conf = action.get("confidence")

        entry = {
            "action_type": action["action_type"],
            "reward": reward,
            "confidence": conf,
            "calibration_score": calib,
        }
        history.append(entry)
        final_reward = reward

        # Determine matrix highlight
        matrix_conf = conf
        matrix_outcome = None
        if conf and calib is not None:
            final_calib = calib
            final_conf = conf
            matrix_outcome = "correct" if calib >= 0 else "wrong"
            final_outcome = matrix_outcome

        yield (
            claim_text,
            _matrix_html(matrix_conf, matrix_outcome),
            _format_action_log(history),
            f"{reward:.3f}",
            f"{calib if calib is not None else '—'}",
            conf or "—",
            "Done ✅" if done else "In progress",
        )
        time.sleep(0.5)

    # Final summary
    outcome_emoji = "✅ CORRECT" if final_outcome == "correct" else "❌ WRONG" if final_outcome == "wrong" else "—"
    yield (
        claim_text,
        _matrix_html(final_conf, final_outcome),
        _format_action_log(history),
        f"{final_reward:.3f}",
        f"{final_calib if final_calib is not None else '—'}",
        final_conf or "—",
        outcome_emoji,
    )


def _format_claim(obs: Dict) -> str:
    claimant = obs.get("claimant", {})
    incident = obs.get("incident", {})
    docs = obs.get("documents", [])
    linked = obs.get("linked_claims", [])

    lines = [
        f"**Claim ID:** `{obs.get('claim_id', '?')}`  |  **Task:** `{obs.get('task_id', '?')}`",
        f"**Max Steps:** {obs.get('max_steps', '?')}  |  **Budget:** {obs.get('investigation_budget', '?')} units",
        "",
        f"**Claimant:** {claimant.get('name', '?')} — Policy `{claimant.get('policy_number', '?')}`",
        f"**Incident:** {incident.get('type', '?')} on {incident.get('date', '?')} at {incident.get('location', '?')}",
        f"**Description:** {incident.get('description', '?')}",
        "",
        f"**Documents ({len(docs)}):**",
    ]
    for d in docs:
        lines.append(f"  - `{d['doc_id']}` [{d['doc_type']}]: {d['content'][:80]}...")
    if linked:
        lines.append(f"\n**Linked Claims:** {len(linked)} related claims flagged")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────

HEADER = """
# DebateFloor ⚖️ — Insurance Calibration RL Environment
### *An agent must decide AND declare how confident it is. Wrong + overconfident = worst outcome.*

Based on [CoCA arXiv:2603.05881](https://arxiv.org/abs/2603.05881) · Built for Meta PyTorch × Scaler Hackathon 2026
"""

with gr.Blocks(title="DebateFloor") as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=list(TASK_STRATEGIES.keys()),
                value="contradictory_claim",
                label="Select Task",
                info="Easy → Medium → Hard (confidence requirements differ)",
            )
            task_desc = gr.Markdown(TASK_DESCRIPTIONS["contradictory_claim"])
            run_btn = gr.Button("▶ Run Episode", variant="primary", size="lg")

            gr.Markdown("### Live Metrics")
            with gr.Row():
                reward_box = gr.Textbox(label="Reward", value="—", interactive=False)
                calib_box  = gr.Textbox(label="Calibration Score", value="—", interactive=False)
            with gr.Row():
                conf_box   = gr.Textbox(label="Confidence Declared", value="—", interactive=False)
                outcome_box = gr.Textbox(label="Outcome", value="—", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 3×2 Calibration Matrix")
            gr.Markdown("*The highlighted cell shows the agent's confidence × outcome. HIGH+wrong = −0.8, the worst penalty.*")
            matrix_html = gr.HTML(_matrix_html())

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Claim Details")
            claim_display = gr.Markdown("*Select a task and click Run Episode.*")
        with gr.Column(scale=1):
            gr.Markdown("### Action Log")
            action_log = gr.Markdown("*Actions will appear here as the episode runs.*")

    # Update task description on dropdown change
    def update_desc(task):
        return TASK_DESCRIPTIONS.get(task, "")

    task_dropdown.change(update_desc, inputs=task_dropdown, outputs=task_desc)

    # Run episode
    run_btn.click(
        fn=run_demo,
        inputs=[task_dropdown],
        outputs=[claim_display, matrix_html, action_log, reward_box, calib_box, conf_box, outcome_box],
    )

    gr.Markdown("""
---
### API (for GRPO training)
```bash
POST /reset  {"task_id": "contradictory_claim", "seed": 42}
POST /step   {"action": {"action_type": "deny_claim", "confidence": "MED", ...}, "session_id": "..."}
GET  /health  →  {"status": "healthy"}
```
Supports 64 concurrent sessions · [GitHub](https://github.com/AniketAslaliya/debateFloor) · [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
    """)


if __name__ == "__main__":
    demo.launch(server_port=7861, share=False)
