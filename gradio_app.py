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
        ("HIGH", "correct"): (1.0,  "#dcfce7"),   # soft green
        ("HIGH", "wrong"):   (-0.8, "#fee2e2"),   # soft red
        ("MED",  "correct"): (0.6,  "#f0fdf4"),   # very light green
        ("MED",  "wrong"):   (-0.2, "#ffedd5"),   # light amber
        ("LOW",  "correct"): (0.1,  "#eff6ff"),   # light blue
        ("LOW",  "wrong"):   (0.0,  "#f8fafc"),   # neutral
}

TASK_DESCRIPTIONS = {
        "clean_claim": "Easy: all documents agree. Correct action: approve_claim with HIGH confidence.",
        "contradictory_claim": "Medium: documents disagree. Correct action: deny_claim with MED confidence.",
        "distribution_shift_claim": "Hard: looks normal at first. Correct action: escalate_to_human with LOW confidence.",
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
        {"action_type": "flag_fraud_signal", "parameters": {"flag_id": "date_mismatch", "evidence": "Claim form date differs from hospital admission date."}, "reasoning": "Date inconsistency flagged.", "confidence": None},
        {"action_type": "flag_fraud_signal", "parameters": {"flag_id": "cost_inflation", "evidence": "Billing is 2.4x the standard rate for this procedure."}, "reasoning": "Cost inflation detected.", "confidence": None},
        {"action_type": "convene_debate_panel", "parameters": {}, "reasoning": "Seek adversarial perspectives before final decision.", "confidence": None},
        {"action_type": "deny_claim", "parameters": {"reason": "Procedure mismatch and cost inflation confirmed by debate panel."}, "reasoning": "Panel leans prosecution — MED confidence appropriate.", "confidence": "MED"},
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
            .matrix-table { border-collapse: collapse; width: 100%; font-family: Inter, ui-sans-serif, system-ui, sans-serif; font-size: 14px; background: white; }
            .matrix-table th { background: #f8fafc; color: #0f172a; padding: 10px 16px; text-align: center; border: 1px solid #e2e8f0; }
            .matrix-table td { padding: 12px 16px; text-align: center; border: 1px solid #e2e8f0; font-weight: 600; color: #0f172a; }
            .matrix-cell-active { outline: 3px solid #2563eb; outline-offset: -3px; transform: scale(1.03); }
            .matrix-label { background: #f8fafc; font-weight: 700; color: #334155; }
            .matrix-note { margin-top: 8px; color: #475569; font-size: 13px; }
    </style>
    <table class="matrix-table">
      <tr>
        <th>Confidence</th><th>✅ Correct Decision</th><th>❌ Wrong Decision</th>
      </tr>
    <div class="matrix-note">Higher confidence is only good when the decision is right. The worst cell is HIGH + wrong.</div>
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
        return "No actions yet. Click Run Episode to see the investigation unfold."
    lines = []
    for i, entry in enumerate(history, 1):
        action = entry.get("action_type", "?")
        reward = entry.get("reward", 0)
        conf = entry.get("confidence", "")
        conf_str = f" | confidence={conf}" if conf else ""
        calib = entry.get("calibration_score")
        calib_str = f" | calibration={calib}" if calib is not None else ""
        lines.append(f"Step {i}: {action}{conf_str} -> reward={reward:.3f}{calib_str}")
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
            _matrix_html(), "*Reset failed.*",
            "—", "—", "—", "—", "",
        )
        return

    session_id = reset_resp["session_id"]
    obs = reset_resp["observation"]
    claim_text = _format_claim(obs)

    yield (
        claim_text, _matrix_html(),
        "*Episode started — running actions...*",
        "—", "—", "—", "In progress", "",
    )
    time.sleep(0.4)

    final_reward = 0.0
    final_calib = None
    final_conf = None
    final_outcome = None
    debate_html = ""

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
                debate_html,
            )
            continue

        reward = step_resp.get("reward", 0.0)
        done = step_resp.get("done", False)
        step_obs = step_resp.get("observation", {})
        rb = step_obs.get("reward_breakdown", {})
        calib = rb.get("calibration_score")
        conf = action.get("confidence")

        # Capture debate transcript when panel is convened
        debate = step_obs.get("debate_transcript")
        if debate:
            debate_html = _debate_html(debate)

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
            debate_html,
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
        debate_html,
    )


def _debate_html(transcript: Dict) -> str:
    if not transcript:
        return ""
    p_strength = transcript.get("prosecutor_strength", "?")
    d_strength = transcript.get("defender_strength", "?")
    lean = transcript.get("panel_lean", "split")
    lean_color = {"prosecution": "#ef4444", "defense": "#22c55e", "split": "#f59e0b"}.get(lean, "#6b7280")

    return f"""
        <div style="font-family:Inter, ui-sans-serif, system-ui, sans-serif;font-size:13px;border:1px solid #cbd5e1;border-radius:12px;padding:16px;margin-top:8px;background:#ffffff;box-shadow:0 8px 20px rgba(15,23,42,0.06);">
            <div style="font-weight:700;font-size:15px;color:{lean_color};margin-bottom:12px;">
                Debate panel opened at step {transcript.get('step_convened','?')}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:#fff7ed;padding:12px;border-radius:10px;border-left:4px solid #f97316;">
                    <div style="font-weight:700;color:#c2410c;margin-bottom:6px;">Prosecutor [{p_strength}]</div>
                    <div style="color:#334155;line-height:1.55;">{transcript.get('prosecutor_argument','')}</div>
        </div>
                <div style="background:#eff6ff;padding:12px;border-radius:10px;border-left:4px solid #2563eb;">
                    <div style="font-weight:700;color:#1d4ed8;margin-bottom:6px;">Defender [{d_strength}]</div>
                    <div style="color:#334155;line-height:1.55;">{transcript.get('defender_argument','')}</div>
        </div>
      </div>
            <div style="margin-top:12px;background:#f8fafc;padding:10px;border-radius:10px;font-weight:700;color:{lean_color};border:1px solid #e2e8f0;">
        VERDICT: {transcript.get('panel_verdict','')}
      </div>
    </div>
    """


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
## DebateFloor ⚖️
### Insurance claims, calibrated confidence, and multi-agent debate in one environment.

An agent must decide and declare confidence before every terminal action. The worst case is being wrong and overconfident.
Based on [CoCA arXiv:2603.05881](https://arxiv.org/abs/2603.05881) · Built for Meta PyTorch × Scaler Hackathon 2026
"""

INTRO = """
### How to use this demo
1. Pick a task from the dropdown.
2. Read the claim and the short task summary.
3. Click **Run Episode**.
4. Watch the action log, matrix, and debate panel update together.
"""

with gr.Blocks(title="DebateFloor", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)
    gr.Markdown(INTRO)
    gr.HTML(
        """
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:14px 16px;margin-bottom:14px;">
          <strong>What to look for:</strong> the highlighted matrix cell shows the confidence/outcome pair, the action log shows the investigation sequence, and the debate panel explains why the final decision changed.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            task_dropdown = gr.Dropdown(
                choices=list(TASK_STRATEGIES.keys()),
                value="contradictory_claim",
                label="Select Task",
                info="Start with contradictory_claim to see the full flow. distribution_shift_claim shows the hardest case.",
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
            gr.Markdown("*The highlighted cell shows the agent's confidence × outcome. HIGH + wrong = −0.8, the worst penalty.*")
            matrix_html = gr.HTML(_matrix_html())

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Claim Details")
            claim_display = gr.Markdown("Select a task and click Run Episode.")
        with gr.Column(scale=1):
            gr.Markdown("### Action Log")
            action_log = gr.Markdown("Actions will appear here as the episode runs.")

    gr.Markdown("### Multi-Agent Debate Panel")
    gr.Markdown("*Appears when the agent calls `convene_debate_panel` — prosecutor vs defender arguments, then the judge decides.*")
    debate_panel = gr.HTML("")

    # Update task description on dropdown change
    def update_desc(task):
        return TASK_DESCRIPTIONS.get(task, "")

    task_dropdown.change(update_desc, inputs=task_dropdown, outputs=task_desc)

    # Run episode
    run_btn.click(
        fn=run_demo,
        inputs=[task_dropdown],
        outputs=[claim_display, matrix_html, action_log, reward_box, calib_box, conf_box, outcome_box, debate_panel],
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
