from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .environment import InsuranceClaimEnvironment
from .models import InsuranceClaimAction, InsuranceClaimObservation
from .tasks import list_tasks_summary
from .session_store import get_confidence_distribution

SESSION_TTL_SECONDS = 1800  # 30 minutes


class SessionEntry:
    def __init__(self, env: InsuranceClaimEnvironment):
        self.env = env
        self.last_used = time.time()


_sessions: Dict[str, SessionEntry] = {}
_sessions_lock = Lock()


def _get_or_create_session(session_id: str) -> InsuranceClaimEnvironment:
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = SessionEntry(InsuranceClaimEnvironment())
        entry = _sessions[session_id]
        entry.last_used = time.time()
        return entry.env


def _cleanup_sessions() -> None:
    now = time.time()
    with _sessions_lock:
        expired = [k for k, v in _sessions.items() if now - v.last_used > SESSION_TTL_SECONDS]
        for k in expired:
            del _sessions[k]


class ResetBody(BaseModel):
    task_id: str | None = None
    seed: int | None = None
    session_id: str | None = None
    episode_id: str | None = None


class StepBody(BaseModel):
    action: Dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None


app = FastAPI(title="DebateFloor — Insurance Calibration RL Environment")

import os
_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
_react_mounted = False

if os.path.isdir(_frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="assets")
    _react_mounted = True
    print("React UI mounted from frontend/dist")
else:
    print(f"WARNING: React UI not mounted. Missing directory: {_frontend_dist}")

@app.get("/")
def index():
    if _react_mounted:
        return FileResponse(os.path.join(_frontend_dist, "index.html"))
    return {
        "name": "DebateFloor — Insurance Calibration RL Environment",
        "status": "running",
        "endpoints": ["/health", "/tasks", "/schema", "/reset", "/step", "/state"],
        "docs": "/docs",
    }


@app.post("/reset")
def reset(body: ResetBody = ResetBody(), background_tasks: BackgroundTasks = BackgroundTasks()) -> dict:
    background_tasks.add_task(_cleanup_sessions)
    session_id = body.session_id or body.episode_id or str(uuid4())
    env = _get_or_create_session(session_id)
    obs = env.reset(task_id=body.task_id, seed=body.seed, episode_id=session_id)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
        "session_id": session_id,
    }


@app.post("/step")
def step(body: StepBody) -> dict:
    session_id = body.session_id or "default"
    env = _get_or_create_session(session_id)
    try:
        action = InsuranceClaimAction(**body.action)
    except (ValidationError, ValueError) as exc:
        errors = exc.errors() if hasattr(exc, "errors") else [{"msg": str(exc)}]
        # Ensure errors are JSON-serialisable (strip non-serialisable ctx values)
        safe = [
            {k: str(v) if not isinstance(v, (str, int, float, bool, list)) else v
             for k, v in e.items() if k != "ctx"}
            for e in errors
        ]
        raise HTTPException(status_code=422, detail=safe)
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
        "session_id": session_id,
    }


@app.get("/state")
def state(session_id: str = Query(default="default")) -> dict:
    env = _get_or_create_session(session_id)
    return env.state.model_dump()


@app.get("/schema")
def schema() -> dict:
    env = InsuranceClaimEnvironment()
    return {
        "action": InsuranceClaimAction.model_json_schema(),
        "observation": InsuranceClaimObservation.model_json_schema(),
        "state": env.state.model_json_schema(),
    }


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": list_tasks_summary()}


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "environment": "debatefloor_insurance_calibration_env",
        "active_sessions": len(_sessions),
    }


@app.get("/stats")
def stats() -> dict:
    """Confidence distribution across all sessions — proves anti-gaming is active."""
    return get_confidence_distribution()


@app.post("/rollout")
def rollout(task_id: str = "contradictory_claim", seed: int = 42) -> dict:
    """Run a scripted demo episode and return the full step-by-step trace for judges."""
    import requests as _req
    session_id = f"rollout-{seed}-{task_id}"
    base = "http://localhost:7860"
    trace = []

    reset_r = _req.post(f"{base}/reset", json={"task_id": task_id, "seed": seed, "session_id": session_id})
    trace.append({"action": "reset", "response": reset_r.json()})

    scripted_steps = [
        {"action_type": "validate_document", "parameters": {"doc_id": "DOC-001"}, "reasoning": "Checking primary document for fraud signals."},
        {"action_type": "flag_fraud_signal", "parameters": {"flag_id": "date_mismatch", "evidence": "Incident date on claim form contradicts hospital admission date."}, "reasoning": "Date inconsistency is a strong fraud indicator."},
        {"action_type": "convene_debate_panel", "parameters": {}, "reasoning": "Evidence is contradictory — convening adversarial debate before terminal decision."},
        {"action_type": "deny_claim", "confidence": "MED", "reason": "Date mismatch confirmed by debate panel.", "reasoning": "MED confidence — debate panel supports denial but evidence is not conclusive."},
    ]

    for action in scripted_steps:
        step_r = _req.post(f"{base}/step", json={"action": action, "session_id": session_id})
        step_data = step_r.json()
        trace.append({"action": action["action_type"], "reward": step_data.get("reward"), "done": step_data.get("done"), "response": step_data})
        if step_data.get("done"):
            break

    return {"task_id": task_id, "seed": seed, "session_id": session_id, "trace": trace}
