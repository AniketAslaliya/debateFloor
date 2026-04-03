from __future__ import annotations

from threading import Lock
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from .environment import InsuranceClaimEnvironment
from .models import InsuranceClaimAction, InsuranceClaimObservation
from .tasks import list_tasks_summary


class ResetBody(BaseModel):
    task_id: str | None = None
    seed: int | None = None
    episode_id: str | None = None


class StepBody(BaseModel):
    action: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Insurance Claim Triage and Fraud Detection Environment")
_env = InsuranceClaimEnvironment()
_lock = Lock()


@app.get("/")
def index() -> dict:
    return {
        "name": "insurance_claim_triage_fraud_env",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/reset")
def reset(body: ResetBody = ResetBody()) -> dict:
    with _lock:
        obs = _env.reset(task_id=body.task_id, seed=body.seed, episode_id=body.episode_id)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
    }


@app.post("/step")
def step(body: StepBody) -> dict:
    try:
        action = InsuranceClaimAction(**body.action)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    with _lock:
        obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(obs.reward or 0.0),
        "done": bool(obs.done),
    }


@app.get("/state")
def state() -> dict:
    with _lock:
        return _env.state.model_dump()


@app.get("/schema")
def schema() -> dict:
    return {
        "action": InsuranceClaimAction.model_json_schema(),
        "observation": InsuranceClaimObservation.model_json_schema(),
        "state": _env.state.model_json_schema(),
    }


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": list_tasks_summary()}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "environment": "insurance_claim_triage_fraud_env"}
