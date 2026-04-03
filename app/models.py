from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    DECIDED = "decided"
    CLOSED = "closed"


class InsuranceClaimReward(BaseModel):
    fraud_detection_score: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    payout_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    efficiency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    penalty: float = Field(default=0.0)
    total: float = Field(default=0.0, ge=0.0, le=1.0)


class InsuranceClaimAction(Action):
    action_type: Literal[
        "validate_document",
        "request_information",
        "flag_fraud_signal",
        "estimate_payout",
        "approve_claim",
        "deny_claim",
        "request_investigation",
    ]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(default="", max_length=4000)


class InsuranceClaimObservation(Observation):
    claim_id: str
    task_id: str
    claimant: Dict[str, Any]
    incident: Dict[str, Any]
    documents: List[Dict[str, Any]]
    linked_claims: List[Dict[str, Any]] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    step_number: int = 0
    max_steps: int = 0
    flags_raised: List[str] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.OPEN
    message: str = ""
    reward_breakdown: InsuranceClaimReward = Field(default_factory=InsuranceClaimReward)


class InsuranceClaimState(State):
    task_id: str = ""
    claim_id: str = ""
    step_number: int = 0
    max_steps: int = 0
    status: ClaimStatus = ClaimStatus.OPEN
    flags_raised: List[str] = Field(default_factory=list)
    found_signals: List[str] = Field(default_factory=list)
    penalty_total: float = 0.0
    done: bool = False
    last_action_error: Optional[str] = None
    payout_estimate_inr: Optional[float] = None
    final_decision: Optional[str] = None
    final_score: float = 0.0
