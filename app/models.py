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
    fraud_detection_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of expected fraud signals found")
    decision_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="1.0 if final decision matches allowed decisions, else 0.0")
    payout_accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Score for payout estimate within the expected band")
    efficiency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Step efficiency: higher when fewer steps used")
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="For coordinated_fraud: quality of linked-claim targeting")
    evidence_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of flagged signals backed by keyword-grounded evidence")
    exploit_penalty: float = Field(default=0.0, ge=0.0, description="Penalty for looping or duplicate actions")
    penalty: float = Field(default=0.0, description="Total accumulated penalty subtracted from weighted score")
    total: float = Field(default=0.0, ge=0.0, le=1.0, description="Final clamped reward in [0.0, 1.0]")


class InsuranceClaimAction(Action):
    action_type: Literal[
        "validate_document",
        "request_information",
        "flag_fraud_signal",
        "estimate_payout",
        "approve_claim",
        "deny_claim",
        "request_investigation",
        "query_linked_claim",  # Required for coordinated_fraud: reveals full linked claim detail
    ] = Field(..., description="The type of action to perform on the claim")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters. See /schema for required fields per action_type.",
    )
    reasoning: str = Field(
        default="",
        max_length=4000,
        description="Agent's reasoning for this action. Used for evidence quality scoring.",
    )


class InsuranceClaimObservation(Observation):
    claim_id: str = Field(..., description="Unique identifier for this claim")
    task_id: str = Field(..., description="Task identifier: clean_claim | contradictory_claim | coordinated_fraud")
    claimant: Dict[str, Any] = Field(..., description="Claimant personal and policy details")
    incident: Dict[str, Any] = Field(..., description="Incident date, location, type, and description")
    documents: List[Dict[str, Any]] = Field(..., description="Claim documents available for validation")
    linked_claims: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="For coordinated_fraud: stub entries with claim_id and claimant only. Use query_linked_claim to retrieve full details.",
    )
    action_history: List[Dict[str, Any]] = Field(default_factory=list, description="Actions taken so far this episode")
    available_actions: List[str] = Field(default_factory=list, description="Valid action_type values for this task")
    step_number: int = Field(default=0, description="Current step number (0-indexed from reset)")
    max_steps: int = Field(default=0, description="Maximum steps allowed before episode closes")
    flags_raised: List[str] = Field(default_factory=list, description="Fraud signal flag IDs raised so far")
    status: ClaimStatus = Field(default=ClaimStatus.OPEN, description="Current claim processing status")
    message: str = Field(default="", description="Human-readable message describing result of last action")
    reward_breakdown: InsuranceClaimReward = Field(default_factory=InsuranceClaimReward, description="Detailed reward components for current step")


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
