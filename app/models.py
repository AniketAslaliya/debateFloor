from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Inline base classes — removes the openenv package dependency so this module
# works both inside the monorepo (with src/ on PYTHONPATH) and in the
# standalone HF Space Docker container.
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Base class for all environment actions."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)


class Observation(BaseModel):
    """Base class for all environment observations."""
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: Union[bool, int, float, None] = Field(
        default=None, description="Reward signal from the last action"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the observation"
    )


class State(BaseModel):
    """Base class for environment state."""
    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    episode_id: Optional[str] = Field(default=None, description="Unique identifier for the current episode")
    step_count: int = Field(default=0, ge=0, description="Number of steps taken in the current episode")


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

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
    calibration_score: Optional[float] = Field(default=None, description="3×2 matrix calibration score in [-1.0, 1.0]. Only populated on terminal actions.")
    exploit_penalty: float = Field(default=0.0, ge=0.0, description="Penalty for looping or duplicate actions")
    penalty: float = Field(default=0.0, description="Total accumulated penalty subtracted from weighted score")
    total: float = Field(default=0.0, ge=0.0, le=1.0, description="Final clamped reward in [0.0, 1.0]")


class InsuranceClaimAction(Action):
    action_type: Literal[
        "validate_document",
        "request_information",
        "lookup_policy_history",   # All tasks: reveals prior claim history
        "compare_documents",       # All tasks: cross-document tamper detection
        "flag_fraud_signal",
        "estimate_payout",
        "approve_claim",
        "deny_claim",
        "request_investigation",
        "query_linked_claim",      # coordinated_fraud only: reveals full linked claim detail
        "verify_identity",         # identity_fraud only: cross-checks claimant against registry
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
    confidence: Optional[Literal["HIGH", "MED", "LOW"]] = Field(
        default=None,
        description="Agent's declared confidence level. Required for terminal actions (approve_claim, deny_claim, escalate_to_human). Graded via 3×2 calibration matrix.",
    )

    def model_post_init(self, __context: Any) -> None:
        terminal_actions = {"approve_claim", "deny_claim", "escalate_to_human"}
        if self.action_type in terminal_actions and self.confidence is None:
            raise ValueError(
                f"confidence is required for terminal action '{self.action_type}'. Must be HIGH, MED, or LOW."
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
    investigation_budget: int = Field(default=0, description="Total budget units for this episode")
    budget_remaining: int = Field(default=0, description="Budget units remaining. Going negative adds a 0.02 penalty per unit over budget.")
    flags_raised: List[str] = Field(default_factory=list, description="Fraud signal flag IDs raised so far")
    discovered_signals: List[str] = Field(
        default_factory=list,
        description="Fraud signals actually discovered through allowed investigative actions.",
    )
    status: ClaimStatus = Field(default=ClaimStatus.OPEN, description="Current claim processing status")
    message: str = Field(default="", description="Human-readable message describing result of last action")
    confidence_required: bool = Field(default=True, description="Whether next action requires a confidence declaration")
    reward_breakdown: InsuranceClaimReward = Field(default_factory=InsuranceClaimReward, description="Detailed reward components for current step")


class InsuranceClaimState(State):
    task_id: str = ""
    claim_id: str = ""
    step_number: int = 0
    max_steps: int = 0
    status: ClaimStatus = ClaimStatus.OPEN
    flags_raised: List[str] = Field(default_factory=list)
    discovered_signals: List[str] = Field(default_factory=list)
    found_signals: List[str] = Field(default_factory=list)
    penalty_total: float = 0.0
    done: bool = False
    last_action_error: Optional[str] = None
    payout_estimate_inr: Optional[float] = None
    final_decision: Optional[str] = None
    final_score: float = 0.0
