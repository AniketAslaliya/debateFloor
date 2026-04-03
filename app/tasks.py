from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import InsuranceClaimReward


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    difficulty: str
    max_steps: int
    claim_id: str
    claimant: Dict[str, Any]
    incident: Dict[str, Any]
    documents: List[Dict[str, Any]]
    linked_claims: List[Dict[str, Any]]
    expected_signals: List[str]
    allowed_final_decisions: List[str]
    payout_band: Optional[tuple[float, float]]
    consistency_group_claim_ids: List[str]


def _base_available_actions() -> List[str]:
    return [
        "validate_document",
        "request_information",
        "flag_fraud_signal",
        "estimate_payout",
        "approve_claim",
        "deny_claim",
        "request_investigation",
    ]


TASKS: Dict[str, TaskDefinition] = {
    "clean_claim": TaskDefinition(
        task_id="clean_claim",
        title="Clean auto claim with complete evidence",
        difficulty="easy",
        max_steps=8,
        claim_id="CLM-AUTO-001",
        claimant={
            "name": "Rajesh Verma",
            "policy_number": "POL-AUTO-8821",
            "contact": "+91-9810012345",
            "claim_date": "2026-03-02",
        },
        incident={
            "date": "2026-02-27",
            "location": "Pune, Maharashtra",
            "type": "auto_collision",
            "description": "Rear-end collision at a traffic signal, bumper and tail-light damage.",
        },
        documents=[
            {
                "doc_id": "DOC-1",
                "doc_type": "claim_form",
                "content": "Claim submitted for rear-end collision on 2026-02-27.",
                "metadata": {"incident_date": "2026-02-27", "declared_cost_inr": 51000},
            },
            {
                "doc_id": "DOC-2",
                "doc_type": "garage_estimate",
                "content": "Repair estimate from authorized center.",
                "metadata": {"estimate_inr": 50500, "garage": "Pune Auto Care"},
            },
            {
                "doc_id": "DOC-3",
                "doc_type": "police_report",
                "content": "Minor collision report with matching date and location.",
                "metadata": {"incident_date": "2026-02-27", "report_id": "PR-112"},
            },
        ],
        linked_claims=[],
        expected_signals=[],
        allowed_final_decisions=["approve_claim"],
        payout_band=(45000, 55000),
        consistency_group_claim_ids=[],
    ),
    "contradictory_claim": TaskDefinition(
        task_id="contradictory_claim",
        title="Medical claim with contradictory evidence",
        difficulty="medium",
        max_steps=12,
        claim_id="CLM-MED-017",
        claimant={
            "name": "Neha Kapoor",
            "policy_number": "POL-HEALTH-2190",
            "contact": "+91-9822211188",
            "claim_date": "2026-03-05",
        },
        incident={
            "date": "2026-02-16",
            "location": "Ahmedabad, Gujarat",
            "type": "medical_procedure",
            "description": "Emergency appendectomy claim with post-op hospitalization.",
        },
        documents=[
            {
                "doc_id": "DOC-10",
                "doc_type": "claim_form",
                "content": "Claim incident date recorded as 2026-02-20.",
                "metadata": {"incident_date": "2026-02-20", "claimed_cost_inr": 240000},
            },
            {
                "doc_id": "DOC-11",
                "doc_type": "hospital_admission",
                "content": "Patient admitted on 2026-02-17 for emergency surgery.",
                "metadata": {"admission_date": "2026-02-17", "procedure": "appendectomy"},
            },
            {
                "doc_id": "DOC-12",
                "doc_type": "billing_summary",
                "content": "Total treatment cost billed: INR 240000.",
                "metadata": {"claimed_cost_inr": 240000, "standard_rate_inr": 100000},
            },
            {
                "doc_id": "DOC-13",
                "doc_type": "discharge_summary",
                "content": "Digitally scanned discharge summary.",
                "metadata": {
                    "doctor_signature": "DR-XYZ-SIGN-ALPHA",
                    "clinic_reference_signature": "DR-XYZ-SIGN-BETA",
                },
            },
        ],
        linked_claims=[],
        expected_signals=[
            "date_mismatch",
            "cost_inflation",
            "signature_mismatch",
        ],
        allowed_final_decisions=["deny_claim", "request_investigation"],
        payout_band=None,
        consistency_group_claim_ids=[],
    ),
    "coordinated_fraud": TaskDefinition(
        task_id="coordinated_fraud",
        title="Coordinated multi-claim fraud ring",
        difficulty="hard",
        max_steps=20,
        claim_id="CLM-GROUP-301",
        claimant={
            "name": "Primary: Arjun Saini",
            "policy_number": "POL-MOTOR-9001",
            "contact": "+91-9898001122",
            "claim_date": "2026-03-09",
        },
        incident={
            "date": "2026-03-01",
            "location": "Jaipur, Rajasthan",
            "type": "multi_vehicle_damage",
            "description": "Three separate claims likely linked by staged repairs and copied narratives.",
        },
        documents=[
            {
                "doc_id": "DOC-21",
                "doc_type": "primary_claim_packet",
                "content": "Primary claim references repair at RapidFix Motors in Kota (340km away).",
                "metadata": {"repair_shop": "RapidFix Motors", "distance_km": 340},
            },
            {
                "doc_id": "DOC-22",
                "doc_type": "narrative",
                "content": "Accident description text is nearly identical to two linked claims.",
                "metadata": {"template_similarity": 0.93},
            },
            {
                "doc_id": "DOC-23",
                "doc_type": "policy_timeline",
                "content": "All related policies purchased within 30 days of incident.",
                "metadata": {"days_since_purchase": [18, 24, 29]},
            },
        ],
        linked_claims=[
            {
                "claim_id": "CLM-GROUP-302",
                "claimant": "Rohit Jain",
                "contact": "+91-9898004455",
                "emergency_contact": "+91-9000002222",
                "repair_shop": "RapidFix Motors",
                "accident_description": "A truck abruptly stopped causing chain collision near city bypass.",
                "policy_purchase_date": "2026-02-06",
            },
            {
                "claim_id": "CLM-GROUP-303",
                "claimant": "Pooja Nair",
                "contact": "+91-9845509988",
                "emergency_contact": "+91-9000002222",
                "repair_shop": "RapidFix Motors",
                "accident_description": "A truck abruptly stopped causing chain collision near city bypass.",
                "policy_purchase_date": "2026-02-11",
            },
            {
                "claim_id": "CLM-GROUP-301",
                "claimant": "Arjun Saini",
                "contact": "+91-9898001122",
                "emergency_contact": "+91-9000003333",
                "repair_shop": "RapidFix Motors",
                "accident_description": "A truck abruptly stopped causing chain collision near city bypass.",
                "policy_purchase_date": "2026-02-02",
            },
        ],
        expected_signals=[
            "shared_repair_shop_far",
            "shared_emergency_contact",
            "near_identical_descriptions",
            "recent_policy_cluster",
        ],
        allowed_final_decisions=["request_investigation"],
        payout_band=None,
        consistency_group_claim_ids=["CLM-GROUP-301", "CLM-GROUP-302", "CLM-GROUP-303"],
    ),
}


def get_task_definition(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASKS)}")
    return TASKS[task_id]


def list_tasks_summary() -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for task in TASKS.values():
        summaries.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "difficulty": task.difficulty,
                "max_steps": task.max_steps,
                "expected_decisions": task.allowed_final_decisions,
            }
        )
    return summaries


def build_initial_payload(task_id: str) -> Dict[str, Any]:
    task = get_task_definition(task_id)
    return {
        "task_id": task.task_id,
        "claim_id": task.claim_id,
        "claimant": deepcopy(task.claimant),
        "incident": deepcopy(task.incident),
        "documents": deepcopy(task.documents),
        "linked_claims": deepcopy(task.linked_claims),
        "max_steps": task.max_steps,
        "available_actions": _base_available_actions(),
    }


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def score_payout_accuracy(amount: Optional[float], payout_band: Optional[tuple[float, float]]) -> float:
    if payout_band is None:
        return 1.0 if amount is None else 0.0
    if amount is None:
        return 0.0

    low, high = payout_band
    if low <= amount <= high:
        return 1.0

    band_center = (low + high) / 2.0
    tolerance = max((high - low) / 2.0, 1.0)
    distance = abs(amount - band_center)
    return clamp01(1.0 - (distance / (2.5 * tolerance)))


def score_consistency(task_id: str, raised_flags: List[str], investigation_targets: List[str]) -> float:
    if task_id != "coordinated_fraud":
        return 0.0

    has_flags = len(raised_flags) > 0
    targets = set(investigation_targets)
    expected = set(get_task_definition(task_id).consistency_group_claim_ids)

    if not has_flags:
        return 1.0

    if targets == expected:
        return 1.0

    if len(targets) == 0:
        return 0.0

    return 0.2


def compute_reward_breakdown(
    task_id: str,
    expected_signals: List[str],
    found_signals: List[str],
    false_flags: int,
    step_number: int,
    max_steps: int,
    final_decision: Optional[str],
    allowed_decisions: List[str],
    payout_estimate_inr: Optional[float],
    payout_band: Optional[tuple[float, float]],
    investigation_targets: List[str],
    penalty_total: float,
) -> InsuranceClaimReward:
    expected = set(expected_signals)
    found = set(found_signals)

    if len(expected) == 0:
        fraud_detection_score = 1.0 if len(found) == 0 else 0.0
    else:
        fraud_detection_score = clamp01(len(found.intersection(expected)) / float(len(expected)))

    if final_decision is None:
        decision_accuracy = 0.0
    else:
        decision_accuracy = 1.0 if final_decision in allowed_decisions else 0.0

    payout_accuracy = score_payout_accuracy(payout_estimate_inr, payout_band)
    efficiency_score = clamp01(1.0 - (max(step_number - 1, 0) / float(max_steps)))
    consistency_score = score_consistency(task_id, found_signals, investigation_targets)

    false_flag_penalty = 0.25 * false_flags if task_id == "clean_claim" else 0.1 * false_flags
    decision_penalty = 0.35 if (final_decision is not None and decision_accuracy == 0.0) else 0.0
    partial_consistency_penalty = 0.2 if (task_id == "coordinated_fraud" and 0.0 < consistency_score < 1.0) else 0.0

    penalty = penalty_total + false_flag_penalty + decision_penalty + partial_consistency_penalty

    # Weighted sum, then subtract penalties and clamp.
    weighted = (
        0.35 * fraud_detection_score
        + 0.25 * decision_accuracy
        + 0.15 * payout_accuracy
        + 0.15 * efficiency_score
        + 0.10 * consistency_score
    )

    total = clamp01(weighted - penalty)

    return InsuranceClaimReward(
        fraud_detection_score=clamp01(fraud_detection_score),
        decision_accuracy=clamp01(decision_accuracy),
        payout_accuracy=clamp01(payout_accuracy),
        efficiency_score=clamp01(efficiency_score),
        consistency_score=clamp01(consistency_score),
        penalty=round(penalty, 4),
        total=round(total, 4),
    )
