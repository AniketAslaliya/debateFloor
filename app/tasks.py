from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
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


@dataclass
class RuntimeTask:
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
    variant_id: int


def _base_available_actions() -> List[str]:
    return [
        "validate_document",
        "request_information",
        "flag_fraud_signal",
        "estimate_payout",
        "approve_claim",
        "deny_claim",
        "request_investigation",
        "query_linked_claim",
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


def _copy_runtime_from_task(task: TaskDefinition, variant_id: int) -> RuntimeTask:
    return RuntimeTask(
        task_id=task.task_id,
        title=task.title,
        difficulty=task.difficulty,
        max_steps=task.max_steps,
        claim_id=task.claim_id,
        claimant=deepcopy(task.claimant),
        incident=deepcopy(task.incident),
        documents=deepcopy(task.documents),
        linked_claims=deepcopy(task.linked_claims),
        expected_signals=deepcopy(task.expected_signals),
        allowed_final_decisions=deepcopy(task.allowed_final_decisions),
        payout_band=deepcopy(task.payout_band),
        consistency_group_claim_ids=deepcopy(task.consistency_group_claim_ids),
        variant_id=variant_id,
    )


def build_runtime_task(task_id: str, seed: Optional[int] = None) -> RuntimeTask:
    task = get_task_definition(task_id)
    variant_id = 0 if seed is None else abs(seed) % 5
    runtime = _copy_runtime_from_task(task, variant_id)

    if task_id == "clean_claim":
        offsets = [-2000, -1000, 0, 1000, 2000]
        offset = offsets[variant_id]
        declared_cost = 51000 + offset
        estimate = 50500 + offset
        runtime.documents[0]["metadata"]["declared_cost_inr"] = declared_cost
        runtime.documents[1]["metadata"]["estimate_inr"] = estimate
        center = 50000 + offset
        runtime.payout_band = (float(center - 5000), float(center + 5000))

    elif task_id == "contradictory_claim":
        admission_date_str = runtime.documents[1]["metadata"]["admission_date"]
        admission_date = datetime.strptime(admission_date_str, "%Y-%m-%d")
        date_gap_days = [3, 4, 2, 5, 3][variant_id]
        incident_date = (admission_date + timedelta(days=date_gap_days)).strftime("%Y-%m-%d")
        runtime.documents[0]["metadata"]["incident_date"] = incident_date
        runtime.documents[0]["content"] = f"Claim incident date recorded as {incident_date}."

        standard_rates = [100000, 105000, 95000, 110000, 98000]
        standard_rate = standard_rates[variant_id]
        claimed_cost = int(standard_rate * 2.4)
        runtime.documents[0]["metadata"]["claimed_cost_inr"] = claimed_cost
        runtime.documents[2]["metadata"]["claimed_cost_inr"] = claimed_cost
        runtime.documents[2]["metadata"]["standard_rate_inr"] = standard_rate
        runtime.documents[2]["content"] = f"Total treatment cost billed: INR {claimed_cost}."

    elif task_id == "coordinated_fraud":
        distances = [340, 360, 320, 380, 300]
        distance = distances[variant_id]
        runtime.documents[0]["metadata"]["distance_km"] = distance
        runtime.documents[0]["content"] = (
            f"Primary claim references repair at RapidFix Motors in Kota ({distance}km away)."
        )

        similarity = [0.93, 0.91, 0.95, 0.9, 0.94][variant_id]
        runtime.documents[1]["metadata"]["template_similarity"] = similarity

        purchase_sets = [
            [18, 24, 29],
            [12, 22, 27],
            [9, 19, 28],
            [16, 21, 26],
            [14, 25, 30],
        ]
        runtime.documents[2]["metadata"]["days_since_purchase"] = purchase_sets[variant_id]

    return runtime


def _stub_linked_claims(linked_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only claim_id and claimant for each linked claim.

    Full details are intentionally withheld until the agent calls query_linked_claim.
    This forces multi-hop reasoning on the coordinated_fraud task.
    """
    return [
        {"claim_id": c["claim_id"], "claimant": c["claimant"]}
        for c in linked_claims
        if "claim_id" in c
    ]


def build_initial_payload(runtime_task: RuntimeTask) -> Dict[str, Any]:
    # For coordinated_fraud, hide fraud signals until agent queries each linked claim.
    if runtime_task.task_id == "coordinated_fraud":
        linked_claims_visible = _stub_linked_claims(runtime_task.linked_claims)
    else:
        linked_claims_visible = deepcopy(runtime_task.linked_claims)

    return {
        "task_id": runtime_task.task_id,
        "claim_id": runtime_task.claim_id,
        "claimant": deepcopy(runtime_task.claimant),
        "incident": deepcopy(runtime_task.incident),
        "documents": deepcopy(runtime_task.documents),
        "linked_claims": linked_claims_visible,
        # Full linked_claims data stored separately for query_linked_claim lookups
        "_full_linked_claims": deepcopy(runtime_task.linked_claims),
        "max_steps": runtime_task.max_steps,
        "variant_id": runtime_task.variant_id,
        "available_actions": _base_available_actions(),
    }


def get_evidence_keyword_hints(task_id: str, flag_id: str) -> List[str]:
    hints: Dict[str, Dict[str, List[str]]] = {
        "contradictory_claim": {
            "date_mismatch": ["date", "admission", "mismatch", "incident"],
            "cost_inflation": ["cost", "rate", "2.4", "inflation", "overbilled"],
            "signature_mismatch": ["signature", "doctor", "clinic", "dr-xyz"],
        },
        "coordinated_fraud": {
            "shared_repair_shop_far": ["repair", "shop", "distance", "340", "kota"],
            "shared_emergency_contact": ["contact", "phone", "emergency", "shared"],
            "near_identical_descriptions": ["identical", "description", "same narrative"],
            "recent_policy_cluster": ["policy", "purchase", "30 days", "cluster"],
        },
    }
    return hints.get(task_id, {}).get(flag_id, [])


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


def score_consistency(
    task_id: str,
    raised_flags: List[str],
    investigation_targets: List[str],
    queried_claims: Optional[set] = None,
) -> float:
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
    evidence_quality_score: float,
    exploit_penalty: float,
    penalty_total: float,
    queried_claims: Optional[set] = None,
) -> InsuranceClaimReward:
    expected = set(expected_signals)
    found = set(found_signals)

    # --- Fraud detection: partial credit at every step ---
    if len(expected) == 0:
        fraud_detection_score = 1.0 if len(found) == 0 else 0.0
    else:
        fraud_detection_score = clamp01(len(found.intersection(expected)) / float(len(expected)))

    # --- Decision accuracy: only when final_decision is set ---
    if final_decision is None:
        decision_accuracy = 0.0
    else:
        decision_accuracy = 1.0 if final_decision in allowed_decisions else 0.0

    # --- Payout accuracy: partial credit once an estimate is given ---
    payout_accuracy = score_payout_accuracy(payout_estimate_inr, payout_band)

    # --- Efficiency: partial credit from step 1 onward if any progress ---
    has_progress = len(found) > 0 or payout_estimate_inr is not None
    if has_progress or final_decision is not None:
        efficiency_score = clamp01(1.0 - (max(step_number - 1, 0) / float(max_steps)))
    else:
        efficiency_score = 0.0

    consistency_score = score_consistency(task_id, found_signals, investigation_targets, queried_claims)

    # --- Evidence quality: partial credit at every step ---
    evidence_quality_score = clamp01(evidence_quality_score)

    exploit_penalty = max(exploit_penalty, 0.0)

    false_flag_penalty = 0.25 * false_flags if task_id == "clean_claim" else 0.1 * false_flags
    decision_penalty = 0.35 if (final_decision is not None and decision_accuracy == 0.0) else 0.0
    partial_consistency_penalty = 0.2 if (task_id == "coordinated_fraud" and 0.0 < consistency_score < 1.0) else 0.0

    # For coordinated_fraud: penalize request_investigation without querying at least 2 linked claims
    query_skip_penalty = 0.0
    if (
        task_id == "coordinated_fraud"
        and final_decision == "request_investigation"
        and (queried_claims is None or len(queried_claims) < 2)
    ):
        query_skip_penalty = 0.15

    penalty = (
        penalty_total
        + false_flag_penalty
        + decision_penalty
        + partial_consistency_penalty
        + query_skip_penalty
        + exploit_penalty
    )

    # Weighted sum, then subtract penalties and clamp.
    weighted = (
        0.30 * fraud_detection_score
        + 0.22 * decision_accuracy
        + 0.12 * payout_accuracy
        + 0.11 * efficiency_score
        + 0.10 * consistency_score
        + 0.15 * evidence_quality_score
    )

    total = clamp01(weighted - penalty)

    return InsuranceClaimReward(
        fraud_detection_score=clamp01(fraud_detection_score),
        decision_accuracy=clamp01(decision_accuracy),
        payout_accuracy=clamp01(payout_accuracy),
        efficiency_score=clamp01(efficiency_score),
        consistency_score=clamp01(consistency_score),
        evidence_quality_score=evidence_quality_score,
        exploit_penalty=round(exploit_penalty, 4),
        penalty=round(penalty, 4),
        total=round(total, 4),
    )
