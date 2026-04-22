from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import InsuranceClaimReward


# Budget units consumed per action type. Final decisions are free.
ACTION_COSTS: Dict[str, int] = {
    "validate_document": 1,
    "request_information": 2,
    "lookup_policy_history": 1,
    "compare_documents": 1,
    "flag_fraud_signal": 1,
    "estimate_payout": 1,
    "query_linked_claim": 1,
    "verify_identity": 2,
    "query_historical_data": 1,
    "verify_provider_registration": 1,
    "convene_debate_panel": 2,   # multi-agent deliberation costs 2 budget units
    "approve_claim": 0,
    "deny_claim": 0,
    "request_investigation": 0,
    "escalate_to_human": 0,
}


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    difficulty: str
    max_steps: int
    investigation_budget: int       # soft budget; overage adds 0.02 penalty per unit
    claim_id: str
    claimant: Dict[str, Any]
    incident: Dict[str, Any]
    documents: List[Dict[str, Any]]
    linked_claims: List[Dict[str, Any]]
    expected_signals: List[str]
    allowed_final_decisions: List[str]
    payout_band: Optional[tuple[float, float]]
    consistency_group_claim_ids: List[str]
    policy_history: Dict[str, Any]
    ground_truth_confidence: float


@dataclass
class RuntimeTask:
    task_id: str
    title: str
    difficulty: str
    max_steps: int
    investigation_budget: int
    claim_id: str
    claimant: Dict[str, Any]
    incident: Dict[str, Any]
    documents: List[Dict[str, Any]]
    linked_claims: List[Dict[str, Any]]
    expected_signals: List[str]
    allowed_final_decisions: List[str]
    payout_band: Optional[tuple[float, float]]
    consistency_group_claim_ids: List[str]
    policy_history: Dict[str, Any]
    ground_truth_confidence: float
    variant_id: int


def _base_available_actions(task_id: str = "") -> List[str]:
    actions = [
        "validate_document",
        "request_information",
        "lookup_policy_history",
        "compare_documents",
        "flag_fraud_signal",
        "estimate_payout",
        "approve_claim",
        "deny_claim",
        "request_investigation",
    ]
    if task_id in ("coordinated_fraud", "distribution_shift_claim"):
        actions.append("query_linked_claim")
        actions.append("query_historical_data")
        actions.append("escalate_to_human")
    if task_id == "distribution_shift_claim":
        actions.append("verify_provider_registration")
    if task_id == "identity_fraud":
        actions.append("verify_identity")
    if task_id != "clean_claim":
        actions.append("convene_debate_panel")
    return actions


TASKS: Dict[str, TaskDefinition] = {
    "clean_claim": TaskDefinition(
        task_id="clean_claim",
        title="Clean auto claim with complete evidence",
        difficulty="easy",
        max_steps=8,
        investigation_budget=8,   # validate×3 + estimate + approve = 5; 3 units slack
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
        policy_history={
            "prior_claims": [],
            "years_as_customer": 6,
            "policy_age_days": 2190,
            "risk_score": "low",
            "note": "Long-standing customer with no prior claims. Low risk.",
        },
        ground_truth_confidence=0.95,
    ),
    "contradictory_claim": TaskDefinition(
        task_id="contradictory_claim",
        title="Medical claim with contradictory evidence",
        difficulty="medium",
        max_steps=12,
        investigation_budget=12,  # validate×4 + lookup + flag×4 + deny = 10; 2 units slack
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
            "prior_similar_claim",
        ],
        allowed_final_decisions=["deny_claim", "request_investigation"],
        payout_band=None,
        consistency_group_claim_ids=[],
        policy_history={
            "prior_claims": [
                {
                    "claim_id": "CLM-MED-008",
                    "date": "2025-06-14",
                    "type": "medical_procedure",
                    "procedure": "appendectomy",
                    "paid_inr": 95000,
                    "note": "Claim paid for same procedure 8 months ago at a different hospital.",
                }
            ],
            "years_as_customer": 3,
            "policy_age_days": 1095,
            "risk_score": "medium",
            "note": "Prior claim for identical procedure 8 months ago — statistical anomaly.",
        },
        ground_truth_confidence=0.80,
    ),
    "coordinated_fraud": TaskDefinition(
        task_id="coordinated_fraud",
        title="Coordinated multi-claim fraud ring",
        difficulty="hard",
        max_steps=20,
        investigation_budget=18,  # validate×3 + query×4 + flag×5 + escalate = 15; 3 units slack
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
                "broker_id": "BRK-441",
            },
            # 4th claim — hidden until agent queries 2 linked claims (dynamic ring expansion)
            {
                "claim_id": "CLM-GROUP-304",
                "claimant": "Vikram Sharma",
                "contact": "+91-9011009988",
                "emergency_contact": "+91-9000002222",
                "repair_shop": "RapidFix Motors",
                "accident_description": "A truck abruptly stopped causing chain collision near city bypass.",
                "policy_purchase_date": "2026-02-08",
                "broker_id": "BRK-441",
                "_hidden_until_queries": 2,   # surfaced only after 2 existing claims are queried
            },
        ],
        expected_signals=[
            "shared_repair_shop_far",
            "shared_emergency_contact",
            "near_identical_descriptions",
            "recent_policy_cluster",
            "clustered_policy_broker",   # discoverable by querying the 4th linked claim
        ],
        allowed_final_decisions=["request_investigation"],
        payout_band=None,
        consistency_group_claim_ids=["CLM-GROUP-301", "CLM-GROUP-302", "CLM-GROUP-303", "CLM-GROUP-304"],
        policy_history={
            "prior_claims": [],
            "years_as_customer": 0,
            "policy_age_days": 18,
            "risk_score": "high",
            "note": "Policy purchased only 18 days before incident. No claim history — all three claimants opened policies within 30 days of each other.",
        },
        ground_truth_confidence=0.90,
    ),
    "distribution_shift_claim": TaskDefinition(
        task_id="distribution_shift_claim",
        title="Cross-claim coordinated ring with distribution shift",
        difficulty="hard",
        max_steps=28,
        investigation_budget=20,
        claim_id="CLM-DIST-601",
        claimant={
            "name": "Suresh Pillai",
            "policy_number": "POL-MOTOR-5541",
            "contact": "+91-9876543210",
            "claim_date": "2026-03-15",
        },
        incident={
            "date": "2026-03-08",
            "location": "Bengaluru, Karnataka",
            "type": "auto_collision",
            "description": "Minor collision at junction. Claim appears routine on surface but cross-claim analysis reveals coordinated ring.",
        },
        documents=[
            {
                "doc_id": "DOC-41",
                "doc_type": "claim_form",
                "content": "Standard auto collision claim submitted on 2026-03-15 for incident on 2026-03-08.",
                "metadata": {"incident_date": "2026-03-08", "declared_cost_inr": 85000},
            },
            {
                "doc_id": "DOC-42",
                "doc_type": "garage_estimate",
                "content": "Repair estimate from FastRepair Hub, Whitefield.",
                "metadata": {"estimate_inr": 84000, "garage": "FastRepair Hub"},
            },
            {
                "doc_id": "DOC-43",
                "doc_type": "police_report",
                "content": "Minor collision report filed. No independent witnesses.",
                "metadata": {"incident_date": "2026-03-08", "witnesses": 0},
            },
        ],
        linked_claims=[
            {
                "claim_id": "CLM-DIST-602",
                "claimant": "Meera Iyer",
                "contact": "+91-9845501234",
                "emergency_contact": "+91-9000005555",
                "repair_shop": "FastRepair Hub",
                "accident_description": "Minor collision at junction. No injuries.",
                "policy_purchase_date": "2026-02-12",
                "broker_id": "BRK-882",
            },
            {
                "claim_id": "CLM-DIST-603",
                "claimant": "Ravi Shankar",
                "contact": "+91-9741200099",
                "emergency_contact": "+91-9000005555",
                "repair_shop": "FastRepair Hub",
                "accident_description": "Minor collision at junction. No injuries.",
                "policy_purchase_date": "2026-02-18",
                "broker_id": "BRK-882",
            },
            {
                "claim_id": "CLM-DIST-604",
                "claimant": "Deepa Nair",
                "contact": "+91-9911200033",
                "emergency_contact": "+91-9000005555",
                "repair_shop": "FastRepair Hub",
                "accident_description": "Minor collision at junction. No injuries.",
                "policy_purchase_date": "2026-02-20",
                "broker_id": "BRK-882",
                "_hidden_until_queries": 2,
            },
        ],
        expected_signals=[
            "shared_repair_shop_far",
            "shared_emergency_contact",
            "recent_policy_cluster",
            "clustered_policy_broker",
            "near_identical_descriptions",
        ],
        allowed_final_decisions=["escalate_to_human", "request_investigation"],
        payout_band=None,
        consistency_group_claim_ids=["CLM-DIST-601", "CLM-DIST-602", "CLM-DIST-603", "CLM-DIST-604"],
        policy_history={
            "prior_claims": [],
            "years_as_customer": 0,
            "policy_age_days": 24,
            "risk_score": "high",
            "note": "Policy purchased 24 days before incident. All 3 linked claimants share broker BRK-882 and same repair shop. Cross-claim cluster detected in historical data.",
        },
        ground_truth_confidence=0.70,
    ),
    "identity_fraud": TaskDefinition(
        task_id="identity_fraud",
        title="Ghost claimant identity fraud",
        difficulty="hard",
        max_steps=15,
        investigation_budget=14,  # verify(2)+lookup+validate×4+flag×4+deny = 11; 3 units slack
        claim_id="CLM-ID-501",
        claimant={
            "name": "Aarav Mehta",
            "policy_number": "POL-HEALTH-7734",
            "contact": "+91-9711100045",
            "claim_date": "2026-03-12",
            "national_id": "XXXX-7821",
        },
        incident={
            "date": "2026-03-07",
            "location": "Mumbai, Maharashtra",
            "type": "medical_procedure",
            "description": "Knee replacement surgery claim with post-op physiotherapy.",
        },
        documents=[
            {
                "doc_id": "DOC-31",
                "doc_type": "claim_form",
                "content": "Claim submitted for knee replacement on 2026-03-07. National ID: XXXX-7821.",
                "metadata": {
                    "incident_date": "2026-03-07",
                    "claimed_cost_inr": 320000,
                    "national_id_suffix": "7821",
                },
            },
            {
                "doc_id": "DOC-32",
                "doc_type": "hospital_record",
                "content": "Hospital system query: No patient named Aarav Mehta with DOB matching policy found. Record shows admission under a different name with similar ID.",
                "metadata": {
                    "patient_found": False,
                    "name_on_record": "Aarav Kumar",
                    "dob_mismatch": True,
                },
            },
            {
                "doc_id": "DOC-33",
                "doc_type": "policy_inception",
                "content": "Policy POL-HEALTH-7734 issued on 2026-03-02. Incident date 2026-03-07 falls within the 30-day exclusion window.",
                "metadata": {
                    "policy_issue_date": "2026-03-02",
                    "incident_date": "2026-03-07",
                    "days_to_claim": 5,
                    "exclusion_window_days": 30,
                },
            },
            {
                "doc_id": "DOC-34",
                "doc_type": "id_proof",
                "content": "Submitted ID proof shows date of birth 1988-04-15. Policy application on file states DOB 1986-11-22. The national registry has no record matching either entry for this ID number.",
                "metadata": {
                    "dob_on_id": "1988-04-15",
                    "dob_on_policy": "1986-11-22",
                    "registry_match": False,
                },
            },
        ],
        linked_claims=[],
        expected_signals=[
            "identity_mismatch",
            "hospital_no_record",
            "recent_policy_purchase",
            "dob_inconsistency",
        ],
        allowed_final_decisions=["deny_claim", "request_investigation"],
        payout_band=None,
        consistency_group_claim_ids=[],
        policy_history={
            "prior_claims": [],
            "years_as_customer": 0,
            "policy_age_days": 5,
            "risk_score": "critical",
            "note": "Policy opened only 5 days before incident. Claimant identity could not be verified at onboarding. KYC status: PENDING.",
        },
        ground_truth_confidence=0.90,
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
        investigation_budget=task.investigation_budget,
        claim_id=task.claim_id,
        claimant=deepcopy(task.claimant),
        incident=deepcopy(task.incident),
        documents=deepcopy(task.documents),
        linked_claims=deepcopy(task.linked_claims),
        expected_signals=deepcopy(task.expected_signals),
        allowed_final_decisions=deepcopy(task.allowed_final_decisions),
        payout_band=deepcopy(task.payout_band),
        consistency_group_claim_ids=deepcopy(task.consistency_group_claim_ids),
        policy_history=deepcopy(task.policy_history),
        ground_truth_confidence=task.ground_truth_confidence,
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

    elif task_id == "identity_fraud":
        # Vary days_to_claim and policy inception date across variants
        days_to_claim_variants = [5, 7, 3, 8, 6]
        days_to_claim = days_to_claim_variants[variant_id]
        runtime.documents[2]["metadata"]["days_to_claim"] = days_to_claim
        runtime.documents[2]["content"] = (
            f"Policy POL-HEALTH-7734 issued 2026-03-{12 - days_to_claim:02d}. "
            f"Incident date 2026-03-07 falls within the 30-day exclusion window."
        )
        runtime.policy_history = deepcopy(task.policy_history)
        runtime.policy_history["policy_age_days"] = days_to_claim

    return runtime


def _stub_linked_claims(linked_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only claim_id and claimant. Hidden claims (with _hidden_until_queries > 0)
    are excluded from the initial list — they surface dynamically in the environment."""
    return [
        {"claim_id": c["claim_id"], "claimant": c["claimant"]}
        for c in linked_claims
        if "claim_id" in c and c.get("_hidden_until_queries", 0) == 0
    ]


def build_initial_payload(runtime_task: RuntimeTask) -> Dict[str, Any]:
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
        "_full_linked_claims": deepcopy(runtime_task.linked_claims),
        "max_steps": runtime_task.max_steps,
        "investigation_budget": runtime_task.investigation_budget,
        "variant_id": runtime_task.variant_id,
        "available_actions": _base_available_actions(runtime_task.task_id),
    }


def get_evidence_keyword_hints(task_id: str, flag_id: str) -> List[str]:
    hints: Dict[str, Dict[str, List[str]]] = {
        "contradictory_claim": {
            "date_mismatch": ["date", "admission", "mismatch", "incident"],
            "cost_inflation": ["cost", "rate", "2.4", "inflation", "overbilled"],
            "signature_mismatch": ["signature", "doctor", "clinic", "dr-xyz"],
            "prior_similar_claim": ["prior", "previous", "history", "appendectomy", "procedure", "8 months", "clm-med-008"],
        },
        "coordinated_fraud": {
            "shared_repair_shop_far": ["repair", "shop", "distance", "km", "kota", "rapidfix"],
            "shared_emergency_contact": ["contact", "phone", "emergency", "shared", "9000002222"],
            "near_identical_descriptions": ["identical", "description", "narrative", "template", "similarity"],
            "recent_policy_cluster": ["policy", "purchase", "days", "cluster", "30"],
            "clustered_policy_broker": ["broker", "brk-441", "same broker", "policy broker", "issued"],
        },
        "identity_fraud": {
            "identity_mismatch": ["identity", "registry", "national", "id", "mismatch", "no record", "7821"],
            "hospital_no_record": ["hospital", "record", "patient", "not found", "name", "admission"],
            "recent_policy_purchase": ["policy", "days", "exclusion", "window", "inception", "5", "30"],
            "dob_inconsistency": ["dob", "date of birth", "1988", "1986", "inconsistency", "mismatch"],
        },
    }
    return hints.get(task_id, {}).get(flag_id, [])


# Cross-document comparison signal mapping: (doc_a, doc_b) → signals discovered
COMPARE_DOCUMENT_SIGNALS: Dict[str, Dict[tuple, List[str]]] = {
    "contradictory_claim": {
        ("DOC-10", "DOC-11"): ["date_mismatch"],
        ("DOC-11", "DOC-10"): ["date_mismatch"],
        ("DOC-10", "DOC-12"): ["cost_inflation"],
        ("DOC-12", "DOC-10"): ["cost_inflation"],
    },
    "coordinated_fraud": {
        ("DOC-21", "DOC-22"): ["near_identical_descriptions"],
        ("DOC-22", "DOC-21"): ["near_identical_descriptions"],
    },
    "identity_fraud": {
        ("DOC-31", "DOC-34"): ["dob_inconsistency"],
        ("DOC-34", "DOC-31"): ["dob_inconsistency"],
        ("DOC-32", "DOC-33"): ["hospital_no_record"],
        ("DOC-33", "DOC-32"): ["hospital_no_record"],
    },
}


def get_compare_signals(task_id: str, doc_id_a: str, doc_id_b: str) -> List[str]:
    return COMPARE_DOCUMENT_SIGNALS.get(task_id, {}).get((doc_id_a, doc_id_b), [])


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


def score_calibration(agent_confidence: Optional[float], ground_truth_confidence: float) -> float:
    """Brier-style calibration score.

    Returns 1 - (agent_confidence - ground_truth)^2, in [0, 1].
    If agent did not provide a confidence, returns 0.0 (no bonus, no penalty).
    """
    if agent_confidence is None:
        return 0.0
    agent_conf = clamp01(float(agent_confidence))
    return clamp01(1.0 - (agent_conf - ground_truth_confidence) ** 2)


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
    agent_confidence: Optional[float] = None,
    ground_truth_confidence: float = 1.0,
) -> InsuranceClaimReward:
    expected = set(expected_signals)
    found = set(found_signals)

    # --- Fraud detection ---
    if step_number == 0:
        fraud_detection_score = 0.0
    elif len(expected) == 0:
        fraud_detection_score = 1.0 if len(found) == 0 else 0.0
    else:
        fraud_detection_score = clamp01(len(found.intersection(expected)) / float(len(expected)))

    # --- Decision accuracy ---
    if final_decision is None:
        decision_accuracy = 0.0
    else:
        decision_accuracy = 1.0 if final_decision in allowed_decisions else 0.0

    # --- Payout accuracy ---
    if step_number == 0:
        payout_accuracy = 0.0
    elif payout_band is None:
        # Non-payout tasks should not receive a free reward bump before a final decision.
        payout_accuracy = 1.0 if final_decision is not None else 0.0
    else:
        payout_accuracy = score_payout_accuracy(payout_estimate_inr, payout_band)

    # --- Efficiency ---
    has_queried = queried_claims is not None and len(queried_claims) > 0
    has_progress = len(found) > 0 or payout_estimate_inr is not None or has_queried
    if has_progress or final_decision is not None:
        efficiency_score = clamp01(1.0 - (max(step_number - 1, 0) / float(max_steps)))
    else:
        efficiency_score = 0.0

    consistency_score = 0.0
    if step_number > 0 and final_decision == "request_investigation":
        consistency_score = score_consistency(task_id, found_signals, investigation_targets, queried_claims)

    evidence_quality_score = clamp01(evidence_quality_score)

    # --- Calibration: only scored when a final decision is made ---
    if final_decision is not None:
        calibration_score = score_calibration(agent_confidence, ground_truth_confidence)
    else:
        calibration_score = 0.0

    exploit_penalty = max(exploit_penalty, 0.0)
    false_flag_penalty = 0.25 * false_flags if task_id == "clean_claim" else 0.1 * false_flags
    decision_penalty = 0.35 if (final_decision is not None and decision_accuracy == 0.0) else 0.0
    partial_consistency_penalty = 0.2 if (task_id == "coordinated_fraud" and 0.0 < consistency_score < 1.0) else 0.0

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

    # Weights: sum = 1.00
    # Reduced fraud/decision/evidence slightly to make room for calibration (0.08)
    weighted = (
        0.28 * fraud_detection_score
        + 0.20 * decision_accuracy
        + 0.11 * payout_accuracy
        + 0.10 * efficiency_score
        + 0.09 * consistency_score
        + 0.14 * evidence_quality_score
        + 0.08 * calibration_score
    )

    total = clamp01(weighted - penalty)

    return InsuranceClaimReward(
        fraud_detection_score=clamp01(fraud_detection_score),
        decision_accuracy=clamp01(decision_accuracy),
        payout_accuracy=clamp01(payout_accuracy),
        efficiency_score=clamp01(efficiency_score),
        consistency_score=clamp01(consistency_score),
        evidence_quality_score=evidence_quality_score,
        calibration_score=clamp01(calibration_score),
        exploit_penalty=round(exploit_penalty, 4),
        penalty=round(penalty, 4),
        total=round(total, 4),
    )
