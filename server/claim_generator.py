"""
server/claim_generator.py
DebateFloor — Procedural Claim Generator

Transforms DebateFloor from a fixed benchmark into a training environment.
Same (seed, fraud_type, coverage, difficulty) always produces the same episode.
Different seeds produce different claimant names, amounts, dates, and signal strengths.

5 fraud types x 4 coverage types x 3 jurisdictions x seed variation = 500+ unique episodes.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

FRAUD_TYPES = [
    "staged_accident",
    "medical_inflation",
    "identity_fraud",
    "coordinated_ring",
    "phantom_provider",
]

COVERAGE_TYPES = ["auto", "health", "property", "life"]

JURISDICTIONS = ["MH", "DL", "KA"]  # Maharashtra, Delhi, Karnataka

DIFFICULTY_SIGNAL_STRENGTH = {
    "easy":   0.90,
    "medium": 0.55,
    "hard":   0.20,
}

DIFFICULTY_AMBIGUITY = {
    "easy":   0.10,
    "medium": 0.45,
    "hard":   0.80,
}

FRAUD_GROUND_TRUTH = {
    "staged_accident":   "deny_claim",
    "medical_inflation": "deny_claim",
    "identity_fraud":    "deny_claim",
    "coordinated_ring":  "escalate_to_human",
    "phantom_provider":  "deny_claim",
    "none":              "approve_claim",
}

_FIRST_NAMES = [
    "Arjun", "Priya", "Rahul", "Sunita", "Vikram", "Meena",
    "Rohit", "Kavita", "Sanjay", "Anjali", "Deepak", "Pooja",
    "Nikhil", "Rekha", "Amit", "Divya", "Suresh", "Nisha",
    "Kiran", "Manoj", "Sneha", "Rajesh", "Lata", "Arun",
]
_LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Joshi", "Verma",
    "Gupta", "Mehta", "Nair", "Reddy", "Das", "Iyer",
    "Bhat", "Rao", "Pillai", "Saxena", "Tiwari", "Mishra",
]
_HOSPITALS = [
    "Apollo Hospital", "Fortis Healthcare", "Manipal Hospital",
    "Max Super Speciality", "Narayana Health", "Medanta",
    "Kokilaben Dhirubhai Ambani", "Aster CMI", "Lilavati Hospital",
]
_GARAGES = [
    "Tata Authorised Service", "Maruti True Value Workshop",
    "Hyundai Care Centre", "Popular Motors", "City Auto Works",
    "Highway Motors", "Star Auto Repair",
]
_INSURERS = ["HDFC ERGO", "ICICI Lombard", "Bajaj Allianz", "New India Assurance", "United India"]


# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────

class ClaimScenario(BaseModel):
    claim_id: str
    seed: int
    fraud_type: str
    coverage_type: str
    jurisdiction: str
    difficulty: str
    claimant: Dict[str, Any]
    incident: Dict[str, Any]
    documents: List[Dict[str, Any]]
    ground_truth: str
    ambiguity_score: float = Field(ge=0.0, le=1.0)
    payout_amount_inr: float
    expected_fraud_signals: List[str]
    linked_claims: List[Dict[str, Any]] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    max_steps: int = 10
    task_id: str = ""


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_claimant(rng: random.Random, jurisdiction: str) -> Dict[str, Any]:
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    return {
        "name": f"{first} {last}",
        "age": rng.randint(24, 62),
        "policy_number": f"POL-{jurisdiction}-{rng.randint(100000, 999999)}",
        "policy_start_date": f"202{rng.randint(1,4)}-{rng.randint(1,12):02d}-01",
        "insurer": rng.choice(_INSURERS),
        "jurisdiction": jurisdiction,
        "phone": f"+91-{rng.randint(7000000000, 9999999999)}",
    }


def _incident_date(rng: random.Random) -> str:
    return f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"


def _base_payout(coverage: str, rng: random.Random) -> float:
    ranges = {
        "auto":     (80_000,  450_000),
        "health":   (120_000, 800_000),
        "property": (200_000, 2_000_000),
        "life":     (500_000, 5_000_000),
    }
    lo, hi = ranges[coverage]
    return round(rng.uniform(lo, hi), -3)


# ─────────────────────────────────────────────────────────────
# FRAUD TYPE BUILDERS
# ─────────────────────────────────────────────────────────────

def _build_staged_accident(rng: random.Random, claimant: Dict, coverage: str, ss: float) -> Dict:
    payout = _base_payout(coverage, rng)
    inflated = round(payout * rng.uniform(1.4, 2.1), -3)
    garage = rng.choice(_GARAGES)
    date = _incident_date(rng)
    cost_mismatch = ss > 0.5

    docs = [
        {
            "doc_id": "DOC-001", "doc_type": "FIR",
            "content": f"FIR filed {date}. Vehicle collision at NH-48. Minor scratches and bumper dent.",
            "is_tampered": False, "tamper_signal": None,
        },
        {
            "doc_id": "DOC-002", "doc_type": "repair_estimate",
            "content": (
                f"Estimate from {garage}: Rs {inflated:,.0f}. "
                f"{'Engine replacement, full front assembly, airbag deployment.' if cost_mismatch else 'Bumper repair, paint job.'}"
            ),
            "is_tampered": cost_mismatch,
            "tamper_signal": "cost_mismatch_with_damage" if cost_mismatch else None,
        },
        {
            "doc_id": "DOC-003", "doc_type": "witness_statement",
            "content": (
                f"Witness {rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}: "
                f"'Vehicle was {'stationary when struck' if ss > 0.6 else 'moving normally'}.'"
            ),
            "is_tampered": ss > 0.75,
            "tamper_signal": "witness_inconsistency" if ss > 0.75 else None,
        },
    ]

    signals = []
    if cost_mismatch:
        signals.append("cost_mismatch_with_damage")
    if ss > 0.75:
        signals.append("witness_inconsistency")
    if ss > 0.85:
        signals.append("no_third_party_damage")

    return {
        "incident": {
            "date": date, "type": "vehicle_collision",
            "location": f"NH-48, {claimant['jurisdiction']}",
            "description": "Collision reported on national highway.",
            "claimed_amount_inr": inflated,
        },
        "documents": docs,
        "payout_amount_inr": inflated,
        "expected_fraud_signals": signals,
        "linked_claims": [],
    }


def _build_medical_inflation(rng: random.Random, claimant: Dict, coverage: str, ss: float) -> Dict:
    actual = _base_payout("health", rng)
    claimed = round(actual * rng.uniform(2.0, 4.5), -3)
    hospital = rng.choice(_HOSPITALS)
    date = _incident_date(rng)
    real_proc = rng.choice(["appendectomy", "knee arthroscopy", "cataract surgery"])
    fake_proc = rng.choice(["cardiac bypass", "spinal fusion", "liver transplant"])
    inflated = ss > 0.4

    docs = [
        {
            "doc_id": "DOC-001", "doc_type": "discharge_summary",
            "content": (
                f"Patient {claimant['name']} admitted {date}. "
                f"Procedure: {fake_proc if inflated else real_proc}. Hospital: {hospital}."
            ),
            "is_tampered": inflated,
            "tamper_signal": "procedure_mismatch" if inflated else None,
        },
        {
            "doc_id": "DOC-002", "doc_type": "hospital_bill",
            "content": f"Total bill: Rs {claimed:,.0f}. ICU: Rs {claimed*0.4:,.0f}. Procedure: Rs {claimed*0.5:,.0f}.",
            "is_tampered": ss > 0.6,
            "tamper_signal": "billing_code_mismatch" if ss > 0.6 else None,
        },
        {
            "doc_id": "DOC-003", "doc_type": "prescription",
            "content": (
                f"Post-procedure medication for {real_proc}. "
                f"{'Inconsistent with discharge summary procedure.' if inflated else 'As prescribed.'}"
            ),
            "is_tampered": inflated,
            "tamper_signal": "prescription_procedure_mismatch" if inflated else None,
        },
    ]

    signals = []
    if inflated:
        signals.append("procedure_mismatch")
    if ss > 0.6:
        signals.append("billing_code_mismatch")
    if ss > 0.8:
        signals.append("hospital_no_record")

    return {
        "incident": {
            "date": date, "type": "medical_procedure",
            "location": hospital,
            "description": f"Hospitalisation claim for {fake_proc if inflated else real_proc}.",
            "claimed_amount_inr": claimed,
        },
        "documents": docs,
        "payout_amount_inr": claimed,
        "expected_fraud_signals": signals,
        "linked_claims": [],
    }


def _build_identity_fraud(rng: random.Random, claimant: Dict, coverage: str, ss: float) -> Dict:
    date = _incident_date(rng)
    payout = _base_payout(coverage, rng)
    age_delta = rng.randint(8, 25)

    docs = [
        {
            "doc_id": "DOC-001", "doc_type": "identity_proof",
            "content": (
                f"Aadhaar: {rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}. "
                f"Name: {claimant['name']}. DOB mismatch: recorded age {claimant['age']}, Aadhaar age {claimant['age']+age_delta}."
            ),
            "is_tampered": ss > 0.5,
            "tamper_signal": "identity_mismatch" if ss > 0.5 else None,
        },
        {
            "doc_id": "DOC-002", "doc_type": "policy_document",
            "content": f"Policy {claimant['policy_number']} issued 5 days before incident. Claimant age discrepancy noted.",
            "is_tampered": True,
            "tamper_signal": "recent_policy_purchase",
        },
        {
            "doc_id": "DOC-003", "doc_type": "hospital_admission",
            "content": f"{'No record of admission for this Aadhaar.' if ss > 0.4 else 'Admission confirmed.'} Hospital: {rng.choice(_HOSPITALS)}.",
            "is_tampered": ss > 0.4,
            "tamper_signal": "hospital_no_record" if ss > 0.4 else None,
        },
    ]

    signals = ["identity_mismatch", "recent_policy_purchase"]
    if ss > 0.4:
        signals.append("hospital_no_record")
    if ss > 0.7:
        signals.append("dob_inconsistency")

    return {
        "incident": {
            "date": date, "type": "identity_verified_claim",
            "location": claimant["jurisdiction"],
            "description": "Claim filed under suspected ghost identity.",
            "claimed_amount_inr": payout,
        },
        "documents": docs,
        "payout_amount_inr": payout,
        "expected_fraud_signals": signals,
        "linked_claims": [],
    }


def _build_coordinated_ring(rng: random.Random, claimant: Dict, coverage: str, ss: float) -> Dict:
    date = _incident_date(rng)
    payout = _base_payout(coverage, rng)
    broker = f"BRK-{rng.randint(1000, 9999)}"

    linked = [
        {
            "claim_id": f"CLM-RING-{rng.randint(10000,99999)}",
            "claimant_name": f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
            "policy_number": f"POL-{claimant['jurisdiction']}-{rng.randint(100000,999999)}",
            "amount_inr": round(payout * rng.uniform(0.7, 1.3), -3),
            "broker_code": broker,
            "incident_date": date,
            "fraud_signal": "clustered_policy_broker" if ss > 0.3 else None,
        }
        for _ in range(rng.randint(3, 5))
    ]

    docs = [
        {
            "doc_id": "DOC-001", "doc_type": "claim_form",
            "content": f"Claim filed {date}. Amount: Rs {payout:,.0f}. Broker: {broker}.",
            "is_tampered": False, "tamper_signal": None,
        },
        {
            "doc_id": "DOC-002", "doc_type": "policy_document",
            "content": f"Policy {claimant['policy_number']}. Broker: {broker}. Same broker across multiple simultaneous claims.",
            "is_tampered": ss > 0.4,
            "tamper_signal": "clustered_policy_broker" if ss > 0.4 else None,
        },
    ]

    signals = []
    if ss > 0.3:
        signals.append("clustered_policy_broker")
    if ss > 0.5:
        signals.append("coordinated_incident_timing")
    if ss > 0.7:
        signals.append("shared_witness_across_claims")

    return {
        "incident": {
            "date": date, "type": "coordinated_fraud_ring",
            "location": claimant["jurisdiction"],
            "description": f"Claim linked to fraud ring via broker {broker}.",
            "claimed_amount_inr": payout,
        },
        "documents": docs,
        "payout_amount_inr": payout,
        "expected_fraud_signals": signals,
        "linked_claims": linked,
    }


def _build_phantom_provider(rng: random.Random, claimant: Dict, coverage: str, ss: float) -> Dict:
    date = _incident_date(rng)
    payout = _base_payout("health", rng)
    fake_hospital = f"Sri {rng.choice(_LAST_NAMES)} Medical Centre"

    docs = [
        {
            "doc_id": "DOC-001", "doc_type": "discharge_summary",
            "content": f"Discharged from {fake_hospital if ss > 0.4 else rng.choice(_HOSPITALS)}. Date: {date}.",
            "is_tampered": ss > 0.4,
            "tamper_signal": "unregistered_provider" if ss > 0.4 else None,
        },
        {
            "doc_id": "DOC-002", "doc_type": "hospital_registration",
            "content": f"{'Hospital not found in IRDAI registry.' if ss > 0.5 else 'Registered provider.'} GST: {'INVALID' if ss > 0.6 else 'VALID'}.",
            "is_tampered": ss > 0.5,
            "tamper_signal": "invalid_gst_registration" if ss > 0.6 else None,
        },
        {
            "doc_id": "DOC-003", "doc_type": "receipt",
            "content": f"Payment Rs {payout:,.0f}. {'No bank transfer record found.' if ss > 0.55 else 'Bank transfer confirmed.'}",
            "is_tampered": ss > 0.55,
            "tamper_signal": "no_payment_trail" if ss > 0.55 else None,
        },
    ]

    signals = []
    if ss > 0.4:
        signals.append("unregistered_provider")
    if ss > 0.5:
        signals.append("invalid_gst_registration")
    if ss > 0.55:
        signals.append("no_payment_trail")
    if ss > 0.8:
        signals.append("cloned_discharge_template")

    return {
        "incident": {
            "date": date, "type": "phantom_provider_claim",
            "location": claimant["jurisdiction"],
            "description": f"Medical claim from provider {fake_hospital} — registration unverifiable.",
            "claimed_amount_inr": payout,
        },
        "documents": docs,
        "payout_amount_inr": payout,
        "expected_fraud_signals": signals,
        "linked_claims": [],
    }


def _build_clean_claim(rng: random.Random, claimant: Dict, coverage: str, ss: float) -> Dict:
    date = _incident_date(rng)
    payout = _base_payout(coverage, rng)
    return {
        "incident": {
            "date": date, "type": f"{coverage}_claim",
            "location": claimant["jurisdiction"],
            "description": "Legitimate claim with all documents in order.",
            "claimed_amount_inr": payout,
        },
        "documents": [
            {
                "doc_id": "DOC-001", "doc_type": "claim_form",
                "content": f"Claim filed {date}. Amount: Rs {payout:,.0f}. Coverage: {coverage}.",
                "is_tampered": False, "tamper_signal": None,
            },
            {
                "doc_id": "DOC-002", "doc_type": "supporting_document",
                "content": f"All documents verified. Policy active since {claimant['policy_start_date']}.",
                "is_tampered": False, "tamper_signal": None,
            },
        ],
        "payout_amount_inr": payout,
        "expected_fraud_signals": [],
        "linked_claims": [],
    }


# ─────────────────────────────────────────────────────────────
# ACTION + TASK MAPPINGS
# ─────────────────────────────────────────────────────────────

_BASE_ACTIONS = [
    "validate_document", "flag_fraud_signal", "request_information",
    "query_historical_data", "estimate_payout",
    "approve_claim", "deny_claim", "escalate_to_human",
]

_EXTRA_ACTIONS: Dict[str, List[str]] = {
    "coordinated_ring":  ["query_linked_claim"],
    "identity_fraud":    ["verify_identity"],
    "phantom_provider":  ["verify_provider_registration"],
    "staged_accident":   [],
    "medical_inflation": [],
    "none":              [],
}

_TASK_ID_MAP: Dict[str, str] = {
    "none":              "clean_claim",
    "medical_inflation": "contradictory_claim",
    "staged_accident":   "contradictory_claim",
    "identity_fraud":    "contradictory_claim",
    "coordinated_ring":  "distribution_shift_claim",
    "phantom_provider":  "distribution_shift_claim",
}

_MAX_STEPS: Dict[str, int] = {"easy": 10, "medium": 18, "hard": 28}

_BUILDERS = {
    "staged_accident":   _build_staged_accident,
    "medical_inflation": _build_medical_inflation,
    "identity_fraud":    _build_identity_fraud,
    "coordinated_ring":  _build_coordinated_ring,
    "phantom_provider":  _build_phantom_provider,
    "none":              _build_clean_claim,
}


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def generate_claim(
    seed: int,
    fraud_type: str = "medical_inflation",
    coverage_type: str = "health",
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    jurisdiction: Optional[str] = None,
) -> ClaimScenario:
    """
    Generate a deterministic insurance claim episode.

    Same (seed, fraud_type, coverage_type, difficulty) always returns the same episode.
    Vary seed across [0, 9999] for 500+ unique training episodes per combination.
    """
    if fraud_type not in FRAUD_TYPES + ["none"]:
        raise ValueError(f"Invalid fraud_type '{fraud_type}'. Choose from {FRAUD_TYPES + ['none']}")
    if coverage_type not in COVERAGE_TYPES:
        raise ValueError(f"Invalid coverage_type '{coverage_type}'. Choose from {COVERAGE_TYPES}")
    if difficulty not in _MAX_STEPS:
        raise ValueError(f"Invalid difficulty '{difficulty}'. Choose from easy, medium, hard")

    rng = random.Random(seed)
    jur = jurisdiction or rng.choice(JURISDICTIONS)
    ss = DIFFICULTY_SIGNAL_STRENGTH[difficulty] * rng.uniform(0.85, 1.0)
    ambiguity = float(max(0.0, min(1.0, DIFFICULTY_AMBIGUITY[difficulty] * rng.uniform(0.9, 1.1))))

    claimant = _make_claimant(rng, jur)
    episode = _BUILDERS[fraud_type](rng, claimant, coverage_type, ss)

    return ClaimScenario(
        claim_id=f"CLM-{seed:04d}-{fraud_type[:3].upper()}-{jur}",
        seed=seed,
        fraud_type=fraud_type,
        coverage_type=coverage_type,
        jurisdiction=jur,
        difficulty=difficulty,
        claimant=claimant,
        incident=episode["incident"],
        documents=episode["documents"],
        ground_truth=FRAUD_GROUND_TRUTH[fraud_type],
        ambiguity_score=ambiguity,
        payout_amount_inr=episode["payout_amount_inr"],
        expected_fraud_signals=episode["expected_fraud_signals"],
        linked_claims=episode.get("linked_claims", []),
        available_actions=_BASE_ACTIONS + _EXTRA_ACTIONS.get(fraud_type, []),
        max_steps=_MAX_STEPS[difficulty],
        task_id=_TASK_ID_MAP.get(fraud_type, "contradictory_claim"),
    )


def generate_episode_pool(
    count: int = 500,
    fraud_types: Optional[List[str]] = None,
    coverage_types: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
) -> List[ClaimScenario]:
    """Generate a pool of training episodes across all fraud/coverage/difficulty combinations."""
    fraud_types = fraud_types or FRAUD_TYPES
    coverage_types = coverage_types or COVERAGE_TYPES
    difficulties = difficulties or list(_MAX_STEPS.keys())

    episodes: List[ClaimScenario] = []
    seed = 0
    while len(episodes) < count:
        for ft in fraud_types:
            for ct in coverage_types:
                for diff in difficulties:
                    if len(episodes) >= count:
                        break
                    episodes.append(generate_claim(seed, ft, ct, diff))
                    seed += 1
    return episodes
