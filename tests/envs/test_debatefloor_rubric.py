"""
tests/envs/test_debatefloor_rubric.py

Verifies the DebateFloorRubric contract after the FATAL-5 rewrite:

  1. The environment exposes the DebateFloorRubric on `env.rubric`.
  2. Every step exposes a rubric reward in [0, 1] and the canonical
     8-key component dict on `observation.rubric_components`.
  3. The rubric is INDEPENDENT of the environment reward — its value is
     allowed to (and routinely does) diverge from `obs.reward`. This is
     the AR-2 contract from HACKATHON_CONSTRAINTS.md and what FATAL-5 fixed.
  4. The reasoning_quality sub-rubric is sensitive to the action's
     reasoning text — empty reasoning yields 0.0, evidence-rich reasoning
     yields a positive score.

NOTE: This file replaces the previous test that asserted
`obs.rubric_reward == obs.reward`, which would re-introduce FATAL-5.
"""
from __future__ import annotations

import pytest

from app.environment import InsuranceClaimEnvironment
from app.models import InsuranceClaimAction
from app.rubrics import DebateFloorRubric


# Canonical component-key set produced by app.rubrics.DebateFloorRubric
# .component_scores() — kept in lockstep with that method.
EXPECTED_COMPONENT_KEYS = {
    "fraud_detection",
    "decision_accuracy",
    "calibration_score",
    "evidence_quality_score",
    "efficiency_score",
    "reasoning_quality",  # independent process signal added by FATAL-5 fix
    "penalty",
    "total",
}


def test_environment_uses_debatefloor_rubric() -> None:
    env = InsuranceClaimEnvironment()
    assert isinstance(env.rubric, DebateFloorRubric)


def test_rubric_components_are_exposed_on_step() -> None:
    env = InsuranceClaimEnvironment()
    env.reset(task_id="contradictory_claim", seed=42)

    obs = env.step(
        InsuranceClaimAction(
            action_type="deny_claim",
            confidence="MED",
            parameters={"reason": "date mismatch confirmed across documents"},
            reasoning=(
                "Date mismatch and cost inflation found across documents — "
                "clear fraud signals on the hospital bill and admission record."
            ),
        )
    )

    assert 0.0 <= obs.rubric_reward <= 1.0
    assert set(obs.rubric_components) == EXPECTED_COMPONENT_KEYS

    # `total` field equals the rubric_reward exposed at the top level
    assert obs.rubric_components["total"] == pytest.approx(obs.rubric_reward)

    # rubric_components is also mirrored on observation.metadata for clients
    # that read the legacy field
    assert obs.metadata.get("rubric_components") == obs.rubric_components


def test_rubric_diverges_from_env_reward() -> None:
    """FATAL-5 contract: independent rubric MUST be able to disagree with env reward.

    The previous test asserted equality, which silently masked FATAL-5. This
    test asserts the opposite: a divergence on at least one realistic action.
    """
    env = InsuranceClaimEnvironment()
    env.reset(task_id="contradictory_claim", seed=42)

    # Identical action to the original (broken) test — this is exactly the
    # call that the old `obs.rubric_reward == obs.reward` assertion failed on.
    obs = env.step(
        InsuranceClaimAction(
            action_type="deny_claim",
            confidence="MED",
            parameters={},
            reasoning="validation check",
        )
    )

    # Both must be valid floats in [0, 1] independently
    assert 0.0 <= obs.reward <= 1.0
    assert 0.0 <= obs.rubric_reward <= 1.0

    # The two MUST be allowed to differ. We assert strict inequality here
    # because for this specific action they currently differ by a margin
    # well above floating-point noise.
    assert obs.rubric_reward != pytest.approx(obs.reward, abs=1e-3), (
        "rubric_reward equals env reward — the rubric has stopped being "
        "independent (FATAL-5 regression)."
    )


def test_reasoning_quality_zero_for_empty_reasoning() -> None:
    """Empty/short reasoning must score reasoning_quality = 0.0."""
    env = InsuranceClaimEnvironment()
    env.reset(task_id="contradictory_claim", seed=42)

    obs = env.step(
        InsuranceClaimAction(
            action_type="deny_claim",
            confidence="MED",
            parameters={"reason": ""},
            reasoning="",  # below the 20-char threshold in _ReasoningQualityRubric
        )
    )

    assert obs.rubric_components["reasoning_quality"] == 0.0


def test_reasoning_quality_positive_for_evidence_rich_reasoning() -> None:
    """Evidence-keyword-rich reasoning must score reasoning_quality > 0."""
    env = InsuranceClaimEnvironment()
    env.reset(task_id="contradictory_claim", seed=42)

    obs = env.step(
        InsuranceClaimAction(
            action_type="deny_claim",
            confidence="MED",
            parameters={"reason": "fraud signals confirmed"},
            # Contains: date, mismatch, document, claim, fraud, hospital,
            # bill, evidence, inconsistency  → well above the 4-keyword
            # threshold for full score.
            reasoning=(
                "Date mismatch detected on the hospital bill versus the "
                "admission document. Inconsistency between procedure code "
                "and billed amount is clear evidence of fraud on this claim."
            ),
        )
    )

    assert obs.rubric_components["reasoning_quality"] > 0.0
    assert obs.rubric_components["reasoning_quality"] <= 1.0


def test_rubric_components_present_on_intermediate_steps() -> None:
    """Rubric must fire on every step, not only terminal ones."""
    env = InsuranceClaimEnvironment()
    env.reset(task_id="contradictory_claim", seed=42)

    obs = env.step(
        InsuranceClaimAction(
            action_type="validate_document",
            parameters={"doc_id": "DOC-10"},
            reasoning="Verifying claim form for date inconsistency evidence.",
        )
    )

    assert set(obs.rubric_components) == EXPECTED_COMPONENT_KEYS
    assert 0.0 <= obs.rubric_reward <= 1.0
