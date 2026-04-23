from __future__ import annotations

import pytest

from app.environment import InsuranceClaimEnvironment
from app.models import InsuranceClaimAction
from app.rubrics import DebateFloorRubric


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
            parameters={},
            reasoning="validation check",
        )
    )

    assert obs.rubric_reward == pytest.approx(obs.reward)
    assert set(obs.rubric_components) == {
        "fraud_detection",
        "decision_accuracy",
        "payout_accuracy",
        "efficiency_score",
        "consistency_score",
        "evidence_quality_score",
        "calibration_score",
        "penalty",
        "total",
    }
    assert obs.rubric_components["total"] == pytest.approx(obs.rubric_reward)