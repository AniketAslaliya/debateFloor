"""
tests/test_calibration.py
Tests for the calibration grader — run with: pytest tests/test_calibration.py -v
ALL tests must pass before pushing to GitHub.
"""

import pytest
from server.calibration_grader import (
    calibration_reward,
    detect_confidence_gaming,
    training_reward,
    eval_reward,
    CALIBRATION_MATRIX,
)


class TestCalibrationMatrix:
    """Test the core 3×2 calibration matrix values."""

    def test_high_correct_returns_1_point_0(self):
        """Confident and right = maximum reward."""
        result = calibration_reward("approve_claim", "HIGH", "approve_claim")
        assert result == 1.0

    def test_high_wrong_returns_minus_0_point_8(self):
        """Confident and wrong = maximum penalty. This is the KEY design decision."""
        result = calibration_reward("approve_claim", "HIGH", "deny_claim")
        assert result == -0.8

    def test_med_correct_returns_0_point_6(self):
        result = calibration_reward("deny_claim", "MED", "deny_claim")
        assert result == 0.6

    def test_med_wrong_returns_minus_0_point_2(self):
        result = calibration_reward("approve_claim", "MED", "deny_claim")
        assert result == -0.2

    def test_low_correct_returns_0_point_1(self):
        result = calibration_reward("escalate_to_human", "LOW", "escalate_to_human")
        assert result == 0.1

    def test_low_wrong_returns_0_point_0(self):
        """Very uncertain and wrong = neutral. At least it knew it didn't know."""
        result = calibration_reward("approve_claim", "LOW", "deny_claim")
        assert result == 0.0

    @pytest.mark.parametrize("confidence,correct,expected", [
        ("HIGH", True,  1.0),
        ("HIGH", False, -0.8),
        ("MED",  True,  0.6),
        ("MED",  False, -0.2),
        ("LOW",  True,  0.1),
        ("LOW",  False, 0.0),
    ])
    def test_all_matrix_values(self, confidence, correct, expected):
        """Parametrised test covering entire matrix."""
        decision = "approve_claim"
        ground_truth = "approve_claim" if correct else "deny_claim"
        result = calibration_reward(decision, confidence, ground_truth)
        assert result == expected

    def test_all_outputs_in_valid_range(self):
        """All calibration reward outputs must be in [-1.0, 1.0]."""
        decisions = ["approve_claim", "deny_claim", "escalate_to_human"]
        confidences = ["HIGH", "MED", "LOW"]
        for d in decisions:
            for c in confidences:
                result = calibration_reward(d, c, "approve_claim")
                assert -1.0 <= result <= 1.0, f"Out of range: {d}, {c} → {result}"

    def test_invalid_confidence_raises_error(self):
        with pytest.raises(ValueError):
            calibration_reward("approve_claim", "VERY_HIGH", "approve_claim")


class TestAntiGaming:
    """Test that systematic confidence manipulation is detected and penalised."""

    def test_systematic_low_triggers_penalty(self):
        """Agent always saying LOW = gaming detected after 10+ episodes."""
        history = [{"confidence": "LOW"}] * 15  # 100% LOW rate
        penalty = detect_confidence_gaming(history)
        assert penalty > 0, "Systematic LOW confidence should trigger gaming penalty"

    def test_systematic_high_triggers_penalty(self):
        """Agent always saying HIGH = overconfidence detected."""
        history = [{"confidence": "HIGH"}] * 15  # 100% HIGH rate
        penalty = detect_confidence_gaming(history)
        assert penalty > 0, "Systematic HIGH confidence should trigger penalty"

    def test_normal_distribution_no_penalty(self):
        """Mixed confidence = no gaming penalty."""
        history = (
            [{"confidence": "HIGH"}] * 5 +
            [{"confidence": "MED"}] * 5 +
            [{"confidence": "LOW"}] * 5
        )  # 33% each
        penalty = detect_confidence_gaming(history)
        assert penalty == 0.0

    def test_gaming_detector_needs_10_episodes_minimum(self):
        """Gaming detector doesn't fire with fewer than 10 episodes."""
        history = [{"confidence": "LOW"}] * 9  # Only 9 episodes
        penalty = detect_confidence_gaming(history)
        assert penalty == 0.0, "Should not fire with < 10 episodes"

    def test_gaming_penalty_applied_to_calibration_reward(self):
        """Gaming penalty is subtracted from calibration reward."""
        no_history = []
        gaming_history = [{"confidence": "LOW"}] * 20

        reward_no_gaming = calibration_reward("approve_claim", "LOW", "deny_claim", no_history)
        reward_with_gaming = calibration_reward("approve_claim", "LOW", "deny_claim", gaming_history)

        assert reward_with_gaming < reward_no_gaming, "Gaming penalty must reduce reward"

    def test_gaming_penalty_capped_at_1_point_0(self):
        """Gaming penalty should never exceed 1.0."""
        history = [{"confidence": "LOW"}] * 1000  # Extreme case
        penalty = detect_confidence_gaming(history)
        assert penalty <= 1.0


class TestTrainingReward:
    """Test the simple training reward used for GRPO."""

    def test_step_penalty_always_applied(self):
        """Every step should have -0.05 step penalty."""
        result = training_reward("approve_claim", "HIGH", "approve_claim", 0, 1, False)
        assert result == pytest.approx(-0.05)

    def test_correct_decision_adds_1_point_0(self):
        result = training_reward("approve_claim", "HIGH", "approve_claim", 0, 1, True)
        assert result > 0  # Should be positive overall

    def test_wrong_decision_subtracts_0_point_5(self):
        result_correct = training_reward("deny_claim", "MED", "deny_claim", 0, 1, True)
        result_wrong = training_reward("approve_claim", "MED", "deny_claim", 0, 1, True)
        assert result_correct > result_wrong

    def test_legitimate_flags_add_bonus(self):
        result_no_flags = training_reward("deny_claim", "MED", "deny_claim", 0, 1, True)
        result_with_flags = training_reward("deny_claim", "MED", "deny_claim", 2, 1, True)
        assert result_with_flags > result_no_flags

    def test_legitimate_flags_capped_at_3(self):
        result_3_flags = training_reward("deny_claim", "MED", "deny_claim", 3, 1, True)
        result_10_flags = training_reward("deny_claim", "MED", "deny_claim", 10, 1, True)
        assert result_3_flags == result_10_flags, "Flags should be capped at 3"

    def test_calibration_bonus_applied(self):
        """HIGH+CORRECT should give higher training reward than LOW+CORRECT."""
        result_high = training_reward("approve_claim", "HIGH", "approve_claim", 0, 1, True)
        result_low = training_reward("approve_claim", "LOW", "approve_claim", 0, 1, True)
        assert result_high > result_low


class TestEvalReward:
    """Test the full evaluation reward (for reporting only, not training)."""

    def test_eval_reward_in_0_to_1_range(self):
        """Evaluation reward must be in [0.0, 1.0] for consistent reporting."""
        result = eval_reward(
            decision="deny_claim",
            confidence="MED",
            ground_truth="deny_claim",
            ambiguity_score=0.5,
            evidence_quality=0.8,
            efficiency_score=0.7,
        )
        assert 0.0 <= result <= 1.0

    def test_high_confidence_correct_scores_highly(self):
        result = eval_reward(
            decision="approve_claim",
            confidence="HIGH",
            ground_truth="approve_claim",
            ambiguity_score=0.1,
            evidence_quality=0.9,
            efficiency_score=0.9,
        )
        assert result > 0.7, "Perfect calibrated decision should score > 0.7"

    def test_high_confidence_wrong_scores_low(self):
        result = eval_reward(
            decision="approve_claim",
            confidence="HIGH",
            ground_truth="deny_claim",
            ambiguity_score=0.1,
            evidence_quality=0.2,
            efficiency_score=0.5,
        )
        assert result < 0.4, "Overconfident wrong decision should score < 0.4"