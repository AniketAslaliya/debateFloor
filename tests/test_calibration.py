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
        result = calibration_reward("approve_claim", "HIGH", "approve_claim")
        assert result == 1.0

    def test_high_wrong_returns_minus_0_point_8(self):
        result = calibration_reward("approve_claim", "HIGH", "deny_claim")
        assert result == -0.8

    def test_med_correct_returns_0_point_6(self):
        result = calibration_reward("deny_claim", "MED", "deny_claim")
        assert result == 0.6

    @pytest.mark.parametrize("confidence,correct,expected", [
        ("HIGH", True,  1.0),
        ("HIGH", False, -0.8),
        ("MED",  True,  0.6),
        ("MED",  False, -0.2),
        ("LOW",  True,  0.1),
        ("LOW",  False, 0.0),
    ])
    def test_all_outputs_in_valid_range(self, confidence, correct, expected):
        decision = "approve_claim"
        ground_truth = "approve_claim" if correct else "deny_claim"
        result = calibration_reward(decision, confidence, ground_truth)
        assert result == expected
        assert -1.0 <= result <= 1.0


class TestAntiGaming:

    def test_systematic_low_triggers_gaming_penalty(self):
        history = [{"confidence": "LOW"}] * 15
        penalty = detect_confidence_gaming(history)
        assert penalty > 0

    def test_systematic_high_triggers_gaming_penalty(self):
        history = [{"confidence": "HIGH"}] * 15
        penalty = detect_confidence_gaming(history)
        assert penalty > 0

    def test_gaming_detector_needs_10_episodes_minimum(self):
        history = [{"confidence": "LOW"}] * 9
        penalty = detect_confidence_gaming(history)
        assert penalty == 0.0


class TestTrainingReward:

    def test_training_reward_step_penalty_applied(self):
        result = training_reward("approve_claim", "HIGH", "approve_claim", 0, 1, False)
        assert result == pytest.approx(-0.05)
