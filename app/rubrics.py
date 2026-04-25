"""
app/rubrics.py — DebateFloor composable reward rubric.

The DebateFloorRubric composes two types of signals:
  1. Environment-derived components (from reward_breakdown) — outcome-based
  2. An independent ReasoningQualityRubric — process-based, can disagree with env reward

This separation ensures rubric_reward != env reward, satisfying the OpenEnv
rubric architecture requirement for independent evaluation signals.
"""
from __future__ import annotations

from typing import Any, Dict

from openenv.core.rubrics import Rubric


class _RewardFieldRubric(Rubric):
    """Reads a named field from observation.reward_breakdown (env-derived)."""

    def __init__(self, field_name: str):
        super().__init__()
        self.field_name = field_name

    def forward(self, action: Any, observation: Any) -> float:
        reward_breakdown = getattr(observation, "reward_breakdown", None)
        if reward_breakdown is None:
            return 0.0
        value = getattr(reward_breakdown, self.field_name, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class _PenaltyRubric(Rubric):
    def forward(self, action: Any, observation: Any) -> float:
        reward_breakdown = getattr(observation, "reward_breakdown", None)
        if reward_breakdown is None:
            return 0.0
        value = getattr(reward_breakdown, "penalty", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class _ReasoningQualityRubric(Rubric):
    """
    INDEPENDENT of environment reward — evaluates reasoning process quality.

    Scores whether the agent's reasoning text references specific evidence keywords.
    This fires on every step, providing a dense process signal the env reward lacks.
    It can disagree with the env reward (e.g., agent got lucky with right answer
    but provided no reasoning — penalised here even if env rewards the correct decision).
    """

    EVIDENCE_KEYWORDS = [
        "date", "mismatch", "document", "inconsistency", "signal", "evidence",
        "policy", "hospital", "bill", "procedure", "claim", "fraud", "verified",
        "mismatch", "tampered", "inflated", "discrepancy", "suspicious", "record",
    ]

    def forward(self, action: Any, observation: Any) -> float:
        reasoning = getattr(action, "reasoning", "") or ""
        if len(reasoning) < 20:
            return 0.0  # too short to be meaningful
        reasoning_lc = reasoning.lower()
        hits = sum(1 for kw in self.EVIDENCE_KEYWORDS if kw in reasoning_lc)
        return min(1.0, hits / 4.0)  # 4 keywords = full score


class DebateFloorRubric(Rubric):
    """
    Composable reward rubric for DebateFloor.

    Combines env-derived outcome signals (fraud_detection, calibration) with an
    independent process signal (reasoning_quality) that evaluates HOW the agent
    reasons, not just WHAT it decided. This ensures rubric_reward != env reward.
    """

    def __init__(self):
        super().__init__()
        # Env-derived components (outcome-based)
        self.fraud_detection = _RewardFieldRubric("fraud_detection_score")
        self.decision_accuracy = _RewardFieldRubric("decision_accuracy")
        self.calibration_score = _RewardFieldRubric("calibration_score")
        self.evidence_quality_score = _RewardFieldRubric("evidence_quality_score")
        self.efficiency_score = _RewardFieldRubric("efficiency_score")
        self.penalty = _PenaltyRubric()
        # Independent process signal — can disagree with env reward
        self.reasoning_quality = _ReasoningQualityRubric()

        self._weights: Dict[str, float] = {
            "fraud_detection":      0.25,
            "decision_accuracy":    0.20,
            "calibration_score":    0.20,
            "evidence_quality_score": 0.15,
            "efficiency_score":     0.00,  # kept for structure, zero-weighted
            "reasoning_quality":    0.20,  # independent signal
        }

    def forward(self, action: Any, observation: Any) -> float:
        component_scores = self._component_scores(action, observation)
        weighted = sum(
            self._weights[name] * component_scores[name] for name in self._weights
        )
        total = weighted - component_scores["penalty"]
        return round(max(0.0, min(1.0, total)), 4)

    def component_scores(self) -> Dict[str, float]:
        """Return the most recent component scores after a rubric pass."""
        return {
            "fraud_detection":      float(self.fraud_detection.last_score or 0.0),
            "decision_accuracy":    float(self.decision_accuracy.last_score or 0.0),
            "calibration_score":    float(self.calibration_score.last_score or 0.0),
            "evidence_quality_score": float(self.evidence_quality_score.last_score or 0.0),
            "efficiency_score":     float(self.efficiency_score.last_score or 0.0),
            "reasoning_quality":    float(self.reasoning_quality.last_score or 0.0),
            "penalty":              float(self.penalty.last_score or 0.0),
            "total":                float(self.last_score or 0.0),
        }

    def _component_scores(self, action: Any, observation: Any) -> Dict[str, float]:
        return {
            "fraud_detection":      self.fraud_detection(action, observation),
            "decision_accuracy":    self.decision_accuracy(action, observation),
            "calibration_score":    self.calibration_score(action, observation),
            "evidence_quality_score": self.evidence_quality_score(action, observation),
            "efficiency_score":     self.efficiency_score(action, observation),
            "reasoning_quality":    self.reasoning_quality(action, observation),
            "penalty":              self.penalty(action, observation),
        }