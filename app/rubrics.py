from __future__ import annotations

from typing import Any, Dict

from openenv.core.rubrics import Rubric


class _RewardFieldRubric(Rubric):
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


class DebateFloorRubric(Rubric):
    """Composable reward rubric for DebateFloor.

    The rubric keeps the existing reward mathematics, but breaks it into
    named components so training infrastructure can introspect and log the
    individual signals that shape the final score.
    """

    def __init__(self):
        super().__init__()
        self.fraud_detection = _RewardFieldRubric("fraud_detection_score")
        self.decision_accuracy = _RewardFieldRubric("decision_accuracy")
        self.payout_accuracy = _RewardFieldRubric("payout_accuracy")
        self.efficiency_score = _RewardFieldRubric("efficiency_score")
        self.consistency_score = _RewardFieldRubric("consistency_score")
        self.evidence_quality_score = _RewardFieldRubric("evidence_quality_score")
        self.calibration_score = _RewardFieldRubric("calibration_score")
        self.penalty = _PenaltyRubric()

        self._weights: Dict[str, float] = {
            "fraud_detection": 0.28,
            "decision_accuracy": 0.20,
            "payout_accuracy": 0.11,
            "efficiency_score": 0.10,
            "consistency_score": 0.09,
            "evidence_quality_score": 0.14,
            "calibration_score": 0.08,
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
            "fraud_detection": float(self.fraud_detection.last_score or 0.0),
            "decision_accuracy": float(self.decision_accuracy.last_score or 0.0),
            "payout_accuracy": float(self.payout_accuracy.last_score or 0.0),
            "efficiency_score": float(self.efficiency_score.last_score or 0.0),
            "consistency_score": float(self.consistency_score.last_score or 0.0),
            "evidence_quality_score": float(self.evidence_quality_score.last_score or 0.0),
            "calibration_score": float(self.calibration_score.last_score or 0.0),
            "penalty": float(self.penalty.last_score or 0.0),
            "total": float(self.last_score or 0.0),
        }

    def _component_scores(self, action: Any, observation: Any) -> Dict[str, float]:
        return {
            "fraud_detection": self.fraud_detection(action, observation),
            "decision_accuracy": self.decision_accuracy(action, observation),
            "payout_accuracy": self.payout_accuracy(action, observation),
            "efficiency_score": self.efficiency_score(action, observation),
            "consistency_score": self.consistency_score(action, observation),
            "evidence_quality_score": self.evidence_quality_score(action, observation),
            "calibration_score": self.calibration_score(action, observation),
            "penalty": self.penalty(action, observation),
        }