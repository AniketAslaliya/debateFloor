"""
server/calibration_grader.py
DebateFloor — Calibrated Uncertainty Training Environment
Core innovation: rewards agents that know when they don't know.

Based on CoCA framework: arXiv:2603.05881
"Co-optimizing Confidence and Accuracy via Segment-Specific GRPO Rewards"

CRITICAL: This file implements the CALIBRATION reward only.
          The TRAINING reward (simple scalar) is also here.
          NEVER use eval_reward() for GRPO training — use training_reward().
"""

from typing import Optional

# ─────────────────────────────────────────────────────────────
# THE 3×2 CALIBRATION MATRIX
# This is the core innovation. Read this before editing anything.
#
# Philosophy:
#   HIGH confidence + CORRECT  = best outcome (1.0)  — decisive and right
#   HIGH confidence + WRONG    = worst outcome (-0.8) — confident and wrong
#   MED  confidence + CORRECT  = good (0.6)           — right but cautious
#   MED  confidence + WRONG    = ok (-0.2)             — wrong but knew it
#   LOW  confidence + CORRECT  = weak (0.1)            — right, wasted escalation
#   LOW  confidence + WRONG    = neutral (0.0)          — at least it knew
# ─────────────────────────────────────────────────────────────
CALIBRATION_MATRIX: dict[tuple[str, bool], float] = {
    ("HIGH", True):   1.0,
    ("HIGH", False): -0.8,
    ("MED",  True):   0.6,
    ("MED",  False): -0.2,
    ("LOW",  True):   0.1,
    ("LOW",  False):  0.0,
}

# Anti-gaming thresholds
LOW_CONFIDENCE_GAMING_THRESHOLD = 0.70   # >70% LOW = gaming
HIGH_CONFIDENCE_GAMING_THRESHOLD = 0.80  # >80% HIGH = overconfidence
MIN_HISTORY_FOR_GAMING_DETECTION = 10    # need at least 10 episodes


def detect_confidence_gaming(episode_history: list[dict]) -> float:
    """
    Detects and penalises systematic confidence manipulation.

    An agent cannot game the calibration reward by always declaring LOW
    confidence (to avoid HIGH+WRONG penalty) or always declaring HIGH
    confidence (to maximise HIGH+CORRECT reward).

    Args:
        episode_history: List of dicts with "confidence" key per episode.
                         Example: [{"confidence": "LOW"}, {"confidence": "HIGH"}, ...]

    Returns:
        float: Penalty to subtract from reward. Always >= 0.
               Returns 0.0 if history is too short to detect gaming.
    """
    if len(episode_history) < MIN_HISTORY_FOR_GAMING_DETECTION:
        return 0.0

    total = len(episode_history)
    low_count = sum(1 for e in episode_history if e.get("confidence") == "LOW")
    high_count = sum(1 for e in episode_history if e.get("confidence") == "HIGH")

    low_rate = low_count / total
    high_rate = high_count / total

    penalty = 0.0

    # Penalise systematic under-confidence (always say LOW to avoid punishment)
    if low_rate > LOW_CONFIDENCE_GAMING_THRESHOLD:
        penalty += (low_rate - LOW_CONFIDENCE_GAMING_THRESHOLD) * 2.0

    # Penalise systematic over-confidence (always say HIGH to maximise reward)
    if high_rate > HIGH_CONFIDENCE_GAMING_THRESHOLD:
        penalty += (high_rate - HIGH_CONFIDENCE_GAMING_THRESHOLD) * 1.5

    return min(penalty, 1.0)  # cap total penalty at 1.0


def calibration_reward(
    decision: str,
    confidence: str,
    ground_truth: str,
    episode_history: Optional[list[dict]] = None,
) -> float:
    """
    Core calibration reward. Used in EVALUATION reward composition.

    Args:
        decision:        Agent's decision ("approve_claim", "deny_claim", "escalate_to_human")
        confidence:      Agent's declared confidence ("HIGH", "MED", "LOW")
        ground_truth:    Correct decision for this episode
        episode_history: List of past episode results for gaming detection

    Returns:
        float: Calibration reward in [-1.0, 1.0]
    """
    if confidence not in ("HIGH", "MED", "LOW"):
        raise ValueError(f"Invalid confidence: {confidence}. Must be HIGH, MED, or LOW.")

    is_correct = (decision == ground_truth)
    base_reward = CALIBRATION_MATRIX[(confidence, is_correct)]

    # Apply anti-gaming penalty if we have enough history
    gaming_penalty = 0.0
    if episode_history:
        gaming_penalty = detect_confidence_gaming(episode_history)

    result = base_reward - gaming_penalty

    # Always clamp to valid range
    return max(-1.0, min(1.0, result))


def escalation_reward(
    decision: str,
    confidence: str,
    ambiguity_score: float,
) -> float:
    """
    Rewards appropriate escalation behaviour.

    An agent should escalate when genuinely uncertain (high ambiguity).
    Escalating on obvious cases wastes resources and is penalised.

    Args:
        decision:        Agent's decision
        confidence:      Agent's declared confidence
        ambiguity_score: How genuinely ambiguous this task is (0.0=obvious, 1.0=very ambiguous)

    Returns:
        float: Escalation reward in [-0.5, 0.7]
    """
    is_escalation = (decision == "escalate_to_human")
    is_genuinely_ambiguous = ambiguity_score > 0.6
    is_obviously_clear = ambiguity_score < 0.3

    if is_escalation and is_genuinely_ambiguous and confidence == "LOW":
        return 0.7   # Perfect: uncertain + ambiguous task + escalated
    elif is_escalation and is_obviously_clear:
        return -0.3  # Bad: escalated on an easy/obvious task
    elif is_escalation and confidence == "HIGH":
        return -0.2  # Bad: escalated but was confident (contradictory)
    else:
        return 0.0   # Neutral: didn't escalate


def training_reward(
    decision: str,
    confidence: Optional[str],
    ground_truth: str,
    legitimate_flags: int,
    step_num: int,
    done: bool,
) -> float:
    """
    SIMPLE shaped scalar reward for GRPO training stability.

    ⚠️  USE THIS FOR GRPO TRAINING — NOT eval_reward().
    Complex compound rewards cause gradient instability in GRPO.
    This function provides a clear, stable learning signal.

    Args:
        decision:         Agent's terminal decision (or None if non-terminal)
        confidence:       Agent's declared confidence (None for non-terminal steps)
        ground_truth:     Correct decision for this episode
        legitimate_flags: Number of correctly identified fraud signals this episode
        step_num:         Current step number
        done:             Whether episode is complete

    Returns:
        float: Training reward (negative at each step, positive signal on completion)
    """
    # Step penalty — encourages efficiency
    r = -0.05

    if done and decision is not None:
        is_correct = (decision == ground_truth)

        # Decision accuracy (main signal)
        r += 1.0 if is_correct else -0.5

        # Legitimate fraud signal detection (partial credit)
        r += 0.3 * min(legitimate_flags, 3)  # cap at 3 flags

        # Calibration bonus (weighted 50% of calibration matrix)
        if confidence and confidence in ("HIGH", "MED", "LOW"):
            calib_value = CALIBRATION_MATRIX.get((confidence, is_correct), 0.0)
            r += 0.5 * calib_value

    return float(r)


def eval_reward(
    decision: str,
    confidence: str,
    ground_truth: str,
    ambiguity_score: float,
    evidence_quality: float,
    efficiency_score: float,
    episode_history: Optional[list[dict]] = None,
) -> float:
    """
    FULL 6-component evaluation reward. Used for REPORTING and DEMO only.

    ⚠️  DO NOT USE FOR GRPO TRAINING. Use training_reward() instead.

    Components:
        35% calibration_reward    — confidence accuracy matrix
        25% escalation_reward     — appropriate uncertainty escalation
        20% evidence_quality      — specificity of fraud signal citations
        10% efficiency_score      — step efficiency (inherited from Round 1)
        10% gaming_penalty pool   — anti-gaming deductions

    Args:
        decision:        Agent's terminal decision
        confidence:      Agent's declared confidence
        ground_truth:    Correct decision
        ambiguity_score: Task ambiguity (0.0=obvious, 1.0=very ambiguous)
        evidence_quality: Quality of fraud signal evidence (0.0–1.0)
        efficiency_score: Step efficiency from environment (0.0–1.0)
        episode_history: For gaming detection

    Returns:
        float: Composite evaluation score in [0.0, 1.0]
    """
    calib_r = calibration_reward(decision, confidence, ground_truth, episode_history)
    escal_r = escalation_reward(decision, confidence, ambiguity_score)
    gaming_p = detect_confidence_gaming(episode_history) if episode_history else 0.0

    raw = (
        0.35 * calib_r +
        0.25 * escal_r +
        0.20 * evidence_quality +
        0.10 * efficiency_score -
        0.10 * gaming_p
    )

    # Normalise to [0.0, 1.0] for evaluation reporting
    # Raw range is approximately [-0.8, 1.0], shift and scale
    normalised = (raw + 0.8) / 1.8
    return max(0.0, min(1.0, normalised))