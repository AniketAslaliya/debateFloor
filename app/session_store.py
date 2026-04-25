"""
app/session_store.py — Global confidence history store for anti-gaming detection.

Moved here (instead of app/main.py) to avoid circular imports between
app/main.py and app/environment.py.
"""
from __future__ import annotations

from collections import deque
from threading import Lock

_global_confidence_history: deque = deque(maxlen=500)  # last 500 episodes, all sessions
_confidence_history_lock = Lock()


def record_episode_confidence(confidence: str) -> list[dict]:
    """Thread-safe append to global confidence history.

    Returns a snapshot of the current history for gaming detection.
    This is called from environment.py after every terminal action.
    """
    with _confidence_history_lock:
        _global_confidence_history.append({"confidence": confidence})
        return list(_global_confidence_history)


def get_confidence_distribution() -> dict:
    """Return current confidence distribution across all sessions."""
    with _confidence_history_lock:
        history = list(_global_confidence_history)
    total = len(history)
    if total == 0:
        return {"episodes_recorded": 0, "distribution": {}}
    return {
        "episodes_recorded": total,
        "distribution": {
            "HIGH": round(sum(1 for e in history if e["confidence"] == "HIGH") / total, 3),
            "MED":  round(sum(1 for e in history if e["confidence"] == "MED")  / total, 3),
            "LOW":  round(sum(1 for e in history if e["confidence"] == "LOW")  / total, 3),
        },
        "gaming_detection_active": total >= 10,
    }
