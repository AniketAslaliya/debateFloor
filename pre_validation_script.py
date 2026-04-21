"""
pre_validation_script.py
DebateFloor — Pre-submission validation

Checks every mandatory requirement before pitching.
Run against a live server: python pre_validation_script.py [--base-url URL]

Exit code 0 = all green. Exit code 1 = failures found.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Tuple

import requests

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    line = f"  [{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


failures: List[str] = []


def run_check(label: str, ok: bool, detail: str = "") -> None:
    if not check(label, ok, detail):
        failures.append(label)


# ──────────────────────────────────────────────
# 1. Health
# ──────────────────────────────────────────────

def validate_health(base: str) -> None:
    print("\n[1] Health endpoint")
    try:
        r = requests.get(f"{base}/health", timeout=10)
        data = r.json()
        run_check("/health returns 200", r.status_code == 200)
        run_check("status == healthy", data.get("status") == "healthy", str(data.get("status")))
        run_check("environment field present", "environment" in data)
        run_check("active_sessions field present", "active_sessions" in data)
    except Exception as e:
        run_check("/health reachable", False, str(e))


# ──────────────────────────────────────────────
# 2. Schema
# ──────────────────────────────────────────────

def validate_schema(base: str) -> None:
    print("\n[2] Schema endpoint")
    try:
        r = requests.get(f"{base}/schema", timeout=10)
        run_check("/schema returns 200", r.status_code == 200)
        data = r.json()
        run_check("action schema present", "action" in data)
        run_check("observation schema present", "observation" in data)
        run_check("state schema present", "state" in data)
        # Check confidence field in action schema
        action_props = data.get("action", {}).get("properties", {})
        run_check("confidence field in action schema", "confidence" in action_props)
        run_check("action_type field in action schema", "action_type" in action_props)
    except Exception as e:
        run_check("/schema reachable", False, str(e))


# ──────────────────────────────────────────────
# 3. Tasks
# ──────────────────────────────────────────────

REQUIRED_TASKS = {"clean_claim", "contradictory_claim", "distribution_shift_claim"}


def validate_tasks(base: str) -> None:
    print("\n[3] Tasks endpoint")
    try:
        r = requests.get(f"{base}/tasks", timeout=10)
        run_check("/tasks returns 200", r.status_code == 200)
        data = r.json()
        task_ids = {t["task_id"] for t in data.get("tasks", [])}
        for tid in REQUIRED_TASKS:
            run_check(f"task '{tid}' registered", tid in task_ids)
    except Exception as e:
        run_check("/tasks reachable", False, str(e))


# ──────────────────────────────────────────────
# 4. Reset — all 3 tasks
# ──────────────────────────────────────────────

def validate_reset(base: str) -> Dict[str, str]:
    print("\n[4] Reset — all 3 tasks")
    session_ids: Dict[str, str] = {}
    for task_id in REQUIRED_TASKS:
        try:
            r = requests.post(
                f"{base}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=15,
            )
            run_check(f"reset '{task_id}' returns 200", r.status_code == 200, f"got {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                sid = data.get("session_id")
                session_ids[task_id] = sid
                obs = data.get("observation", {})
                run_check(f"'{task_id}' observation has claim_id", "claim_id" in obs)
                run_check(f"'{task_id}' confidence_required=True", obs.get("confidence_required") is True)
        except Exception as e:
            run_check(f"reset '{task_id}' reachable", False, str(e))
    return session_ids


# ──────────────────────────────────────────────
# 5. Step — terminal action with confidence
# ──────────────────────────────────────────────

TERMINAL_ACTIONS = {
    "clean_claim":              ("approve_claim", "HIGH"),
    "contradictory_claim":      ("deny_claim", "MED"),
    "distribution_shift_claim": ("escalate_to_human", "LOW"),
}


def validate_step(base: str, session_ids: Dict[str, str]) -> None:
    print("\n[5] Step — terminal actions with confidence")
    for task_id, (action_type, confidence) in TERMINAL_ACTIONS.items():
        sid = session_ids.get(task_id)
        if not sid:
            run_check(f"step '{task_id}' (no session)", False, "reset failed")
            continue
        try:
            r = requests.post(
                f"{base}/step",
                json={
                    "session_id": sid,
                    "action": {
                        "action_type": action_type,
                        "confidence": confidence,
                        "parameters": {},
                        "reasoning": "validation check",
                    },
                },
                timeout=15,
            )
            run_check(f"step '{task_id}' returns 200", r.status_code == 200, f"got {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                obs = data.get("observation", {})
                rb = obs.get("reward_breakdown", {})
                calib = rb.get("calibration_score")
                run_check(
                    f"'{task_id}' calibration_score populated on terminal",
                    calib is not None,
                    f"got {calib}",
                )
                run_check(f"'{task_id}' done=True after terminal", data.get("done") is True)
        except Exception as e:
            run_check(f"step '{task_id}' reachable", False, str(e))


# ──────────────────────────────────────────────
# 6. Calibration scores in valid range
# ──────────────────────────────────────────────

def validate_calibration(base: str) -> None:
    print("\n[6] Calibration score range [-1.0, 1.0]")
    # Quick reset + terminal step for each task
    cases = [
        ("clean_claim",              "approve_claim",     "HIGH", 43),
        ("contradictory_claim",      "deny_claim",        "MED",  44),
        ("distribution_shift_claim", "escalate_to_human", "LOW",  45),
    ]
    for task_id, action_type, confidence, seed in cases:
        try:
            r1 = requests.post(f"{base}/reset", json={"task_id": task_id, "seed": seed}, timeout=15)
            if r1.status_code != 200:
                run_check(f"calib range '{task_id}'", False, "reset failed")
                continue
            sid = r1.json().get("session_id")
            r2 = requests.post(
                f"{base}/step",
                json={"session_id": sid, "action": {
                    "action_type": action_type, "confidence": confidence,
                    "parameters": {}, "reasoning": "calib range check",
                }},
                timeout=15,
            )
            if r2.status_code != 200:
                run_check(f"calib range '{task_id}'", False, f"step {r2.status_code}")
                continue
            data = r2.json()
            rb = data.get("observation", {}).get("reward_breakdown", {})
            calib = rb.get("calibration_score")
            in_range = calib is not None and -1.0 <= calib <= 1.0
            run_check(
                f"calib score in [-1,1] for '{task_id}'",
                in_range,
                f"got {calib}",
            )
            total = rb.get("total", -1)
            run_check(
                f"total reward in [0,1] for '{task_id}'",
                0.0 <= total <= 1.0,
                f"got {total}",
            )
        except Exception as e:
            run_check(f"calib range '{task_id}'", False, str(e))


# ──────────────────────────────────────────────
# 7. Concurrent sessions
# ──────────────────────────────────────────────

def validate_concurrent_sessions(base: str) -> None:
    print("\n[7] Concurrent sessions (4 parallel resets)")
    import threading

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def do_reset(i: int) -> None:
        try:
            r = requests.post(
                f"{base}/reset",
                json={"task_id": "clean_claim", "seed": i},
                timeout=15,
            )
            with lock:
                results.append({"ok": r.status_code == 200, "sid": r.json().get("session_id")})
        except Exception as e:
            with lock:
                results.append({"ok": False, "sid": None, "err": str(e)})

    threads = [threading.Thread(target=do_reset, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)

    all_ok = all(r["ok"] for r in results)
    unique_sids = len({r["sid"] for r in results if r["sid"]})
    run_check("4 parallel resets all succeeded", all_ok)
    run_check("4 unique session IDs returned", unique_sids == 4, f"got {unique_sids}")


# ──────────────────────────────────────────────
# 8. Error handling — missing confidence on terminal
# ──────────────────────────────────────────────

def validate_error_handling(base: str) -> None:
    print("\n[8] Error handling — terminal action without confidence")
    try:
        r1 = requests.post(f"{base}/reset", json={"task_id": "clean_claim", "seed": 99}, timeout=15)
        sid = r1.json().get("session_id")
        r2 = requests.post(
            f"{base}/step",
            json={"session_id": sid, "action": {
                "action_type": "approve_claim",
                "parameters": {},
                "reasoning": "test",
                # confidence intentionally omitted
            }},
            timeout=15,
        )
        run_check(
            "missing confidence returns 422",
            r2.status_code == 422,
            f"got {r2.status_code}",
        )
    except Exception as e:
        run_check("error handling check", False, str(e))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="DebateFloor pre-submission validation")
    parser.add_argument("--base-url", default="http://localhost:7860")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"DebateFloor Pre-Validation")
    print(f"Target: {base}")
    print("=" * 50)

    validate_health(base)
    validate_schema(base)
    validate_tasks(base)
    session_ids = validate_reset(base)
    validate_step(base, session_ids)
    validate_calibration(base)
    validate_concurrent_sessions(base)
    validate_error_handling(base)

    print("\n" + "=" * 50)
    if failures:
        print(f"\033[91mFAILED — {len(failures)} check(s) failed:\033[0m")
        for f in failures:
            print(f"  x {f}")
        return 1
    else:
        print("\033[92mALL CHECKS PASSED — ready to pitch!\033[0m")
        return 0


if __name__ == "__main__":
    sys.exit(main())
