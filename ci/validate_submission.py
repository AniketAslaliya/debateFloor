"""Round-1 submission validator for local pre-flight checks.

This script mirrors the Scaler dashboard checklist as closely as possible
without requiring external network access or a deployed Hugging Face Space.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "inference.py"
OPENENV = ROOT / "openenv.yaml"
README = ROOT / "README.md"
DOCKERFILE = ROOT / "Dockerfile"
REQUIREMENTS = ROOT / "requirements.txt"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    for path in [INFERENCE, OPENENV, README, DOCKERFILE, REQUIREMENTS]:
        assert_true(path.exists(), f"Missing required file: {path.name}")

    inference_text = INFERENCE.read_text(encoding="utf-8")
    for marker in ["[START]", "[STEP]", "[END]"]:
        assert_true(marker in inference_text, f"inference.py is missing required log marker {marker}")
    for token in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN", "OpenAI("]:
        assert_true(token in inference_text, f"inference.py is missing required token '{token}'")

    openenv_text = OPENENV.read_text(encoding="utf-8")
    for required_key in ["spec_version", "name:", "runtime:", "app:", "tasks:", "action_space:", "observation_space:"]:
        assert_true(required_key in openenv_text, f"openenv.yaml is missing '{required_key}'")

    readme_text = README.read_text(encoding="utf-8")
    for required_section in [
        "## Problem Statement",
        "## Quickstart",
        "## Architecture",
        "## Evaluation",
        "## Dashboard Compliance Checklist",
    ]:
        assert_true(required_section in readme_text, f"README is missing '{required_section}'")

    client = TestClient(app)

    health = client.get("/health")
    assert_true(health.status_code == 200, "/health did not return 200")

    tasks = client.get("/tasks")
    assert_true(tasks.status_code == 200, "/tasks did not return 200")
    payload = tasks.json()
    assert_true(len(payload.get("tasks", [])) >= 4, "Expected at least 4 tasks in /tasks")

    schema = client.get("/schema")
    assert_true(schema.status_code == 200, "/schema did not return 200")
    schema_payload = schema.json()
    assert_true(all(k in schema_payload for k in ["action", "observation", "state"]), "Schema response is incomplete")

    reset = client.post("/reset", json={"task_id": "clean_claim", "seed": 42, "session_id": "validator-session"})
    assert_true(reset.status_code == 200, "/reset did not return 200")
    reset_payload = reset.json()
    assert_true(reset_payload.get("session_id") == "validator-session", "Reset did not respect session_id")
    assert_true(reset_payload.get("reward") == 0.0, "Step-0 reward must be 0.0")

    print("Submission validator PASS")


if __name__ == "__main__":
    main()
