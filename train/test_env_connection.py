"""
test_env_connection.py — Validates that train_minimal.py is correctly
wired to call the live environment via HTTP.

This script verifies:
  1. reward_fn signature accepts **kwargs (not positional args like ground_truths)
  2. make_row() produces task_id and seed columns
  3. run_episode_via_http() makes actual HTTP POST calls
  4. _start_env_server_if_needed() raises when server is unreachable
  5. The word "no server" / "no HTTP" does NOT appear in the docstring

Usage:
  python train/test_env_connection.py
"""

import json
import os
import re
import sys
import inspect
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")

# Mock out heavy training-only dependencies that may not be installed locally
# We only need to test the HTTP wiring logic, not actual GPU training
for mod_name in ["wandb", "trl", "trl.GRPOConfig", "trl.GRPOTrainer",
                 "datasets", "datasets.Dataset", "unsloth"]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# torch may or may not be installed — mock if missing
try:
    import torch
except ImportError:
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.cuda"] = MagicMock()

# ── Test 1: Verify the module docstring says server IS required ─────────────
print("Test 1: Checking module docstring...")
with open("train/train_minimal.py", encoding="utf-8") as f:
    source = f.read()

# MUST NOT contain anti-patterns
forbidden = ["no server required", "no HTTP server", "no server needed", "direct-reward", "no-server"]
for phrase in forbidden:
    if phrase.lower() in source.lower():
        print(f"  ❌ FAIL: Found forbidden phrase '{phrase}' in train_minimal.py")
        sys.exit(1)

# MUST contain these indicators that it's env-connected
required = [
    "POST /step",
    "POST /reset",
    "/reset",
    "/step",
    "env-connected",
    "http-reward",
    "MR-2",
]
for phrase in required:
    if phrase not in source:
        print(f"  ❌ FAIL: Missing required phrase '{phrase}' in train_minimal.py")
        sys.exit(1)

print("  ✅ Module docstring correctly declares env-connected training")


# ── Test 2: make_row() includes task_id and seed ────────────────────────────
print("Test 2: Checking make_row() output columns...")
from server.claim_generator import generate_claim

class MockTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return json.dumps(messages)

# Import make_row
from train.train_minimal import make_row

ep = generate_claim(seed=42, fraud_type="medical_inflation", coverage_type="health", difficulty="medium")
tok = MockTokenizer()
row = make_row(ep, tok)

assert "task_id" in row, f"❌ FAIL: make_row() missing 'task_id'. Got keys: {list(row.keys())}"
assert "seed" in row, f"❌ FAIL: make_row() missing 'seed'. Got keys: {list(row.keys())}"
assert row["task_id"] == "contradictory_claim", f"❌ FAIL: task_id should be 'contradictory_claim', got '{row['task_id']}'"
assert row["seed"] == "42", f"❌ FAIL: seed should be '42' (str), got '{row['seed']}'"
print(f"  ✅ make_row() includes task_id='{row['task_id']}' and seed='{row['seed']}'")


# ── Test 3: reward_fn uses **kwargs (not positional) ────────────────────────
print("Test 3: Checking reward_fn signature...")
from train.train_minimal import reward_fn

sig = inspect.signature(reward_fn)
params = list(sig.parameters.keys())

# Must accept **kwargs
assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()), \
    f"❌ FAIL: reward_fn does not accept **kwargs. Params: {params}"

# Must NOT have 'expected_signals_list' as a positional param (old signature)
assert "expected_signals_list" not in params, \
    f"❌ FAIL: reward_fn still has 'expected_signals_list' positional param (old signature)"

# Must NOT have 'ground_truths' as a positional param (should come via **kwargs)
assert "ground_truths" not in params, \
    f"❌ FAIL: reward_fn still has 'ground_truths' as positional param. Should come via **kwargs"

print(f"  ✅ reward_fn signature: ({', '.join(params)}) — uses **kwargs correctly")


# ── Test 4: run_episode_via_http makes HTTP calls ───────────────────────────
print("Test 4: Verifying run_episode_via_http() makes HTTP POST calls...")
from train.train_minimal import run_episode_via_http

# Mock requests to verify it makes the right calls
with patch("train.train_minimal.http_client") as mock_http:
    # Setup mock responses
    mock_reset_resp = MagicMock()
    mock_reset_resp.json.return_value = {"session_id": "test-session-123"}
    mock_reset_resp.raise_for_status = MagicMock()

    mock_step_resp = MagicMock()
    mock_step_resp.json.return_value = {"reward": 0.85, "done": True}
    mock_step_resp.raise_for_status = MagicMock()

    mock_http.post.side_effect = [mock_reset_resp, mock_step_resp]

    reward = run_episode_via_http(
        task_id="clean_claim",
        seed=42,
        decision="approve_claim",
        confidence="HIGH",
        reason="All documents verified.",
        base_url="http://fake:7860",
    )

    # Verify POST /reset was called
    calls = mock_http.post.call_args_list
    assert len(calls) == 2, f"❌ FAIL: Expected 2 POST calls, got {len(calls)}"

    reset_call = calls[0]
    assert "/reset" in reset_call[0][0], f"❌ FAIL: First POST not to /reset"
    reset_body = reset_call[1]["json"]
    assert reset_body["task_id"] == "clean_claim", f"❌ FAIL: /reset body missing task_id"
    assert reset_body["seed"] == 42, f"❌ FAIL: /reset body missing seed"

    step_call = calls[1]
    assert "/step" in step_call[0][0], f"❌ FAIL: Second POST not to /step"
    step_body = step_call[1]["json"]
    assert step_body["session_id"] == "test-session-123", f"❌ FAIL: /step missing session_id from /reset"
    assert step_body["action"]["action_type"] == "approve_claim", f"❌ FAIL: action_type wrong"
    assert step_body["action"]["confidence"] == "HIGH", f"❌ FAIL: confidence wrong"

    assert reward == 0.85, f"❌ FAIL: reward should be 0.85 from /step, got {reward}"

print("  ✅ run_episode_via_http() makes POST /reset then POST /step correctly")
print(f"     → /reset body: {{task_id, seed}}")
print(f"     → /step body: {{action: {{action_type, confidence, reasoning}}, session_id}}")
print(f"     → reward returned from /step response: 0.85")


# ── Test 5: reward_fn calls run_episode_via_http (not training_reward) ──────
print("Test 5: Verifying reward_fn calls HTTP, not training_reward()...")

with patch("train.train_minimal.run_episode_via_http") as mock_episode:
    mock_episode.return_value = 0.75

    completions = [
        [{"content": "DECISION: approve_claim\nCONFIDENCE: HIGH\nREASON: docs verified"}],
        [{"content": "DECISION: deny_claim\nCONFIDENCE: MED\nREASON: suspicious docs"}],
    ]
    prompts = ["prompt1", "prompt2"]

    rewards = reward_fn(
        completions,
        prompts,
        task_id=["clean_claim", "contradictory_claim"],
        seed=["42", "43"],
        ground_truth=["approve_claim", "deny_claim"],
    )

    assert mock_episode.call_count == 2, f"❌ FAIL: Expected 2 HTTP calls, got {mock_episode.call_count}"
    assert rewards == [0.75, 0.75], f"❌ FAIL: rewards should be [0.75, 0.75], got {rewards}"

print("  ✅ reward_fn calls run_episode_via_http() for each completion")


# ── Test 6: _start_env_server_if_needed fails without server ────────────────
print("Test 6: Verifying training fails without server...")
from train.train_minimal import _wait_for_env

try:
    # Use very short retries to a port that's definitely not running
    _wait_for_env("http://localhost:19999", retries=1)
    print("  ❌ FAIL: Should have raised RuntimeError when server is unreachable")
    sys.exit(1)
except RuntimeError as e:
    assert "not reachable" in str(e).lower(), f"❌ FAIL: Error message unclear: {e}"
    print(f"  ✅ _wait_for_env raises RuntimeError when server is down")


# ── Test 7: WandB config says env-connected ─────────────────────────────────
print("Test 7: Checking WandB tags and config...")
assert '"env-connected"' in source, "❌ FAIL: WandB tags don't include 'env-connected'"
assert '"http-reward"' in source, "❌ FAIL: WandB tags don't include 'http-reward'"
assert '"env_http_reward"' in source, "❌ FAIL: reward_type not set to 'env_http_reward'"
assert '"no-server"' not in source, "❌ FAIL: WandB tags still contain 'no-server'"
assert '"direct-reward"' not in source, "❌ FAIL: WandB tags still contain 'direct-reward'"
print("  ✅ WandB config correctly reflects env-connected training")


# ── Final Summary ───────────────────────────────────────────────────────────
print()
print("=" * 70)
print("  ALL 7 TESTS PASSED ✅")
print()
print("  MR-2 Compliance verified:")
print("    • reward_fn calls POST /reset + POST /step (not training_reward)")
print("    • make_row() includes task_id + seed for /reset")
print("    • Training WILL FAIL if environment server is not running")
print("    • No 'no-server' or 'direct-reward' remnants in code")
print("=" * 70)
"""
This script validates:
  1. The module docstring declares env-connected training
  2. make_row() includes task_id and seed columns
  3. reward_fn uses **kwargs (not positional args)
  4. run_episode_via_http() makes correct POST /reset then POST /step
  5. reward_fn dispatches to run_episode_via_http (not training_reward)
  6. _wait_for_env raises RuntimeError when server is unreachable
  7. WandB config has correct env-connected tags
"""
