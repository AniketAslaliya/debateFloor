"""Push NEW-4 + HIGH-4/CF-1 fix (rev 8) to Hugging Face Space."""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token, whoami


REPO_ID = "AniketAsla/debatefloor"
REPO_TYPE = "space"

FILES_TO_UPLOAD = [
    "PLAN.md",
    "inference_debatefloor.py",
    "train/train_minimal.py",
    "train/generate_eval_report.py",
    "reports/eval_report.json",
    "reports/eval_report.md",
]


def main() -> int:
    token = get_token()
    if not token:
        print("ERROR: no cached HF token. Run `huggingface-cli login` first.", file=sys.stderr)
        return 2

    user = whoami(token=token)
    print(f"Authenticated as: {user.get('name')}")

    api = HfApi(token=token)
    info = api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Space found: {REPO_ID}  (sha={info.sha[:8] if info.sha else 'unknown'})")

    repo_root = Path(__file__).resolve().parent
    missing = [f for f in FILES_TO_UPLOAD if not (repo_root / f).exists()]
    if missing:
        print(f"ERROR: missing local files: {missing}", file=sys.stderr)
        return 4

    print("\nUploading the following files to the Space:")
    for f in FILES_TO_UPLOAD:
        size = (repo_root / f).stat().st_size
        print(f"  - {f}  ({size:,} bytes)")

    operations = [
        CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(repo_root / f))
        for f in FILES_TO_UPLOAD
    ]

    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message="feat(NEW-4 + HIGH-4/CF-1): coordinated_fraud + identity_fraud strategies + variance RuntimeError",
        commit_description=(
            "NEW-4: added _strategy_coordinated_fraud and _strategy_identity_fraud\n"
            "to inference_debatefloor.py with matching TASK_CONFIG entries. Both\n"
            "strategies trigger the env's full discovery path before flagging:\n"
            "  coordinated_fraud: validate DOC-21/22/23 + query 3 linked claims\n"
            "    (CLM-GROUP-302/303/304) records 4 of 5 expected_signals.\n"
            "    Skips shared_emergency_contact (no auto-record path).\n"
            "  identity_fraud: validate DOC-31/32 + compare DOC-31 vs DOC-34 +\n"
            "    lookup_policy_history records all 4 expected_signals.\n\n"
            "Live HF Space verification (5 seeds x 5 tasks = 25 episodes):\n"
            "  coordinated_fraud reward 0.7670 / evidence 4/4 / calib 0.6\n"
            "  identity_fraud    reward 0.8180 / evidence 4/4 / calib 0.6\n"
            "  exploit_penalty 0.000 on every seed (no raised-before-discovered).\n\n"
            "reports/eval_report.json regenerated against live HF:\n"
            "  25 rows (was 15) covering all 5 tasks x 5 variant_ids\n"
            "  average_reward 0.6988 (was 0.6363)\n"
            "  20/25 rows with evidence_quality > 0 (was 10/15)\n"
            "  completion_rate 100%, 5 distinct rewards\n\n"
            "HIGH-4 / CF-1: train/train_minimal.py reward_fn variance < 0.01\n"
            "now raises RuntimeError after a 2-batch warmup, matching the\n"
            "HACKATHON_CONSTRAINTS Part 4 CF-1 contract. Validator confirms\n"
            "warmup batches 1-2 do not raise, batch 3 raises with the\n"
            "contracted message, and high-variance batches never raise.\n\n"
            "Regression: 49/49 DebateFloor tests pass."
        ),
    )

    print(f"\nSpace commit created: {commit.commit_url}")
    print(f"Pushed OID: {commit.oid[:8] if commit.oid else 'n/a'}")
    print(f"Watch rebuild: https://huggingface.co/spaces/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
