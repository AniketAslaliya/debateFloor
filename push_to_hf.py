"""
Push updated files to HF Space using huggingface_hub API.
This bypasses the git history issue caused by old .mov files in the remote.
"""
import os
import sys
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: Set HF_TOKEN environment variable first.")
    print("  $env:HF_TOKEN = 'hf_your_token_here'")
    sys.exit(1)

REPO_ID = "AniketAsla/debatefloor"
REPO_TYPE = "space"

# Files to upload (relative to repo root)
FILES_TO_UPLOAD = [
    "README.md",
    "docs/reward_curve.svg",
    "docs/component_shift.svg",
    "reports/training_summary.json",
    "BRAHMASTRA.md",
]

api = HfApi(token=HF_TOKEN)

print(f"Pushing {len(FILES_TO_UPLOAD)} files to {REPO_ID}...")
for filepath in FILES_TO_UPLOAD:
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} — file not found")
        continue
    print(f"  [UPLOAD] {filepath}")
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=filepath,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=f"docs: update {filepath} with final hackathon polish",
    )
    print(f"  [OK] {filepath}")

print("\n✅ All files pushed to HF Space!")
print(f"   → https://huggingface.co/spaces/{REPO_ID}")
