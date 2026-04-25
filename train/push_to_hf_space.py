"""
push_to_hf_space.py — Upload code + artifacts directly to HF Space via API.

Bypasses git (which has .mov file issues) by using huggingface_hub upload_folder.
Only uploads the files that matter for the Space runtime, not media assets.

**Not uploaded** (intentional — Space serves the env API only): `tests/`,
`inference_debatefloor.py`, `pre_validation_script.py`, `.claude/`, notebooks,
and one-off root `push_*.py` scripts. Training runs locally or on a separate
HF GPU job; do not bloat the Space with full `train/` except the few files
listed in UPLOAD_PATTERNS below.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file

REPO_ID    = "AniketAsla/debatefloor"
REPO_TYPE  = "space"
LOCAL_ROOT = Path(__file__).parent.parent  # repo root

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN env var not set. Export it and re-run.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# Directories + files to upload (relative to repo root)
UPLOAD_PATTERNS = [
    "app",
    "server",
    "docs",
    "reports",
    "train/train_minimal.py",
    "train/real_model_eval.py",
    "openenv.yaml",
    "requirements.txt",
    "README.md",
]

# Files to explicitly skip (they're too large or not needed by the Space)
SKIP_SUFFIXES = {".mov", ".mp4", ".avi", ".safetensors", ".bin"}
SKIP_DIRS     = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache"}

def should_skip(path: Path) -> bool:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return False

print(f"Uploading to {REPO_ID} ({REPO_TYPE}) ...")

uploaded = 0
errors   = 0
for pattern in UPLOAD_PATTERNS:
    local_path = LOCAL_ROOT / pattern
    if not local_path.exists():
        print(f"  [skip] {pattern} — not found")
        continue

    if local_path.is_file():
        if should_skip(local_path):
            print(f"  [skip] {pattern} — large/binary")
            continue
        rel = str(local_path.relative_to(LOCAL_ROOT)).replace("\\", "/")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=rel,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"deploy: update {rel}",
            )
            print(f"  [ok] {rel}")
            uploaded += 1
        except Exception as exc:
            print(f"  [err] {rel}: {exc}")
            errors += 1

    elif local_path.is_dir():
        for fpath in sorted(local_path.rglob("*")):
            if not fpath.is_file():
                continue
            if should_skip(fpath):
                continue
            rel = str(fpath.relative_to(LOCAL_ROOT)).replace("\\", "/")
            try:
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=rel,
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    commit_message=f"deploy: update {rel}",
                )
                print(f"  [ok] {rel}")
                uploaded += 1
            except Exception as exc:
                print(f"  [err] {rel}: {exc}")
                errors += 1

print(f"\nDone: {uploaded} uploaded, {errors} errors.")
