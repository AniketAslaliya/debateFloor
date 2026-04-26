"""
push_to_hf_space.py — Make the Hugging Face Space file tree match a Git revision.

Exports `git archive <ref>` (exactly the files in that commit) and uploads to the
Space. Uses `delete_patterns` so old paths on the Hub are removed.

Environment:
  HF_SPACE_GIT_REF   — revision to archive (default: HEAD). Use `origin/main` to
                       match GitHub after `git fetch origin` even if local HEAD is behind.
  HF_SPACE_FETCH=1   — run `git fetch origin` before archiving (uses your default remote).
  HF_SPACE_BUILD_FRONTEND=1 — `npm run build` in frontend/ before archiving (optional).

If the Space is *linked* to GitHub in HF Settings, the UI may show the linked repo’s
commit until that link rebuilds. API uploads still update Space *files*; to avoid
confusion, either disconnect the Git link and deploy only via this script, or push
to GitHub and let the Space pull from there.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "AniketAsla/debatefloor"
REPO_TYPE = "space"
LOCAL_ROOT = Path(__file__).resolve().parent.parent

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN:
    try:
        from huggingface_hub import get_token

        HF_TOKEN = (get_token() or "").strip()
    except Exception:
        HF_TOKEN = ""
if not HF_TOKEN:
    print("ERROR: No HF token. Set HF_TOKEN or run `hf auth login`.")
    sys.exit(1)

GIT_REF = os.getenv("HF_SPACE_GIT_REF", "HEAD").strip()
FETCH_FIRST = os.getenv("HF_SPACE_FETCH", "0").strip().lower() in ("1", "true", "yes")
BUILD_FRONTEND = os.getenv("HF_SPACE_BUILD_FRONTEND", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _maybe_fetch() -> None:
    if not FETCH_FIRST:
        return
    print("HF_SPACE_FETCH=1 - git fetch origin ...")
    subprocess.run(
        ["git", "-C", str(LOCAL_ROOT), "fetch", "origin"],
        check=False,
    )


def _resolve_full_sha(git_ref: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(LOCAL_ROOT), "rev-parse", git_ref],
        text=True,
    ).strip()


def _git_short(git_ref: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(LOCAL_ROOT), "rev-parse", "--short", git_ref],
        text=True,
    ).strip()


def _export_git_tar_extract(dest: Path, git_ref: str) -> None:
    archived = subprocess.run(
        ["git", "-C", str(LOCAL_ROOT), "archive", "--format=tar", git_ref],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(dest)],
        input=archived.stdout,
        check=True,
    )


def _maybe_build_frontend() -> None:
    if not BUILD_FRONTEND:
        return
    fe = LOCAL_ROOT / "frontend"
    if not (fe / "package.json").is_file():
        return
    print("HF_SPACE_BUILD_FRONTEND=1 - npm run build ...")
    if sys.platform == "win32":
        subprocess.check_call("npm run build", cwd=str(fe), shell=True)
    else:
        subprocess.check_call(["npm", "run", "build"], cwd=str(fe))


def main() -> None:
    _maybe_fetch()
    _maybe_build_frontend()

    # Validate ref exists
    full_sha = _resolve_full_sha(GIT_REF)
    short = _git_short(GIT_REF)
    print(f"Sync Space <- git {GIT_REF} = {short} ({full_sha[:12]}...)")

    api = HfApi(token=HF_TOKEN)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _export_git_tar_extract(root, GIT_REF)
        nfiles = sum(1 for p in root.rglob("*") if p.is_file())
        print(f"  Archive: {nfiles} files -> {REPO_ID}")

        api.upload_folder(
            folder_path=str(root),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"sync: git {short} ({full_sha})",
            # Drop previous tree so Hub matches this archive only.
            delete_patterns=["**/*", "*"],
        )

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
