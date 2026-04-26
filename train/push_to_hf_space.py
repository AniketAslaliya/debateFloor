"""
push_to_hf_space.py — Make the Hugging Face Space match this Git repo (like GitHub).

Exports `git archive HEAD` (exactly what is committed) and uploads it to the Space
in one Hub commit. Remote files not present in that tree are removed via
`delete_patterns` so the Space does not keep stale paths from older deploys.

Optional env:
  HF_SPACE_BUILD_FRONTEND=1 — run `npm run build` in frontend/ before `git archive`
      (only affects your working tree if you have uncommitted dist changes;
       normally commit built `frontend/dist` so GitHub and HF match.)
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

BUILD_FRONTEND = os.getenv("HF_SPACE_BUILD_FRONTEND", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _git_rev_short() -> str:
    return subprocess.check_output(
        ["git", "-C", str(LOCAL_ROOT), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def _export_git_head_tar_extract(dest: Path) -> None:
    """Materialize committed tree only (matches GitHub after push)."""
    archived = subprocess.run(
        ["git", "-C", str(LOCAL_ROOT), "archive", "--format=tar", "HEAD"],
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
    print("HF_SPACE_BUILD_FRONTEND=1 — npm run build …")
    if sys.platform == "win32":
        subprocess.check_call("npm run build", cwd=str(fe), shell=True)
    else:
        subprocess.check_call(["npm", "run", "build"], cwd=str(fe))


def main() -> None:
    _maybe_build_frontend()
    rev = _git_rev_short()
    print(f"Sync Space <- git HEAD {rev} (full mirror of committed files)")

    api = HfApi(token=HF_TOKEN)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _export_git_head_tar_extract(root)
        nfiles = sum(1 for p in root.rglob("*") if p.is_file())
        print(f"  Archive: {nfiles} files -> {REPO_ID}")

        # Remove remote files not in this upload so HF matches Git (no orphans).
        api.upload_folder(
            folder_path=str(root),
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"sync: mirror git {rev} to Space",
            delete_patterns="**/*",
        )

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
