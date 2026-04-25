"""
jobs_run.py — single-entry driver for HF Jobs.

Designed to run inside `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` on HF
Jobs (L4/A10G/A100). Submits as:

    hf jobs run \\
        --flavor l4x1 \\
        --timeout 12h \\
        --secret HF_TOKEN=hf_xxx \\
        --secret WANDB_API_KEY=wandb_xxx \\
        --env EPISODES=10000 \\
        --env EPOCHS=2 \\
        --env DISABLE_VARIANCE_GUARD=1 \\
        --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \\
        python train/jobs_run.py

Phases (each one logs a clear banner so you can grep the log):

    [1/6] Install deps from train/requirements.txt + root requirements.txt
    [2/6] Boot env server (uvicorn) on 127.0.0.1:7860
    [3/6] Wait for /health == healthy
    [4/6] Run train.train_minimal.main()
    [5/6] Push checkpoint + reports/ + docs/ to the HF model repo
    [6/6] Cleanly exit (kills env server so billing stops)

Environment variables consumed:

    Required:
        HF_TOKEN           — HF write token (used to push checkpoint)
    Optional (with defaults):
        WANDB_API_KEY      — enables WandB logging if set
        WANDB_ENTITY       — wandb entity (default: aniketaslaliya-lnmiit)
        EPISODES           — training episodes (default: 10000)
        EPOCHS             — training epochs (default: 2)
        BATCH_SIZE         — per-device batch (default: 4)
        NUM_GENERATIONS    — GRPO group size (default: 4)
        GRAD_ACCUM         — gradient accumulation steps (default: 2)
        MAX_COMPLETION_LENGTH — output token cap (default: 80)
        MAX_PROMPT_LENGTH  — prompt token cap (default: 512)
        DISABLE_VARIANCE_GUARD — bypass CF-1 guard (default: 1)
        HF_MODEL_REPO      — where to push the trained model
                             (default: AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct)
"""
from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Force unbuffered stdout/stderr so HF Jobs log viewer shows every line in
# real time. Without this, prints sit in a 4KB buffer and the user only sees
# "Job started" for several minutes — making working jobs look broken.
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass
print = functools.partial(print, flush=True)  # noqa: A001 — intentional shadow

# Heartbeat: a single line every minute so the user knows the job is alive
# even during slow phases (pip install, model download, dataset prep).
_HEARTBEAT_START = time.time()


def _hb(label: str) -> None:
    elapsed = int(time.time() - _HEARTBEAT_START)
    mm, ss = divmod(elapsed, 60)
    print(f"[heartbeat +{mm:02d}:{ss:02d}] {label}")


# ── [0/6] Bootstrap the repo (when running as a one-shot script) ────────────
# When this file is executed via `python -c "exec(...)"` or downloaded as a
# raw script, it has no surrounding repo. Detect that and `git clone` ourselves
# so the rest of the script sees the real layout.
_BOOTSTRAP_MARKER = Path(__file__).resolve().parent.parent / "app" / "main.py"
if not _BOOTSTRAP_MARKER.exists():
    print("[0/6] Bootstrap: no repo on disk, cloning from GitHub", flush=True)
    _clone_dir = Path("/tmp/debatefloor")
    if not _clone_dir.exists():
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/AniketAslaliya/debateFloor.git",
             str(_clone_dir)]
        )
    os.chdir(_clone_dir)
    REPO_ROOT = _clone_dir
else:
    REPO_ROOT = Path(__file__).resolve().parent.parent
    os.chdir(REPO_ROOT)

sys.path.insert(0, str(REPO_ROOT))

_hb("driver script started")
print("=" * 70)
print("[1/6] Installing pinned deps from requirements files")
print("=" * 70)


def _pip_install(*args: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *args]
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


_pip_install("--upgrade", "pip")
_hb("upgraded pip")
_pip_install("-r", "requirements.txt")
_hb("installed root requirements.txt")
_pip_install("-r", "train/requirements.txt")
_hb("installed train/requirements.txt")

# ── [1.4/6] Purge torchvision AND evict it from sys.modules.
#
# Two-part problem:
#   (1) The HF Jobs base image claims 'pytorch:2.4.0-cuda12.1' but actually
#       ships torch 2.11.0+cu130, so any torchvision pin we make is wrong.
#   (2) Even after `pip uninstall torchvision`, Python keeps the partially-
#       loaded torchvision modules in sys.modules from earlier `pip install`
#       work, so `import transformers` still hits the broken cached state and
#       fails with "partially initialized module 'torchvision' has no
#       attribute 'extension'".
#
# Fix: uninstall the package AND surgically evict every torchvision.* entry
# from sys.modules so the next import attempt sees a clean slate.
print("\n  Purging torchvision (text-only training, not needed)...")
try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchvision"]
    )
    print("    Removed torchvision package from environment")
except subprocess.CalledProcessError:
    print("    torchvision not installed — nothing to remove")

_evicted = [k for k in list(sys.modules) if k == "torchvision" or k.startswith("torchvision.")]
for _k in _evicted:
    del sys.modules[_k]
if _evicted:
    print(f"    Evicted {len(_evicted)} torchvision modules from sys.modules cache")

# Also evict any partially-loaded transformers modules that might have already
# tried to import torchvision and cached a broken state (e.g. from this script
# importing `requests` earlier, which doesn't touch transformers, but be safe).
_tf_evicted = [k for k in list(sys.modules) if k == "transformers" or k.startswith("transformers.")]
for _k in _tf_evicted:
    del sys.modules[_k]
if _tf_evicted:
    print(f"    Evicted {len(_tf_evicted)} transformers modules from sys.modules cache")

# Tell transformers to be tolerant of missing optional vision deps (defense in
# depth; the uninstall + sys.modules eviction is what actually fixes it).
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ── [1.5/6] Sanity-check critical imports BEFORE we boot the env + load model.
print("\n  Sanity-checking critical imports...")
_failed = []
for _mod, _from in [
    ("torch", None),
    ("transformers", "PreTrainedModel"),  # forces full transformers init
    ("trl", "GRPOConfig"),  # forces grpo_trainer import
    ("peft", "LoraConfig"),
    ("accelerate", "Accelerator"),
    ("datasets", "Dataset"),
    ("wandb", None),
]:
    try:
        if _from:
            _m = __import__(_mod, fromlist=[_from])
            getattr(_m, _from)
        else:
            __import__(_mod)
        try:
            _v = __import__(_mod).__version__
        except Exception:
            _v = "?"
        print(f"    ok  {_mod:14s} {_v}")
    except Exception as _e:
        print(f"    FAIL {_mod:14s} → {type(_e).__name__}: {_e}")
        _failed.append((_mod, _from, _e))

if _failed:
    print("\n  Sanity check failed — aborting before model download.")
    raise SystemExit(1)

print("  All critical imports OK.\n")
_hb("import sanity check passed")
print("  Deps installed.\n")


# ── [2/6] Boot the env server in the background ─────────────────────────────
import requests as _requests  # imported AFTER pip install -r requirements.txt

print("=" * 70)
print("[2/6] Booting DebateFloor env server on 127.0.0.1:7860")
print("=" * 70)

ENV_BASE_URL = "http://127.0.0.1:7860"
_log_path = Path("/tmp/uvicorn_debatefloor.log")
_log_file = open(_log_path, "w")

env_proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "7860",
        "--log-level",
        "warning",
    ],
    cwd=str(REPO_ROOT),
    stdout=_log_file,
    stderr=subprocess.STDOUT,
)
print(f"  uvicorn PID = {env_proc.pid}")


# ── [3/6] Wait for /health ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[3/6] Waiting for env server /health")
print("=" * 70)


def _wait_for_env(max_tries: int = 60) -> None:
    for i in range(max_tries):
        if env_proc.poll() is not None:
            log = _log_path.read_text()[-4000:]
            raise RuntimeError(f"uvicorn died before /health was ready. Log:\n{log}")
        try:
            r = _requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                print(f"  Healthy after {i + 1} attempts.")
                return
        except Exception:
            pass
        time.sleep(2)
    log = _log_path.read_text()[-4000:]
    raise RuntimeError(f"Env never became healthy. Log:\n{log}")


_wait_for_env()
_hb("env server is healthy and accepting requests")


# ── [4/6] Run training ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("[4/6] Running train.train_minimal.main()")
print("=" * 70)
_hb("starting training phase — model download may take 1–2 min on first run")

# Surface key config so the log shows what we ran with
EPISODES = int(os.environ.get("EPISODES", "10000"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
print(f"  EPISODES={EPISODES} EPOCHS={EPOCHS} BATCH_SIZE={BATCH_SIZE}")
print(f"  NUM_GENERATIONS={os.environ.get('NUM_GENERATIONS', '4')}")
print(f"  GRAD_ACCUM={os.environ.get('GRAD_ACCUM', '2')}")
print(f"  MAX_COMPLETION_LENGTH={os.environ.get('MAX_COMPLETION_LENGTH', '80')}")
print(
    f"  DISABLE_VARIANCE_GUARD={os.environ.get('DISABLE_VARIANCE_GUARD', '1')}"
)
os.environ.setdefault("DISABLE_VARIANCE_GUARD", "1")
os.environ.setdefault("NUM_GENERATIONS", "4")
os.environ.setdefault("GRAD_ACCUM", "2")
os.environ.setdefault("MAX_COMPLETION_LENGTH", "80")
os.environ.setdefault("MAX_PROMPT_LENGTH", "512")
os.environ["ENV_BASE_URL"] = ENV_BASE_URL

import train.train_minimal as tm  # noqa: E402

tm.MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
tm.EPISODES = EPISODES
tm.EPOCHS = EPOCHS
tm.BATCH_SIZE = BATCH_SIZE
tm.USE_WANDB = bool(os.environ.get("WANDB_API_KEY", ""))
tm.WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
tm.WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "aniketaslaliya-lnmiit")
tm.ENV_BASE_URL = ENV_BASE_URL

import torch  # noqa: E402

tm.HAS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
tm.USE_FP16 = torch.cuda.is_available() and not tm.HAS_BF16
tm.DTYPE = torch.bfloat16 if tm.HAS_BF16 else torch.float16
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  dtype: {tm.DTYPE} | Unsloth: {tm.USE_UNSLOTH}\n")

train_exit_code = 0
try:
    tm.main()
    print("  Training completed.")
except Exception as exc:  # don't crash the whole job — we still want artifacts
    train_exit_code = 1
    print(f"  Training raised: {type(exc).__name__}: {exc}")
    import traceback

    traceback.print_exc()


# ── [5/6] Push artifacts to the HF Hub model repo ───────────────────────────
print("\n" + "=" * 70)
print("[5/6] Uploading artifacts to HF Hub")
print("=" * 70)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "AniketAsla/debatefloor-grpo-qwen2.5-0.5b-instruct",
)

if not HF_TOKEN:
    print("  HF_TOKEN not set — skipping upload (artifacts remain in job storage).")
else:
    try:
        from huggingface_hub import HfApi, login

        login(token=HF_TOKEN, add_to_git_credential=False)
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)

        ckpt_dir = Path("./debatefloor_checkpoint")
        if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            print(f"  Uploading checkpoint folder -> {HF_MODEL_REPO}")
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=HF_MODEL_REPO,
                repo_type="model",
                commit_message=f"GRPO HF Jobs run: {EPISODES} episodes x {EPOCHS} epochs",
            )
        else:
            print("  No ./debatefloor_checkpoint to upload (training may have failed early).")

        for artifact in [
            "reports/training_summary.json",
            "reports/component_shift_summary.json",
            "docs/reward_curve.svg",
            "docs/component_shift.svg",
        ]:
            p = Path(artifact)
            if p.exists():
                print(f"  Uploading {artifact}")
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=artifact,
                    repo_id=HF_MODEL_REPO,
                    repo_type="model",
                    commit_message=f"Update {artifact} from HF Jobs run",
                )
            else:
                print(f"  Skipping {artifact} (not found)")
    except Exception as exc:
        print(f"  Upload step raised: {type(exc).__name__}: {exc}")


# ── [6/6] Clean shutdown so HF Jobs stops billing ───────────────────────────
print("\n" + "=" * 70)
print("[6/6] Shutting down env server cleanly")
print("=" * 70)
try:
    env_proc.send_signal(signal.SIGTERM)
    env_proc.wait(timeout=10)
except Exception:
    env_proc.kill()
print("  Done.")
sys.exit(train_exit_code)
