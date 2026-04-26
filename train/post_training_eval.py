"""
post_training_eval.py — Re-run before/after component eval without GRPO training.

Use when:
  - Training finished but the process exited before save_training_artifacts(), or
  - You want fresh eval plots/JSON from an existing checkpoint.

Prerequisites:
  - Live ClaimCourt / DebateFloor env at ENV_BASE_URL (or let this script start uvicorn on :7860).
  - Checkpoint folder from training (default ./debatefloor_checkpoint).

Match training episode count so eval episodes are drawn from the same pool as train_minimal:
  EPISODES=10000 EPOCHS=2 BATCH_SIZE=4 python train/post_training_eval.py

Optional: larger held-out eval (more stable headline numbers):
  EVAL_EPISODES=18 python train/post_training_eval.py

Usage:
  cd repo-root
  set PYTHONPATH=.
  python train/post_training_eval.py
  python train/post_training_eval.py --checkpoint path/to/merged_model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Repo root = parent of train/
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-training eval only (refresh reports + docs plots).")
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT_PATH", "debatefloor_checkpoint"),
        help="HF-style folder with config + weights (default: ./debatefloor_checkpoint)",
    )
    p.add_argument(
        "--fresh-summary",
        action="store_true",
        help="Do not merge log_history from reports/training_summary.json (eval-only; empty reward curve).",
    )
    return p.parse_args()


def run_eval(
    checkpoint: str | Path,
    *,
    fresh_summary: bool = False,
    stop_env_server: bool | None = None,
) -> None:
    """
    Run before/after component eval and write reports + docs plots.

    stop_env_server: if True, terminate subprocess uvicorn started here.
        If False, leave running. If None (default), stop only if we started it
        (same as CLI behaviour).
    """
    ckpt = Path(checkpoint).resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt}")

    import torch

    import train.train_minimal as tm

    tm.EPISODES = int(os.environ.get("EPISODES", str(tm.EPISODES)))
    tm.EPOCHS = int(os.environ.get("EPOCHS", str(tm.EPOCHS)))
    tm.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", str(tm.BATCH_SIZE)))
    tm.ENV_BASE_URL = os.environ.get("ENV_BASE_URL", tm.ENV_BASE_URL)
    tm.MODEL_NAME = os.environ.get("MODEL_NAME", tm.MODEL_NAME)
    tm.HAS_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tm.USE_FP16 = torch.cuda.is_available() and not tm.HAS_BF16
    tm.DTYPE = torch.bfloat16 if tm.HAS_BF16 else torch.float16

    server_proc = tm._start_env_server_if_needed(tm.ENV_BASE_URL)
    _we_started_env = server_proc is not None
    if stop_env_server is None:
        stop_env_server = _we_started_env

    print(f"[OK] Env: {tm.ENV_BASE_URL} | EPISODES={tm.EPISODES} EVAL_EPISODES={tm.EVAL_EPISODES}")

    episode_pool = tm.generate_episode_pool(count=tm.EPISODES + (tm.EVAL_EPISODES * 4))
    eval_episodes = tm._select_eval_episodes(episode_pool[tm.EPISODES :])
    print(f"  Eval pool: {len(eval_episodes)} episodes")

    if tm.USE_UNSLOTH:
        print(f"Loading base via Unsloth: {tm.MODEL_NAME}")
        model, tok = tm.FastLanguageModel.from_pretrained(
            model_name=tm.MODEL_NAME,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        tm.FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base via transformers: {tm.MODEL_NAME}")
        tok = AutoTokenizer.from_pretrained(tm.MODEL_NAME)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            tm.MODEL_NAME,
            torch_dtype=tm.DTYPE,
            device_map="auto",
        )

    tm._tok_ref = tok
    print("Baseline eval (before)...")
    before_eval = tm.evaluate_component_shift(model, tok, eval_episodes)
    print(f"  Before: {before_eval['means']}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading checkpoint: {ckpt}")
    tok_ft = AutoTokenizer.from_pretrained(str(ckpt))
    if tok_ft.pad_token is None:
        tok_ft.pad_token = tok_ft.eos_token
    model_ft = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        torch_dtype=tm.DTYPE,
        device_map="auto",
    )
    tm._tok_ref = tok_ft

    print("Post-training eval (after)...")
    after_eval = tm.evaluate_component_shift(model_ft, tok_ft, eval_episodes)
    print(f"  After: {after_eval['means']}")

    log_history: list = []
    global_step = 0
    training_loss = 0.0
    summary_path = Path("reports/training_summary.json")
    if not fresh_summary and summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            log_history = list(prev.get("log_history") or [])
            global_step = int(prev.get("global_step") or 0)
            training_loss = float(prev.get("training_loss") or 0.0)
            print(f"  Preserved {len(log_history)} log_history rows from existing summary.")
        except Exception as exc:
            print(f"  [WARN] Could not read prior summary: {exc}")

    trainer = SimpleNamespace(state=SimpleNamespace(log_history=log_history))
    result = SimpleNamespace(global_step=global_step, training_loss=training_loss)

    tm.save_training_artifacts(
        trainer,
        result,
        before_eval["means"],
        after_eval["means"],
    )
    print("[OK] Updated reports/training_summary.json, docs/*.svg, reports/component_shift_summary.json")

    if stop_env_server and server_proc is not None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()
        print("[STOP] Stopped subprocess env server.")


def main() -> None:
    args = _parse_args()
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_dir():
        print(f"ERROR: checkpoint directory not found: {ckpt}")
        print("Train first (saves ./debatefloor_checkpoint) or pass --checkpoint /path/to/model")
        sys.exit(1)
    try:
        run_eval(ckpt, fresh_summary=args.fresh_summary)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
