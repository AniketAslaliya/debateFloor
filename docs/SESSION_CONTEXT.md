# DebateFloor Session Context

Updated: April 22, 2026

## Current Status

DebateFloor is feature-complete for the Meta PyTorch x Scaler Grand Finale.
The FastAPI/OpenEnv server, Gradio UI, calibration grader, procedural claim
generator, debate panel, inference script, tests, and live HF Space are all in
place.

The older `docs/CONTEXT.md` is a stale Day 1 log. Use this file plus
`CLAUDE.md` as the current handoff.

## Project Snapshot

- Repo: `github.com/AniketAslaliya/debateFloor`
- Branch: `main`
- HF Space: `https://aniketasla-debatefloor.hf.space`
- Local path: `c:\Users\Dell\Documents\debatefloor`
- Hackathon: Meta PyTorch x Scaler Grand Finale, April 25-26 2026, Bangalore
- Team: Aniket Aslaliya, Mitali Mehta, Aditya Sharma

## What Is Done

- OpenEnv endpoints: `/reset`, `/step`, `/state`, `/tasks`, `/health`, `/schema`
- Core environment in `app/environment.py`
- Calibration matrix and anti-gaming logic in `server/calibration_grader.py`
- Procedural episode generation in `server/claim_generator.py`
- Multi-agent debate panel through `convene_debate_panel`
- Gradio UI at `/ui` with live calibration matrix and debate rendering
- Baseline inference script for all 3 required demo tasks
- Unit tests for calibration and generator
- Pre-validation script for local and HF Space checks
- Training script: `train/train_minimal.py`
- HF blog draft: `docs/HFBlogPost.md`
- Submission checklist: `docs/SUBMISSION_CHECKLIST.md`
- HTTP rollout evaluator: `scripts/evaluate_http_rollouts.py`
- Training dependency file: `train/requirements.txt`

## Current Priority

1. Get the Colab/WandB training result.
2. Save the reward curve as `docs/reward_curve.png`.
3. Commit and push the reward curve plus `reports/training_summary.json`.
4. Upload the same image to the HF Space using `HfApi.upload_file`.
5. Publish the HF blog from `docs/HFBlogPost.md`.
6. Add the final blog URL to `README.md`.

## Known Local State

- `PITCH_CHECKLIST.md` is currently untracked.
- `docs/reward_curve.png` does not exist yet.
- README already references `docs/reward_curve.png`.
- The HF blog draft now also references `docs/reward_curve.png`.
- Live HTTP rollout report exists at `reports/http_rollout_eval.md`.
- Live pre-validation passed on April 22, 2026 against `https://aniketasla-debatefloor.hf.space`.

## Critical Rules

- Never mix `training_reward` with `eval_reward`.
- Terminal actions must include `confidence` as `HIGH`, `MED`, or `LOW`.
- Never hardcode `HF_TOKEN`.
- Never push large files to the HF Space through git; use `HfApi.upload_file`.
- On this machine, do not use `pip` without `--break-system-packages`.
- Do not change T4 training dtype behavior away from the current auto-detection.

## Resume Command Checklist

```powershell
git status --short
python pre_validation_script.py --base-url https://aniketasla-debatefloor.hf.space
python inference_debatefloor.py --all-tasks --base-url https://aniketasla-debatefloor.hf.space
```
