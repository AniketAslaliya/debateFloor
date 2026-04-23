# DebateFloor Submission Checklist

Use this before final submission. Judges will evaluate the live URL and the
linked supporting material, not only the source code.

## Minimum Requirements

| Requirement | Repo status | Final action |
|---|---|---|
| OpenEnv latest release | `requirements.txt` and `pyproject.toml` require `openenv-core>=0.2.3` | Rebuild/redeploy HF Space to pull the newest compatible OpenEnv release |
| OpenEnv base classes | `InsuranceClaimEnvironment` subclasses `Environment`; models inherit OpenEnv `Action` / `Observation` / `State` | Run import smoke test after edits |
| OpenEnv manifest | `openenv.yaml` present | Keep `spec_version: 1` and Space URL submitted |
| Hosted HF Space | Live at `AniketAsla/debatefloor` | Run live validation on pitch day |
| REST endpoints | `/reset`, `/step`, `/state`, `/tasks`, `/health`, `/schema` in `app/main.py` | Validate with `pre_validation_script.py` |
| Training script | `train/train_minimal.py` uses TRL `GRPOTrainer`; `train/train_debatefloor.ipynb` provides Colab notebook flow with Unsloth | Re-run either path on Colab T4 and keep logs/artifacts |
| Training dependencies | `train/requirements.txt` | Install this in Colab |
| Training evidence | Script saves `docs/reward_curve.png`, `docs/component_shift.png`, `reports/training_summary.json`, and `reports/component_shift_summary.json` | Commit artifacts after real run |
| Env interaction evidence | `scripts/evaluate_http_rollouts.py` drives `/reset` and `/step` | Commit generated report |
| Client/server separation | External scripts use HTTP clients instead of importing `app.environment` | Keep server imports limited to tests/training internals |
| Mini-blog/video/slides | Draft exists at `docs/HFBlogPost.md` | Publish and add URL to README |
| README links | README has Space, UI, plot, report placeholders | Replace blog placeholder after publish |

## Final Commands

```powershell
python pre_validation_script.py --base-url https://aniketasla-debatefloor.hf.space
python inference_debatefloor.py --all-tasks --base-url https://aniketasla-debatefloor.hf.space
python scripts/evaluate_http_rollouts.py --base-url https://aniketasla-debatefloor.hf.space
```

## Real Data Rule

Do not commit real customer data. If real examples are used, convert them into
anonymous templates: remove names, phone numbers, addresses, policy numbers,
claim IDs, vehicle numbers, hospital identifiers, and exact dates. Keep only
aggregate patterns such as "recent policy purchase", "same broker across claims",
"bill/discharge mismatch", or "provider not in registry".
