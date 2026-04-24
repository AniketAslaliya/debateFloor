# DebateFloor — Complete Project Reference (Brahmastra)
> **Read this file first, every time. It is the single source of truth.**
> Last updated: April 24, 2026 — Hackathon eve.

---

## 1. What Is DebateFloor (30-second pitch)

DebateFloor trains LLMs to declare **calibrated confidence** before every insurance claim decision. The core innovation: a 3×2 calibration matrix where being *wrong and overconfident* (HIGH + wrong = **−0.8**) is far worse than being *wrong but humble* (LOW + wrong = **0.0**). This teaches the model **when to be confident**, not just what to say — directly fixing the overconfidence problem proven by the CAPO paper (April 2026).

**The unique mechanic no other OpenEnv environment has:** Before every hard decision, the agent calls `convene_debate_panel` — spinning up a Prosecutor (fraud signals) and a Defender (document consistency) who argue independently. The Judge (main agent) reads both and declares calibrated confidence.

**Based on:** [CoCA arXiv:2603.05881](https://arxiv.org/abs/2603.05881)

---

## 2. Links (All the judges need)

| Resource | URL |
|----------|-----|
| **Live HF Space (UI + API)** | https://huggingface.co/spaces/AniketAsla/debatefloor |
| **GitHub Repo** | https://github.com/AniketAslaliya/debateFloor |
| **Mini-blog (in repo)** | [docs/HFBlogPost.md](docs/HFBlogPost.md) |
| **Training notebook** | [train/train_debatefloor.ipynb](https://github.com/AniketAslaliya/debateFloor/blob/main/train/train_debatefloor.ipynb) |
| **CoCA paper** | https://arxiv.org/abs/2603.05881 |
| **WandB project** | https://wandb.ai/aniketaslaliya-lnmiit/debatefloor-insurance-rl |

---

## 3. Project Structure (Code Map)

```
debatefloor/
├── README.md                       ← submission-facing, judges read this
├── BRAHMASTRA.md                   ← YOU ARE HERE (team reference, not for judges)
├── openenv.yaml                    ← OpenEnv manifest (spec_version:1)
├── Dockerfile                      ← HF Space deployment
├── requirements.txt                ← server deps (fastapi, uvicorn, pydantic)
├── inference_debatefloor.py        ← mandatory baseline agent ([START]/[STEP]/[END])
├── pre_validation_script.py        ← runs all endpoint checks against live URL
│
├── app/                            ← FastAPI server (OpenEnv contract)
│   ├── main.py                     ← serves React UI at / + all endpoints
│   ├── environment.py              ← InsuranceClaimEnvironment + debate wiring
│   ├── models.py                   ← Pydantic (confidence: HIGH|MED|LOW)
│   └── tasks.py                    ← task definitions + reward computation
│
├── server/                         ← DebateFloor core
│   ├── calibration_grader.py       ← 3×2 matrix + anti-gaming + training_reward()
│   └── claim_generator.py          ← procedural episode generator (500+ episodes)
│
├── frontend/                       ← React UI (Vite build)
│   ├── src/App.jsx                 ← main UI (hero, debate panel, matrix, terminal)
│   ├── src/tasks.js                ← task strategies + descriptions for demo
│   ├── src/index.css               ← all styles (glassmorphism, animations)
│   └── dist/                       ← compiled build (committed, served by FastAPI)
│
├── train/
│   ├── train_minimal.py            ← PRIMARY training script (pure TRL, T4 in 15 min)
│   ├── train_debatefloor.ipynb     ← Colab notebook (wraps train_minimal.py)
│   └── requirements.txt            ← training deps (trl, transformers, wandb...)
│
├── tests/
│   ├── test_calibration.py         ← 13 tests for calibration_grader.py
│   └── test_generator.py           ← 32 tests (500-episode uniqueness check)
│
├── docs/
│   ├── HFBlogPost.md               ← HF blog (markdown in repo = valid per organizers)
│   ├── reward_curve.svg            ← training reward curve (from train_minimal.py)
│   ├── component_shift.svg         ← before/after component scores
│   └── confidence_distribution.svg ← HIGH/MED/LOW distribution shift histogram
│
└── reports/
    ├── training_summary.json       ← full log_history from real training run
    ├── component_shift_summary.json← before/after component means (JSON)
    └── http_rollout_eval.md        ← live Space rollout validation report
```

---

## 4. The Calibration Matrix (Core Innovation)

```
              Correct Decision    Wrong Decision
HIGH conf  →      +1.0               -0.8   ← worst possible outcome
MED  conf  →      +0.6               -0.2
LOW  conf  →      +0.1                0.0
```

**Anti-gaming system** (in `server/calibration_grader.py`):
- LOW rate > 70% across 10+ episodes → penalty `(rate − 0.70) × 2.0`
- HIGH rate > 80% across 10+ episodes → penalty `(rate − 0.80) × 1.5`
- Only winning strategy: accurate calibration matching task difficulty

**Two separate rewards — NEVER mix them:**
```python
training_reward()  # simple scalar → use for GRPO (stable gradients)
eval_reward()      # 6-component   → use for demo/README only
```

---

## 5. Three Tasks (Demo Order)

| Task | Difficulty | Max Steps | Correct Decision | Required Confidence |
|------|-----------|-----------|-----------------|-------------------|
| `clean_claim` | Easy | 10 | `approve_claim` | HIGH |
| `contradictory_claim` | Medium | 18 | `deny_claim` | MED + Debate Panel |
| `distribution_shift_claim` | Hard | 28 | `escalate_to_human` | LOW (HIGH always penalised) |

**Always demo `contradictory_claim` first** — it triggers the Debate Panel and shows all 3 agents.

---

## 6. Debate Panel — 90-Second Demo Script

> *Run this sequence when presenting to judges.*

1. Select **`contradictory_claim`** → click **Run Episode**
2. Watch terminal: agent validates documents → flags `date_mismatch` + `cost_inflation`
3. At step 5–6: `convene_debate_panel` fires
4. **Prosecutor [STRONG]** appears: *"2 fraud signals found — recommend denial"*
5. **Defender [WEAK]** appears: *"Document consistency exists — insufficient proof"*
6. **Verdict**: Prosecution wins → agent declares `deny_claim` + **MED** confidence
7. Matrix lights up: `MED × correct = +0.6` (green cell glows)
8. Say: *"The agent didn't say HIGH because it respected the Defender's argument. That's calibration."*

---

## 7. Training Evidence (Real Numbers)

From `reports/training_summary.json` (real T4 Colab run):

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Episodes | 100 |
| Epochs | 2 |
| Reward at step 5 | **−0.342** |
| Reward at step 45 (peak) | **+1.178** |
| Reward at step 100 (final) | **+0.828** |
| Training time | ~18.6 min on T4 |
| Calibration before | **−0.8** |
| Calibration after | **0.0** (delta: **+0.8**) |

**Confidence distribution shift:**
| Confidence | Before | After |
|------------|--------|-------|
| HIGH | ~82% | **~44%** ↓ |
| MED | ~12% | **~36%** ↑ |
| LOW | ~6% | **~20%** ↑ |

> **Note for judges:** Training reward is an unbounded shaped scalar for GRPO gradient stability. Evaluation reward is clamped to `[0.0, 1.0]`.

---

## 8. API Quick Reference

```bash
# Base URL
HF_SPACE=https://aniketasla-debatefloor.hf.space

# Health
curl $HF_SPACE/health

# Reset an episode
curl -X POST $HF_SPACE/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "contradictory_claim", "seed": 42}'

# Step (non-terminal)
curl -X POST $HF_SPACE/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action": {"action_type": "validate_document", "reasoning": "check docs"}}'

# Step (terminal — confidence REQUIRED)
curl -X POST $HF_SPACE/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "action": {"action_type": "deny_claim", "confidence": "MED", "reasoning": "procedure mismatch"}}'

# All tasks
curl $HF_SPACE/tasks
```

**All available actions:**
```
Non-terminal:  validate_document, flag_fraud_signal, request_information,
               lookup_policy_history, compare_documents, estimate_payout,
               query_historical_data, query_linked_claim, verify_identity,
               verify_provider_registration, convene_debate_panel

Terminal:      approve_claim, deny_claim, escalate_to_human
               (all require: "confidence": "HIGH"|"MED"|"LOW")
```

---

## 9. Local Development Commands

```powershell
# Run server locally
cd c:\Users\Dell\Documents\debatefloor
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# Run tests
PYTHONPATH=. pytest tests/test_calibration.py tests/test_generator.py -v

# Validate against live HF Space
python pre_validation_script.py --base-url https://aniketasla-debatefloor.hf.space

# Build React UI (after frontend changes)
cd frontend && npm run build && cd ..

# Run baseline agent
python inference_debatefloor.py --task contradictory_claim --base-url https://aniketasla-debatefloor.hf.space
```

---

## 10. Training Script — Two Versions

### `train/train_minimal.py` (PRIMARY — use this)
- **What it does:** GRPO trains Qwen2.5-0.5B. Saves `docs/reward_curve.svg`, `docs/component_shift.svg`, `reports/training_summary.json`
- **How to run:** `python train/train_minimal.py` (set `WANDB_API_KEY` env var for logging)
- **Why it's primary:** Already ran successfully. Produced the reward curve in the README.

### `train/train_debatefloor.ipynb` (Colab version)
- **What it does:** Wraps `train_minimal.py` with a Cell 2 config section (model, episodes, WandB key)
- **How it works:** Cell 4 patches config into `train_minimal.py` and calls `tm.main()`
- **Why dynamic:** Change only Cell 2 — everything else follows automatically

**To run in Colab:**
1. Open: `https://github.com/AniketAslaliya/debateFloor/blob/main/train/train_debatefloor.ipynb`
2. Click "Open in Colab"
3. Switch runtime to T4 GPU
4. Cell 1: installs deps + clones repo → **restart runtime**
5. Cell 2: paste your `WANDB_API_KEY` + set `WANDB_ENTITY`
6. Run all remaining cells
7. Cell 7: prints the specific WandB run URL → paste into README

---

## 11. Tomorrow's Hackathon — Step-by-Step

### Before you leave home (tonight / early morning)
- [ ] Verify HF Space is Running: `curl https://aniketasla-debatefloor.hf.space/health`
- [ ] Open the UI in browser and run `contradictory_claim` once — confirm debate panel appears
- [ ] Run pre-validation: `python pre_validation_script.py --base-url https://aniketasla-debatefloor.hf.space`

### At the venue — with compute credits
1. Open `train/train_debatefloor.ipynb` in Colab
2. Switch to T4 GPU runtime
3. Paste `WANDB_API_KEY` and your `WANDB_ENTITY` into Cell 2
4. Run all cells (~15 min)
5. Cell 7 prints the specific WandB run URL
6. Update README: replace the WandB project link with the **specific run URL**
7. `git add README.md && git commit -m "docs: add specific WandB run URL from hackathon training" && git push`

### During demo (3-minute script)
1. Open https://huggingface.co/spaces/AniketAsla/debatefloor
2. **Start with `contradictory_claim`** (not clean_claim — it's boring)
3. Click Run Episode — narrate as the steps appear in the terminal
4. When debate panel appears: *"This is the multi-agent mechanic — Prosecutor vs Defender"*
5. Point at the matrix: *"MED confidence + correct decision = +0.6. HIGH would have been +1.0 but the Defender created doubt."*
6. Then show `distribution_shift_claim`: *"This one punishes HIGH confidence regardless. The model must escalate."*

### Q&A answers (memorise these)
| Question | Answer |
|----------|--------|
| Is this a benchmark? | No — episodes are procedurally generated from seeds. Same seed = same episode, different seed = different episode. 500+ unique training episodes. |
| Can agents game it by saying LOW always? | Anti-gaming fires if LOW > 70% across 10+ episodes. Penalty = `(rate−0.7)×2.0`. Only accurate calibration wins. |
| Why is reward modest? | The real signal is the confidence distribution shift: HIGH drops 82%→44%, MED rises 12%→36%. Model learns WHEN to be confident. |
| How is it multi-agent? | `convene_debate_panel` triggers Prosecutor and Defender reasoning from different evidence sets. The Judge reads both. Three independent reasoning contexts per episode. |
| What if training curve isn't impressive? | We already have real evidence: −0.342 → +0.828. The calibration score shifted from −0.8 to 0.0. The distribution histogram shows the shift visually. |

---

## 12. Theme Alignment (What the judges are scoring on)

| Theme | Bonus | What We Built |
|-------|-------|---------------|
| Theme 3.1 — World Modeling | Scaler AI Labs: Multi-App RL | 5 fraud types, multi-doc investigation, IRDAI registry, policy history |
| Theme 1 — Multi-Agent | Fleet AI: Scalable Oversight | 3-agent debate: Prosecutor + Defender + Judge |
| Theme 4 — Self-Improvement | Curriculum | easy→medium→hard + anti-gaming detector |

---

## 13. Critical Rules (Never Break)

- **Never mix `training_reward` and `eval_reward`** — compound rewards break GRPO gradients
- **Never hardcode `HF_TOKEN`** in any file
- **Never push large video/binary files** to HF Space via git — use `HfApi.upload_file`
- **Terminal actions always need `confidence`** — `"HIGH"`, `"MED"`, or `"LOW"`
- **Frontend changes** → always run `npm run build` in `frontend/` before committing

---

## 14. Team

- **Aniket Aslaliya** — environment core, claim generator, calibration grader, UI
- **Mitali Mehta** — domain knowledge (fraud types, IRDAI regulations), grader design
- **Aditya Sharma** — training pipeline, GRPO notebook, WandB integration
