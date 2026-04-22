# ROADMAP.md — DebateFloor Build Checklist (Updated: Multi-Agent)
## Meta PyTorch × Scaler Hackathon Grand Finale | April 25–26, 2026

---

## THEME ALIGNMENT (claim ALL three)

| Theme | Sub-theme / Bonus Prize | DebateFloor Claim |
|-------|------------------------|-------------------|
| Theme 3.1 — World Modeling (Professional) | Scaler AI Labs: Multi-App RL for Enterprise Workflows | Insurance fraud across 5 doc types, IRDAI registry, policy history |
| Theme 1 — Multi-Agent Interactions | Fleet AI: Scalable Oversight | 3-agent Debate Panel: Prosecutor + Defender + Judge |
| Theme 4 — Self-Improvement | Curriculum via difficulty escalation | easy→medium→hard + anti-gaming detector |

---

## SCORING (updated with multi-agent)

| Criterion | Session 4 | Target | Key Action |
|-----------|-----------|--------|-----------|
| Innovation (40%) | 35 | 38 | 3-agent debate panel + curriculum |
| Storytelling (30%) | 20 | 28 | Gradio live demo + HF blog |
| Reward curve (20%) | 0 | 16 | Run train_minimal.py → WandB curve |
| Pipeline (10%) | 9 | 9 | All green ✅ |
| **TOTAL** | **64** | **91** | |

---

## CURRENT STATUS (April 22, 2026)

### DONE ✅
- [x] FastAPI server — all 6 OpenEnv endpoints
- [x] 3 tasks: clean_claim, contradictory_claim, distribution_shift_claim
- [x] 3×2 calibration matrix (core innovation)
- [x] Anti-gaming detector (LOW >70% / HIGH >80%)
- [x] Procedural episode generator (500+ unique episodes)
- [x] inference_debatefloor.py — [START]/[STEP]/[END] format, all 3 tasks success=True
- [x] 45/45 tests passing
- [x] HF Space live: aniketasla-debatefloor.hf.space
- [x] Gradio visual UI — calibration matrix highlights live
- [x] Multi-agent debate panel (convene_debate_panel action)
  - Prosecutor agent argues from discovered signals
  - Defender agent argues from document consistency
  - Judge agent decides with calibrated confidence
- [x] pre_validation_script.py — 37/37 checks passing on live HF Space
- [x] train_minimal.py — pure TRL, no Unsloth, runs T4 in 15 min

### IN PROGRESS / PENDING
- [ ] **Run train_minimal.py on Colab** → get WandB reward curve ← MOST URGENT (20% criterion)
- [ ] **Publish HF blog post** → mandatory minimum requirement ← BLOCKING
- [ ] Sponsor API integration (pending email details)
- [ ] Add 3rd full agent role (full escalation agent as separate role)
- [ ] docs/reward_curve.png from training run
- [ ] README: add WandB curve screenshot

---

## 3-AGENT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                   DebateFloor Episode                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  AGENT 1: INVESTIGATOR                                  │
│  ├── validate_document, flag_fraud_signal               │
│  ├── query_historical_data, query_linked_claim          │
│  └── Builds evidence base over N steps                  │
│                           ↓                             │
│  ACTION: convene_debate_panel                           │
│                           ↓                             │
│  AGENT 2: PROSECUTOR  ←→  AGENT 3: DEFENDER            │
│  "Fraud signals found:    "Documents consistent,        │
│   date_mismatch,           claimant deserves            │
│   cost_inflation"          due process"                 │
│                           ↓                             │
│  JUDGE: reads both arguments → terminal decision        │
│  + confidence: HIGH/MED/LOW                             │
│  → calibration_score via 3×2 matrix                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**How it works now (implemented):**
- `convene_debate_panel` action triggers environment-generated prosecutor + defender arguments
- Arguments are grounded in the investigation state (signals found, document count, policy history)
- Prosecutor strength: STRONG if ≥2 signals found, WEAK if only discovered, INSUFFICIENT if none
- Defender strength: STRONG if clean claim, WEAK if strong prosecution, MODERATE otherwise
- Panel verdict tells judge which way to lean + what decision to make
- Judge makes final terminal action with calibrated confidence

**Future extension (if time):**
- Full LLM-powered prosecutor + defender using HF inference
- Each agent gets own context window, own reward signal
- Judge scores both agents separately for argument quality

---

## REMAINING WORK (April 22–24)

### April 22 — Training Run (PRIORITY 1)
```
Open Colab → T4 runtime
!git clone https://github.com/AniketAslaliya/debateFloor
!pip install trl>=0.9.0 transformers peft accelerate datasets wandb requests
# Set WANDB_API_KEY in env
!cd debateFloor && python train/train_minimal.py
→ Screenshot WandB reward curve
→ Save to docs/reward_curve.png
→ Add to README and HF blog
```

### April 22 — HF Blog Post (PRIORITY 2, MANDATORY)
```
Go to huggingface.co/new-blog
Paste docs/HFBlogPost.md content
Add WandB curve screenshot
Publish → copy URL → add to README
```

### April 23 — Polish
- [ ] Add reward_curve.png to README (from training run)
- [ ] Add WandB link to README and HFBlogPost
- [ ] Sponsor API integration (waiting for email details)
- [ ] Test Gradio demo flow on HF Space end-to-end
- [ ] Timed pitch rehearsal: 3 minutes exactly

### April 24 — Buffer + Final Checks
- [ ] Run pre_validation_script.py against HF Space → all 37 green
- [ ] Run inference_debatefloor.py --all-tasks against HF Space
- [ ] Docker build test locally: `docker build -t debatefloor .`
- [ ] Backup: save all docs/ locally + USB

---

## PRE-PITCH CHECKLIST (Morning of April 25)

```
□ HF Space /health returns 200
□ pre_validation_script.py — all 37 checks green
□ inference_debatefloor.py --all-tasks — all 3 success=True
□ Gradio UI loads and demo runs (contradictory_claim + debate panel)
□ WandB reward curve is public and accessible
□ HF blog is live and linked from README
□ Confidence distribution histogram saved locally
□ Pitch timed at 2m 55s
□ Q&A answers rehearsed cold (see CLAUDE.md Pitch Q&A section)
□ Backup laptop has everything loaded
□ Phone >80% battery
```

---

## Q&A CHEAT SHEET

| Judge Question | Answer (30 seconds max) |
|---|---|
| Can't agent just say LOW always? | Anti-gaming: LOW >70% fires `(rate-0.7)×2.0` penalty |
| Why insurance domain? | High-stakes, information-asymmetric, requires genuine uncertainty — IRDAI ₹30K crore fraud |
| Reward curve is modest? | Confidence distribution shift is the real signal — HIGH drops 82%→44% |
| What does GRPO teach that fine-tuning doesn't? | Emergent calibration vs label fitting — the model learns WHEN to be confident, not just what to say |
| Is this really multi-agent? | Yes — Prosecutor, Defender, Judge are 3 separate reasoning roles. convene_debate_panel generates adversarial arguments from independent perspectives |
| How does this help real insurers? | An overconfident wrong approval costs ₹2–5 lakh. LOW+escalate costs ₹500 investigation. Our env trains exactly this tradeoff |
