# Video Script — "ClaimCourt: Teaching LLMs to Be Honest About What They Don't Know"

**Runtime target:** ~1:55 (buffer under 2:00).  
**Codename** `debatefloor` in all URLs. **Brand** **ClaimCourt** on screen.

**Numbers below** are from committed `reports/training_summary.json` and `reports/component_shift_summary.json` (5,000-episode GRPO, Qwen2.5-0.5B-Instruct, 2,500 steps, HF Jobs L4).

| Metric | Before | After |
|--------|--------|-------|
| Mean training reward (logged scalar) | **0.130** | **0.469** |
| Decision accuracy (held-out eval) | **0.00** | **1.00** |
| Calibration (held-out eval) | **0.00** | **1.00** |
| Fraud detection (held-out eval) | **0.00** | **0.33** |
| Final training loss | — | **0.00565** |
| Train wall time | — | **~3 h 3 min** |

---

## [0:00 – 0:12] HOOK — The Problem (12s)

**Visual:** Full-screen news headline montage (3 quick cards, ~4s each, Ken Burns zoom):

1. *"India loses ₹8,000–10,000 crore/year to insurance fraud"* — BCG–Medi Assist report  
2. *"LLMs hallucinate confidently — even when wrong"* — generic AI news framing  
3. *"Existing RL methods don't reward calibrated uncertainty"* — cite [CAPO arXiv:2604.12632](https://arxiv.org/abs/2604.12632)

**Narration:**

> "Indian insurance fraud costs us ten thousand crore rupees a year. The reason? Models — both human and AI — say *yes* when they should say *I'm not sure*. Today's LLMs are confidently wrong. We built an RL environment that punishes that exact behaviour."

---

## [0:12 – 0:25] THE PITCH — What Is ClaimCourt (13s)

**Visual:** Live HF Space tab, full-screen browser. Hero banner with **ClaimCourt** + tagline. Slow zoom on hero text.

**Narration:**

> "This is **ClaimCourt** — an OpenEnv-compliant reinforcement-learning environment that trains language models to make insurance-fraud decisions *with calibrated confidence*. Approve, deny, or escalate — and tell us how sure you are."

---

## [0:25 – 0:45] THE UI WALKTHROUGH — Show, Don't Tell (20s)

**Visual:** Stay on HF Space. Scroll slowly. **Cursor highlight** (yellow ring). ~3s each:

- Claim card (point at **seed**)
- **APPROVE / DENY / ESCALATE**
- Confidence pills **HIGH / MED / LOW**
- **3×2** reward matrix
- Live **reward** badge on click

**Narration:**

> "Every claim is procedurally generated from a seed — so judges can reproduce any episode. The agent picks an action, declares confidence, and gets a reward from this pre-committed three-by-two matrix. Overconfident wrong answers are punished harder than honest *I don't know* answers — that's the calibration signal."

**Cherry-on-top:** Bottom-right overlay: `{"action":"DENY","confidence":"HIGH",...}` → `{"reward":-0.8}` (or your env’s actual scalar) so viewers see real JSON, not a mockup.

---

## [0:45 – 1:05] BACKEND CONNECTION — UI ↔ Code (20s)

**Visual:** Split: **left** HF Space; **right** VS Code `app/main.py` → `/step`. Highlight scoring path. Then **right** → `train/train_minimal.py` → HTTP `POST /step` for GRPO rewards.

**Narration:**

> "Every button on the left hits this FastAPI `/step` endpoint on the right. The same endpoint serves the live demo *and* GRPO training — the model trains against the exact environment judges are clicking through. No simulation gap, no static dataset."

---

## [1:05 – 1:30] PROOF OF TRAINING — The Numbers (25s)

**Visual:** Three cuts (~8s each):

1. **WandB** — [project workspace](https://wandb.ai/aniketaslaliya-lnmiit/debatefloor-insurance-rl): open the **latest** run `grpo-qwen0.5b-env-connected`; reward vs step ~**0.13 → ~0.47** over **2,500** steps. Zoom the trend.  
2. **Terminal / HF Jobs log** — one line with `reward`, `reward_std`, `loss` (~`0.006`, `~0.47`). Green box.  
3. **`reports/training_summary.json`** in editor — `eval_reward_before` vs `eval_reward_after` side by side (or `component_shift_summary.json`).

**Narration:**

> "Here's the receipt. **Five thousand episodes** of GRPO on **Qwen two-point-five zero-point-five-B Instruct**, on Hugging Face Jobs. Mean training reward goes from **zero-point-one-three** to **zero-point-four-seven** — about three-point-six times higher. On held-out eval, **calibration** goes from **zero** to **one**, and **decision accuracy** from **zero** to **one**. Fraud detection moves from **zero** to **zero-point-three-three**. Final training loss is about **zero-point-zero-zero-five-seven**. Every number is in a JSON file in the repo — nothing is hand-tuned for the demo."

**Optional tighter line (if you prefer one headline pair):**

> "Calibration on held-out eval: **zero** → **one**. Decision accuracy: **zero** → **one**."

---

## [1:30 – 1:45] WHY IT MATTERS (15s)

**Visual:** HF Space. **Wrong** combo (e.g. approve + HIGH on fraud) → red / worst matrix cell. **Safer** combo (e.g. escalate + LOW where appropriate) → green / better reward. Hold green ~2s.

**Narration:**

> "An overconfident wrong call costs the insurer real money. Saying *escalate, I'm not sure* is the safe, profitable behaviour. ClaimCourt trains **calibrated uncertainty** through the reward — not through prompting alone."

---

## [1:45 – 1:55] CALL TO ACTION (10s)

**Visual:** End card; cursor hovers each line:

- **Space:** https://huggingface.co/spaces/AniketAsla/debatefloor  
- **GitHub:** https://github.com/AniketAslaliya/debateFloor  
- **Mini-blog (markdown in Space repo):** https://huggingface.co/spaces/AniketAsla/debatefloor/blob/main/docs/HFBlogPost.md  

**Narration:**

> "Code, live environment, training notebook, and mini-blog are linked from the README. Try it yourself. Thank you."

---

## Production checklist

| # | Item | Notes |
|---|------|--------|
| 1 | HF Space shows **ClaimCourt** | Rebrand already deployed; confirm in incognito before record. |
| 2 | WandB tab open | Project URL above → pick latest **grpo-qwen0.5b-env-connected**; **canonical** curves are still `docs/reward_curve.svg` in GitHub. |
| 3 | `reports/training_summary.json` open in editor | For the proof segment. |
| 4 | OBS cursor highlight | Yellow ring on clicks. |
| 5 | 1080p, 30 fps, screen + mic | No huge `.mov` in repo — upload video to YouTube only. |
| 6 | Optional music | Royalty-free, low under narration. |
| 7 | Title card 1.5s | `ClaimCourt — OpenEnv Hackathon India 2026` |

---

## Corrections vs older draft

- **Model:** was "Llama 3.2 1B" → **Qwen2.5-0.5B-Instruct** (matches training).  
- **Status:** was "running right now" → **completed** 5K run (adjust tense when you record).  
- **`[TBD]` filled:** calibration **0 → 1**, decision **0 → 1**, mean reward **0.130 → 0.469**, loss **~0.00565**, fraud **0 → 0.33**.  
- **Blog URL:** was `huggingface.co/blog/...` → **blob URL** on the Space repo (matches submission form / README).
