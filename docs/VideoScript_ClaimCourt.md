# ClaimCourt — Demo video script (non-technical + full UI)

**Goal:** Someone with *no* ML background understands *what* ClaimCourt is, *why* it matters, sees *every major part* of your live Space, and wants to open the link.  
**Length:** ~1:55–2:00. Speak slowly; pause on numbers.  
**Brand on screen:** **ClaimCourt**. **URLs always use codename** `debatefloor` (unchanged links).

**Training proof (for one short segment):** 5,000 practice claims, reward **0.13 → 0.47**, held-out **calibration 0 → 1** and **decision accuracy 0 → 1** (see table at bottom).

---

## One-line premise (say this if you freeze)

> "ClaimCourt is a **practice courtroom for AI on insurance claims** — it learns not just *what* to decide, but *how sure* it should be."

---

## ACT 1 — Why this exists (~25 s)

### [0:00 – 0:08] Hook — money + mistake everyone makes

**Visual:** 2–3 full-screen title cards (no small text). Optional stock: busy hospital/claim desk silhouette.

**Say:**

> "India loses a staggering amount to insurance fraud every year — on the order of **eight to ten thousand crore rupees**. A lot of that isn’t cartoon villains — it’s **honest-looking paperwork** with something wrong underneath. The expensive mistake isn’t only *getting the answer wrong* — it’s being **sure** when you shouldn’t be. We built **ClaimCourt** so an AI can practice that skill."

*(Optional lower-third once: source — BCG × Medi Assist style reports; keep it readable.)*

### [0:08 – 0:25] What ClaimCourt is — no jargon first

**Visual:** Open **[ClaimCourt on Hugging Face](https://huggingface.co/spaces/AniketAsla/debatefloor)** full screen. Hero / top bar with **ClaimCourt** visible.

**Say:**

> "You’re looking at **ClaimCourt** — a **free, in-browser demo**. Pick a fake insurance case. Watch an AI **investigate** it like an analyst: read documents, spot red flags, sometimes call a **mini trial** with two opposing voices. At the end it must **approve**, **deny**, or **hand off to a human** — and say whether it’s **high**, **medium**, or **low** confidence. Same rules every time. You can try the next three examples yourself — link at the end."

**Avoid until later:** “OpenEnv”, “GRPO”, “reward shaping” — introduce in Act 3 in one sentence each.

---

## ACT 2 — The product tour: every UI piece (~55 s)

Use **one continuous screen recording** with a **yellow cursor ring**. Pause ~2–3 s on each labelled area below.

### [0:25 – 0:35] Left column — “Run an Episode”

**Visual:** **Run an Episode** card. Open the **dropdown**: show all three:

| Task (dropdown) | Plain-English pitch (say while hovering) |
|-----------------|------------------------------------------|
| **clean claim** | “Everything lines up — the honest answer is *approve*, and you should sound **sure**.” |
| **contradictory claim** | “Documents **fight each other** — dates, costs, procedures don’t match. The AI should dig, then often **deny** — with **medium** confidence, not bravado.” |
| **distribution shift claim** | “Looks normal **until** you pull in **linked** claims — shared brokers, patterns. Here the *right* move is often **hand to a human** and say **low** confidence — because the full picture is murky.” |

**Say (short):**

> "Three levels of difficulty — **easy**, **tricky**, and **‘looks fine until you connect the dots’**. Same button for all: **Run Episode**."

Click **Run Episode** once on **clean claim** so the audience sees the flow start.

### [0:35 – 0:50] Middle — “Claim Under Investigation”

**Visual:** Claim card: **ID**, **claimant name**, **incident** line, **document list** (DOC-1, DOC-2…).

**Say:**

> "Middle of the screen: the **fake claim file** — who it is, what happened, which PDFs exist. You’re not reading a research paper — you’re reading a **case file**. That’s deliberate: insurers think in cases, not equations."

### [0:50 – 1:05] Right — “agent-trace.log” (the story of the investigation)

**Visual:** Scroll the **agent-trace.log** panel. Point at lines like `validate_document`, `flag_fraud_signal`, `convene_debate_panel`, final `approve_claim` / `deny_claim` / `escalate_to_human` with **`[CONF: HIGH]`** or **`[CONF: MED]`** or **`[CONF: LOW]`**.

**Say:**

> "Right side: a **plain-English diary** of what the AI did, step by step — not a black box. Each line is an action you could imagine a junior analyst taking: *check this document*, *flag this inconsistency*, *call for a second opinion*. That’s the transparency insurers actually need."

### [1:05 – 1:15] Bottom-left — “LIVE METRICS”

**Visual:** **LIVE METRICS**: **Reward** (green number), **Calibration score**, **Declared confidence** pill (HIGH / MED / LOW), **Steps taken**. Optionally **CORRECT** badge when the outcome matches the scenario’s goal.

**Say:**

> "Numbers on the left aren’t magic scores for geeks — think of **reward** as ‘**did the behaviour we want just go up?**’ **Calibration** is ‘**did its confidence match reality?**’ High confidence on an easy honest claim — good. High confidence on a murky ring-fraud case — **bad**. The UI makes that visible in one glance."

### [1:15 – 1:25] “3×2 Calibration Matrix” — explain like a traffic light

**Visual:** The **3×2 Calibration Matrix** card. Point at **HIGH + Correct = +1** (highlighted), then **HIGH + Wrong = −0.8** (red warning).

**Say:**

> "This little grid is the **rulebook for confidence**. If you’re **right** and **appropriately sure**, you get the **best** score. If you’re **wrong** but you **acted like a genius** — that’s the **worst** cell: we penalise **cocky mistakes** harder than cautious ones. That single design choice is what teaches ‘**honest uncertainty**’ instead of fake confidence."

### [1:25 – 1:40] “Multi-Agent Court Panel” — two lawyers in software

**Visual:** First the **empty state** (“run contradictory claim to see…”). Switch dropdown to **contradictory claim**, **Run Episode**, scroll until **Court Panel Convened** — **Prosecutor (STRONG)** vs **Defender (WEAK)** and the **VERDICT** bar.

**Say:**

> "When the case is adversarial, the AI can **open a court** — not a gimmick, a **stress test**. One side argues *fraud from the evidence we found*; the other argues *innocent explanations still exist*. You see **strong** vs **weak** right on the card — then a **recommended action**. That’s how we stop one lazy headline from deciding someone’s claim."

### [1:40 – 1:55] Third scenario — humility pays

**Visual:** **distribution shift claim** → **Run Episode** → trace with **`query_linked_claim`**, **`flag_fraud_signal`**, final **`escalate_to_human [CONF: LOW]`**. LIVE METRICS showing **LOW** confidence and a solid **reward** (e.g. ~0.7).

**Say:**

> "Last trick: the fraud hides in **links between claims** — same broker, same pattern. The winning move isn’t bragging — it’s **raising your hand**: *human needed, I’m only low confidence*. ClaimCourt **rewards that humility**. In the real world, that’s fewer multi-crore mistakes."

---

## ACT 3 — “Yes, we actually trained it” (~20 s) — keep light

**Visual:** Quick montage: **WandB** project page (reward climbing) **or** `docs/reward_curve.svg` in GitHub; optional 1 clip of HF Jobs log line.

**Say (one breath, then slow on numbers):**

> "We didn’t just draw a pretty UI. We ran the AI through **five thousand** practice claims on cloud GPUs. The training score — think ‘**overall lesson learned**’ — went from about **zero-point-one-three** to **zero-point-four-seven**. On held-out checks, **decision accuracy** and **calibration** both went from **zero to perfect one-point-zero**. Under the hood that’s **reinforcement learning** with Hugging Face’s **TRL** library — same family of tech behind recent open reasoning models. The details are in our **README** and **mini-blog** for anyone who wants to dig."

**Numbers table (optional on-screen end card):**

| What we measure | Before training | After training |
|-----------------|-----------------|----------------|
| “Lesson learned” score (mean reward) | **0.13** | **0.47** |
| Decision matches the right action | **0%** | **100%** |
| Confidence matches reality | **0%** | **100%** |
| Catching fraud signals (partial credit) | **0%** | **33%** |

---

## ACT 4 — Close + try it (~15 s)

**Visual:** Full-screen end card. Large QR optional. Cursor hovers each line.

**Say:**

> "If you work in risk, ops, or policy — or you’re just curious — **open ClaimCourt**, pick **contradictory claim**, hit **Run Episode**, and watch the trace and the court panel. If you’re a builder, everything is on **GitHub** under the codename **debatefloor** — links in the description. **Try one case** — that’s all it takes to see why this matters. Thank you."

**Links (read slowly or show as text):**

- **Try it:** https://huggingface.co/spaces/AniketAsla/debatefloor  
- **Code:** https://github.com/AniketAslaliya/debateFloor  
- **Mini-blog (markdown):** https://huggingface.co/spaces/AniketAsla/debatefloor/blob/main/BLOG.md  
- **Weights & Biases (all training runs):** https://wandb.ai/aniketaslaliya-lnmiit/debatefloor-insurance-rl  

---

## UI → script mapping (checklist so nothing is missing)

| UI element | Act / time | Covered? |
|------------|------------|----------|
| ClaimCourt header + Space chrome | Act 1 | ✓ |
| Run an Episode + **task dropdown** (3 tasks) | Act 2 | ✓ |
| Task description + **Run Episode** + **CORRECT** | Act 2 | ✓ |
| **Claim Under Investigation** (ID, claimant, docs) | Act 2 | ✓ |
| **agent-trace.log** (steps, CONF tags) | Act 2 | ✓ |
| **LIVE METRICS** (Reward, Calibration, Confidence, Steps) | Act 2 | ✓ |
| **3×2 Calibration Matrix** | Act 2 | ✓ |
| **Multi-Agent Court Panel** (empty + **live debate + verdict**) | Act 2 | ✓ |
| **distribution_shift** + linked claims + **LOW** + reward | Act 2 | ✓ |
| Training proof + numbers | Act 3 | ✓ |
| Links + “try one case” CTA | Act 4 | ✓ |

---

## Optional segments (if you have +15 s)

- **Split screen (technical viewers only):** Space left, `app/main.py` `/step` right — *“same server answers the demo and the training job.”* Skip for a general audience.  
- **JSON overlay (2 s):** tiny corner: request/response — proves it’s not canned video.

---

## Production checklist

| # | Do this | Why |
|---|---------|-----|
| 1 | Rehearse **one full run** per task so clicks are smooth | Saves retakes |
| 2 | **1080p**, clear browser zoom (~110%) | Readable on phones |
| 3 | **Yellow cursor** in OBS | Viewers follow the story |
| 4 | **No facecam** needed | Keeps focus on product |
| 5 | Export **YouTube** as public URL; **no** huge video in HF repo | Matches hackathon rules |
| 6 | **1.5 s** title card: `ClaimCourt — OpenEnv Hackathon India 2026` | Brand + context |

---

## Jargon one-liners (if you use a term, follow with this)

| Term | One-liner for family & friends |
|------|--------------------------------|
| OpenEnv | “A standard way to package ‘**AI + environment + rules**’ so researchers can compare apples to apples.” |
| GRPO / TRL | “**Practice + score + repeat** — like flight simulators for pilots, but for language models.” |
| Reward | “**Did we like that behaviour?** — summed up as a number.” |
| Calibration | “**Was its confidence honest** — not just lucky?” |

---

## Canonical stats (technical backup — same as repo JSON)

Source: `reports/training_summary.json`, `reports/component_shift_summary.json` — Qwen2.5-0.5B-Instruct, 5k episodes, 2500 GRPO steps, ~3h on L4.

| Metric | Before | After |
|--------|--------|-------|
| Mean training reward | 0.130 | 0.469 |
| Decision accuracy (eval) | 0.00 | 1.00 |
| Calibration (eval) | 0.00 | 1.00 |
| Fraud detection (eval) | 0.00 | 0.33 |
| Final train loss | — | ~0.00565 |
