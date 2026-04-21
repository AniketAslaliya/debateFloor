# CONTEXT.md — Living Session Log
## DebateFloor | Updated after every coding session

---

## HOW TO USE THIS FILE

After EVERY coding session, Claude Code updates this file with:
1. What was built/changed
2. What's broken or pending
3. Current score estimate
4. Next session's first task

**First thing every new session:** Read this file before touching any code.
**Last thing every session:** Ask Claude Code to update this file.

---

## 📍 CURRENT STATUS
<!-- Claude Code updates this block after every session -->

**Last updated:** [Session 0 — Project Init]
**Current branch:** main
**HF Space status:** NOT DEPLOYED (Round 1 space exists, needs fork)
**Colab notebook:** NOT CREATED
**HF Blog:** NOT PUBLISHED
**Validation status:** NOT RUN

**Estimated score:** 38/100

---

## ✅ COMPLETED WORK
<!-- Add items here as they're done -->

- [x] Round 1: insuranceClaim env built and validated (11 runs, Phase 1+2 passed)
- [x] Architecture decision: DebateFloor = Round 1 + calibration grader + procedural generator
- [x] All documentation files created (CLAUDE.md, CONTEXT.md, SKILL.md, ROADMAP.md)

---

## 🔄 IN PROGRESS
<!-- Currently being worked on -->

- [ ] Forking Round 1 repo into debatefloor
- [ ] Designing 5 fraud type templates (Mitali)
- [ ] Verifying TRL + Unsloth on Colab T4 (Aditya)

---

## ❌ KNOWN ISSUES / BLOCKERS
<!-- Problems found, to be fixed -->

*None yet — project starting*

---

## 📁 FILE STATE
<!-- Current state of each key file -->

| File | Status | Last Modified | Notes |
|------|--------|---------------|-------|
| openenv.yaml | 🔴 NOT CREATED | — | Needs confidence_required + concurrent support |
| app/main.py | 🟡 INHERITED | Round 1 | Minor additions needed |
| server/insurance_env.py | 🟡 INHERITED | Round 1 | Add confidence param to terminal actions |
| server/claim_generator.py | 🔴 NOT CREATED | — | HIGHEST PRIORITY |
| server/calibration_grader.py | 🔴 NOT CREATED | — | CORE INNOVATION |
| models.py | 🟡 INHERITED | Round 1 | Add confidence field to ClaimAction |
| inference_debatefloor.py | 🔴 NOT CREATED | — | Mandatory deliverable |
| train/train_debatefloor.ipynb | 🔴 NOT CREATED | — | Mandatory deliverable |
| Dockerfile | 🟡 INHERITED | Round 1 | Should work unchanged |
| tests/ | 🔴 NOT CREATED | — | Write after core files done |

---

## 🧪 TEST RESULTS
<!-- Run: pytest tests/ -v and paste results here -->

*No tests run yet*

---

## 📈 REWARD CURVE STATUS
<!-- After first training run, paste curve description here -->

*No training run yet*
- Baseline (untrained) calibration score: NOT MEASURED
- Post-training calibration score: NOT MEASURED
- Confidence distribution before training: NOT MEASURED
- Confidence distribution after training: NOT MEASURED

---

## 💬 SESSION LOG

### Session 0 — April 19, 2026 (Project Init)
**What happened:**
- Created all documentation files
- Decided on DebateFloor architecture
- Locked problem statement: Theme 3.1 + calibrated uncertainty innovation
- Assigned roles: Aniket (env core), Mitali (domain + grader), Aditya (training pipeline)

**Decisions made:**
- Use Round 1 insurance domain (don't switch domains with 5 days left)
- Split training reward (simple) from evaluation reward (complex) — never mix
- Target: claim_generator.py as Day 1 priority

**Next session first task:**
```
Read CONTEXT.md and CLAUDE.md. Then build server/claim_generator.py 
from scratch. It must accept (seed, fraud_type, coverage_type, difficulty) 
and return a ClaimScenario object. Use the 5 fraud types defined in CLAUDE.md.
Target: 500+ unique episodes from seed variation.
```

---

### Session 1 — [DATE]
**What happened:**
*[Claude Code fills this in]*

**Files changed:**
*[Claude Code fills this in]*

**Tests passing:**
*[Claude Code fills this in]*

**Current score estimate:**
*[Claude Code fills this in]*

**Next session first task:**
*[Claude Code fills this in]*

---

### Session 2 — [DATE]
*[Template repeats]*

---

## 🎯 MILESTONE TRACKER

| Milestone | Target Date | Status | Score Impact |
|-----------|------------|--------|-------------|
| claim_generator.py working | Day 2 | 🔴 | +12 pts |
| calibration_grader.py working | Day 2 | 🔴 | +8 pts |
| HF Space deployed + validated | Day 3 | 🔴 | +4 pts |
| Colab training run complete | Day 3 | 🔴 | +14 pts |
| HF blog published | Day 4 | 🔴 | +6 pts |
| Pitch rehearsed 3× | Day 4 | 🔴 | +3 pts |
| All extras complete | Day 5 | 🔴 | +8 pts |