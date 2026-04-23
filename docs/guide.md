# STUCK_GUIDE.md — When You're Blocked, Use This
## Every problem has a solution path. Follow it.

---

## RULE 1: Before asking Claude Code anything when stuck

1. Read the error message completely (not just the last line)
2. Check CONTEXT.md — has this broken before?
3. Check SKILL.md SKILL 7 — is there a pattern for this?
4. Then ask Claude Code with this format:

```
I'm stuck on [problem].
Error: [full error message]
File: [which file]
What I tried: [what you already tried]
Read SKILL 7 and tell me what caused this before suggesting a fix.
```

---

## PROBLEM 1: Docker build fails

**Symptoms:** `docker build` returns error

**Decision tree:**
```
Error mentions "module not found"?
  → Add module to requirements.txt
  → Rebuild

Error mentions "port already in use"?
  → docker ps, kill existing container
  → docker run again

Error mentions "permission denied"?
  → Check Dockerfile has correct file permissions
  → Make sure WORKDIR is /app

Error is cryptic Python traceback?
  → Run the server locally first: uvicorn app.main:app --port 7860
  → Fix the Python error first, then rebuild Docker
```

**Fast fix command:**
```bash
docker build -t debatefloor . 2>&1 | tail -50
# Read the last 50 lines of build output carefully
```

---

## PROBLEM 2: HF Space returns non-200 on /health

**Symptoms:** pre_validation_script.py says "HF Space ping failed"

**Decision tree:**
```
Space is building?
  → Wait 3 minutes, try again

Space shows error in HF logs?
  → Click "Logs" tab on HF Space page
  → Find the Python traceback
  → Fix it locally first, push again

/health endpoint missing?
  → Add to app/main.py:
    @app.get("/health")
    async def health():
        return {"status": "healthy"}

Port mismatch?
  → Verify Dockerfile: CMD [..., "--port", "7860"]
  → HF Spaces always uses port 7860
```

---

## PROBLEM 3: Grader returns value outside [0.0, 1.0]

**Symptoms:** Validation script says "grader score out of range"

**Fix — always clamp:**
```python
# Find the return statement in your grader
# Replace: return score
# With:    return max(0.0, min(1.0, score))
```

**Prevention:** Every grader must have this. No exceptions.

---

## PROBLEM 4: Concurrent sessions failing

**Symptoms:** Two parallel /reset calls return same episode, or crash

**Root cause:** Shared mutable state in environment class

**Fix:**
```python
# WRONG — shared state breaks concurrent sessions
class InsuranceEnv:
    def __init__(self):
        self.current_claim = None  # shared across sessions!

# RIGHT — stateless per request using session_id
class InsuranceEnv:
    def __init__(self):
        self.sessions = {}  # keyed by session_id
    
    def reset(self, session_id: str):
        self.sessions[session_id] = generate_claim(seed=random.randint(...))
        return self.sessions[session_id]
```

**openenv.yaml must have:**
```yaml
supports_concurrent_sessions: true
max_concurrent_envs: 64
```

**Build order:** bootstrap the OpenEnv skeleton first, then fill in behavior.

1. Define action, observation, and state.
2. Implement `reset()` and `step()`.
3. Expose the environment through FastAPI.
4. Keep trainer logic outside the environment.

---

## PROBLEM 5: Inference script times out (>20 min)

**Symptoms:** Evaluation fails with timeout

**Quick fixes in order:**
1. Reduce `MAX_STEPS` in task config (try 5 for testing)
2. Add step timeout in inference script:
   ```python
   import signal
   signal.alarm(30)  # 30 second timeout per LLM call
   ```
3. Use a faster model for testing (gpt-4o-mini instead of gpt-4o)
4. Reduce number of tasks run per inference call

---

## PROBLEM 6: GRPO training curve is completely flat

**Symptoms:** WandB shows reward hovering around -0.05 with no improvement

**Decision tree:**
```
Are you using the EVALUATION reward (complex)?
  → Switch to TRAINING reward (simple scalar)
  → See SKILL 8 in SKILL.md

Is the model getting ANY positive reward on any episode?
  → Run 10 episodes, print raw reward values
  → If all negative: model never explores correct actions
  → Fix: add a hint in the system prompt to guide initial exploration

Is the batch size too small?
  → Increase per_device_train_batch_size to 8
  → More episodes per gradient update = more stable signal

Is learning rate too high?
  → Reduce from 5e-6 to 1e-6
  → High LR causes oscillation, not learning
```

**If still flat after all above:**
→ Show confidence distribution shift instead of reward curve
→ A distribution shift from 85% HIGH to 45% HIGH is compelling evidence
→ Judges understand behaviour change better than loss curves

**Reward design rule:** prefer multiple independent checks over one fragile score.

- execution success
- correctness
- format compliance
- timeouts
- resource usage
- safety constraints
- anti-cheating checks

---

## PROBLEM 7: Claude Code context gets noisy / confused

**Symptoms:** Claude Code starts making mistakes, forgetting the architecture, suggesting things that contradict CLAUDE.md

**Fix:**
```
1. Run /clear in Claude Code
2. Paste this:
   "Read CLAUDE.md and docs/CONTEXT.md. 
    Tell me the project status in 3 sentences.
    Do not write any code yet."
3. Verify Claude Code understands the project correctly
4. Then give the specific task
```

**Prevention:** Use /clear at the start of every new session.
Context degradation is the primary failure mode in Claude Code. Aggressive /clear usage is the most effective fix.

---

## PROBLEM 8: [START]/[STEP]/[END] format failing evaluation

**Symptoms:** Evaluation script reports incorrect scoring

**Exact format — character for character:**
```python
# START line
print(f"[START] task={task_id} env=debatefloor model={model_name} confidence_required=true episode={episode_num}")

# STEP line — note: reward has 4 decimal places
print(f"[STEP] step={step_num} action={action_name} reward={reward:.4f} confidence={confidence} done={done} error={error}")

# END line
print(f"[END] success={success} steps={steps} total_reward={total:.4f} calibration_score={calib:.4f} decision={decision}")
```

**Common mistakes:**
- `done=True` should be Python bool printed as `True` not `true`
- `error=None` should print as `None` not `null`
- `reward` must have exactly 4 decimal places
- Field order must match exactly

---

## PROBLEM 9: GitHub push rejected

```bash
# If push fails due to untracked changes:
git status          # see what's changed
git add -A          # stage everything
git commit -m "feat: describe what changed"
git push origin main

# If push fails due to diverged history:
git pull --rebase origin main
git push origin main

# If push fails due to large files:
# Add to .gitignore: *.bin, *.pt, checkpoints/
```

---

## ESCALATION PATH

If none of the above work, in order:
1. **Discord** — Join the hackathon finalists Discord (link in email)
2. **WhatsApp** — Chat link in the selection email
3. **Email** — help_openenvhackathon@scaler.com
4. **OpenEnv GitHub Issues** — github.com/openenv/openenv/issues

**When escalating, always share:**
- Full error message
- Which file is broken
- What you've already tried
- Link to your HF Space

---

## MENTAL HEALTH NOTE

If you've been stuck for more than 30 minutes:
1. Stop. Take a 10-minute break.
2. Come back and read the error message again — fresh eyes find the issue.
3. Rubber duck debug: explain the problem out loud to a teammate.
4. The problem is almost never what you think it is. It's usually something simpler.

You made it to the top 800 teams from 52,000. You know how to solve hard problems. This is just another one.