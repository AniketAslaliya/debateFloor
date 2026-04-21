# SKILL.md — Token-Efficient Patterns for DebateFloor
## Read this before any code generation task

---

## WHY THIS FILE EXISTS

Context management is the primary failure mode in Claude Code.
This file gives Claude Code reusable patterns so it never generates
boilerplate from scratch, uses fewer tokens per session, and produces
consistent, correct code every time.

---

## SKILL 1: Generate a ClaimScenario (procedural generator pattern)

**When to use:** Any time claim_generator.py needs a new fraud type template

**Pattern:**
```python
# ALWAYS follow this exact structure for new fraud types
TEMPLATE_{FRAUD_TYPE}_{COVERAGE} = {
    "claim_id_prefix": "CLM",
    "payout": {base_amount_in_rupees},
    "decision": "deny_claim",  # or "approve_claim" or "escalate"
    "documents": [
        {
            "id": "DOC-01",
            "type": "{document_type}",
            "content": "{content_with_fraud_signal}",
            "is_fraudulent": True/False
        }
    ],
    "fraud_signals": ["{signal_1}", "{signal_2}"],
    "ambiguity_score": 0.0-1.0,  # 0=obvious, 1=very ambiguous
    "linked_claims": []  # only for coordinated_ring type
}
```

**Do NOT:** Hardcode payout amounts. Always use `base * rng.uniform(0.8, 1.2)`.

---

## SKILL 2: Write a grader function

**When to use:** Creating or modifying any grader in the environment

**Pattern:**
```python
def grade_{task_name}(observation: ClaimObservation,
                       action: ClaimAction,
                       ground_truth: str) -> float:
    """
    Returns float in [0.0, 1.0].
    Always clamp output: return max(0.0, min(1.0, score))
    Always document what earns full score and what earns zero.
    """
    score = 0.0
    
    # Check 1: [what you're checking]
    if {condition}:
        score += {weight}
    
    # Check 2: [what you're checking]
    if {condition}:
        score += {weight}
    
    return max(0.0, min(1.0, score))
```

**NEVER:** Return values outside [0.0, 1.0]. Always clamp.
**NEVER:** Use LLM calls inside a grader — graders must be deterministic.

---

## SKILL 3: Write a FastAPI endpoint

**When to use:** Adding new endpoints to app/main.py

**Pattern:**
```python
@app.post("/{endpoint_name}", response_model={ResponseModel})
async def {endpoint_name}({params}) -> {ResponseModel}:
    """One-line docstring."""
    try:
        result = env.{method}({params})
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
```

**Always:** Include try/except. Always return typed response model.

---

## SKILL 4: Write the [START]/[STEP]/[END] stdout format

**When to use:** Any time inference_debatefloor.py is edited

**Exact format — do not deviate:**
```python
# At episode start:
print(f"[START] task={task_id} env=debatefloor model={model_name} "
      f"confidence_required=true episode={episode_num}")

# At each step:
print(f"[STEP] step={step_num} action={action_name} "
      f"reward={reward:.4f} confidence={confidence} "
      f"done={done} error={error}")

# At episode end:
print(f"[END] success={success} steps={steps} "
      f"total_reward={total:.4f} calibration_score={calib:.4f} "
      f"decision={decision}")
```

**CRITICAL:** Field names must be exact. Any deviation = evaluation failure.

---

## SKILL 5: Add a test

**When to use:** After creating any new function

**Pattern:**
```python
import pytest
from server.{module} import {function}

class Test{FunctionName}:
    def test_{happy_path}(self):
        """Describe what correct behaviour looks like."""
        result = {function}({valid_inputs})
        assert result == {expected}
        assert 0.0 <= result <= 1.0  # always check bounds for rewards
    
    def test_{edge_case}(self):
        """Describe the edge case."""
        result = {function}({edge_inputs})
        assert result == {expected}
    
    def test_anti_gaming(self):
        """Verify systematic LOW confidence is penalised."""
        history = [{"confidence": "LOW"}] * 15  # 100% LOW rate
        result = calibration_reward("deny", "LOW", "deny", history)
        assert result < 0  # gaming penalty must fire
```

---

## SKILL 6: Update CONTEXT.md after a session

**When to use:** End of every Claude Code session

**Prompt to give Claude Code:**
```
Update docs/CONTEXT.md with this session's work:
1. List every file you created or modified
2. List any tests that are passing
3. List any known issues or broken things
4. Update the current score estimate
5. Write the exact first task for next session as a code block
6. Add a new session entry to the Session Log
```

---

## SKILL 7: Debug a failing validation

**When to use:** pre_validation_script.py returns errors

**Checklist in order:**
```
1. Docker build failure?
   → Check requirements.txt for missing packages
   → Check Dockerfile CMD matches uvicorn command

2. /health returns non-200?
   → Check app/main.py has @app.get("/health")
   → Check port is 7860 in Dockerfile

3. Task grader returns outside [0.0, 1.0]?
   → Check all graders have max(0.0, min(1.0, score)) clamp

4. Concurrent sessions failing?
   → Check openenv.yaml has supports_concurrent_sessions: true
   → Check environment class is stateless per session (no shared state)

5. Inference script timing out (>20 min)?
   → Reduce max_steps in task config
   → Add timeout per step in inference script
```

---

## SKILL 8: Generate the training reward function

**When to use:** Any time training reward needs modification

**Rule:** Simple. Single scalar. No compound penalties.
```python
def training_reward(obs, action, ground_truth, confidence,
                    legitimate_flags, step_num, done):
    """Simple shaped reward for GRPO stability."""
    r = -0.05  # step penalty always
    
    if done:
        correct = (action == ground_truth)
        # Decision accuracy component
        r += 1.0 if correct else -0.5
        # Legitimate fraud flags component
        r += 0.3 * min(legitimate_flags, 3)  # cap at 3 flags
        # Calibration component (weighted 50%)
        calib_matrix = {
            ("HIGH", True): 1.0, ("HIGH", False): -0.8,
            ("MED", True): 0.6,  ("MED", False): -0.2,
            ("LOW", True): 0.1,  ("LOW", False): 0.0,
        }
        if confidence:
            r += 0.5 * calib_matrix.get((confidence, correct), 0.0)
    
    return float(r)
```

---

## TOKEN-SAVING SHORTCUTS

When asking Claude Code to build something, use these shorthand prompts:

| Shorthand | Means |
|-----------|-------|
| "Use SKILL 1" | Follow the ClaimScenario template pattern |
| "Use SKILL 2" | Follow the grader function pattern |
| "Use SKILL 4" | Produce correct [START]/[STEP]/[END] format |
| "Add SKILL 5 tests" | Write tests following the test pattern |
| "End session, use SKILL 6" | Update CONTEXT.md |
| "Debug validation, use SKILL 7" | Follow the debug checklist |

---

## COMMON MISTAKES TO AVOID

```
❌ Mixing training and evaluation reward functions
❌ Graders that call an LLM (must be deterministic)
❌ Terminal actions without confidence field
❌ Hardcoded payout amounts
❌ Forgetting concurrent session support in openenv.yaml
❌ stdout format with wrong field names
❌ Pushing without running pre_validation_script.py
❌ Using pip without --break-system-packages
```