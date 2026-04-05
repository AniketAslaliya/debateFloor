"""Verify environment imports and step-0 reward is 0.0 for all tasks."""
from app.environment import InsuranceClaimEnvironment

env = InsuranceClaimEnvironment()
for task in ["clean_claim", "contradictory_claim", "coordinated_fraud"]:
    obs = env.reset(task_id=task, seed=42)
    assert obs.reward == 0.0, f"{task} step-0 reward must be 0.0, got {obs.reward}"
    print(f"  {task}: step-0 reward={obs.reward}  PASS")
print("All task resets OK")
