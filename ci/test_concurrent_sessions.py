"""Assert that two concurrent /reset calls produce isolated sessions."""
import threading, requests

BASE = "http://localhost:7860"
results = {}

def run_reset(task, key):
    r = requests.post(f"{BASE}/reset", json={"task_id": task, "seed": 0})
    r.raise_for_status()
    results[key] = r.json()["observation"]["task_id"]

t1 = threading.Thread(target=run_reset, args=("clean_claim",       "A"))
t2 = threading.Thread(target=run_reset, args=("coordinated_fraud", "B"))
t1.start(); t2.start(); t1.join(); t2.join()

assert results["A"] == "clean_claim",       f"Got {results['A']}"
assert results["B"] == "coordinated_fraud", f"Got {results['B']}"
print("Concurrent sessions isolated  PASS")
