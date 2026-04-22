# DebateFloor HTTP Rollout Evaluation

Base URL: `https://aniketasla-debatefloor.hf.space`

| Policy | Episodes | Mean reward | Mean calibration | Success rate |
|---|---:|---:|---:|---:|
| naive_high_no_investigation | 3 | 0.267 | -0.200 | 33.33% |
| calibrated_scripted_investigator | 3 | 0.554 | 0.567 | 100.00% |

## Per-Episode Rows

| Policy | Task | Seed | Reward | Calibration | Confidence | Steps |
|---|---|---:|---:|---:|---|---:|
| naive_high_no_investigation | clean_claim | 42 | 0.800 | 1.0 | HIGH | 1 |
| naive_high_no_investigation | contradictory_claim | 42 | 0.000 | -0.8 | HIGH | 1 |
| naive_high_no_investigation | distribution_shift_claim | 42 | 0.000 | -0.8 | HIGH | 1 |
| calibrated_scripted_investigator | clean_claim | 42 | 0.762 | 1.0 | HIGH | 4 |
| calibrated_scripted_investigator | contradictory_claim | 42 | 0.547 | 0.6 | MED | 7 |
| calibrated_scripted_investigator | distribution_shift_claim | 42 | 0.352 | 0.1 | LOW | 8 |
