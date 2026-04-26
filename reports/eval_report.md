# Evaluation Report

Generated at: 2026-04-25T18:12:09.069260+00:00
Base URL: http://localhost:7860
Tasks: clean_claim, contradictory_claim, coordinated_fraud, distribution_shift_claim, identity_fraud
Seeds: 7, 11, 13, 19, 25
Distinct variant_ids: [0, 1, 2, 3, 4]

| Task | Seed | Variant | Steps | Done | Reward | Evidence Quality | Exploit Penalty |
|---|---:|---:|---:|:---:|---:|---:|---:|
| clean_claim | 7 | 2 | 4 | yes | 0.8725 | 1.0000 | 0.0000 |
| clean_claim | 11 | 1 | 4 | yes | 0.8725 | 1.0000 | 0.0000 |
| clean_claim | 13 | 3 | 4 | yes | 0.8725 | 1.0000 | 0.0000 |
| clean_claim | 19 | 4 | 4 | yes | 0.8725 | 1.0000 | 0.0000 |
| clean_claim | 25 | 0 | 4 | yes | 0.8725 | 1.0000 | 0.0000 |
| contradictory_claim | 7 | 2 | 8 | yes | 0.7497 | 1.0000 | 0.0000 |
| contradictory_claim | 11 | 1 | 8 | yes | 0.7497 | 1.0000 | 0.0000 |
| contradictory_claim | 13 | 3 | 8 | yes | 0.7497 | 1.0000 | 0.0000 |
| contradictory_claim | 19 | 4 | 8 | yes | 0.7497 | 1.0000 | 0.0000 |
| contradictory_claim | 25 | 0 | 8 | yes | 0.7497 | 1.0000 | 0.0000 |
| coordinated_fraud | 7 | 2 | 12 | yes | 0.8230 | 1.0000 | 0.0000 |
| coordinated_fraud | 11 | 1 | 12 | yes | 0.8230 | 1.0000 | 0.0000 |
| coordinated_fraud | 13 | 3 | 12 | yes | 0.8230 | 1.0000 | 0.0000 |
| coordinated_fraud | 19 | 4 | 12 | yes | 0.8230 | 1.0000 | 0.0000 |
| coordinated_fraud | 25 | 0 | 12 | yes | 0.8230 | 1.0000 | 0.0000 |
| distribution_shift_claim | 7 | 2 | 12 | yes | 0.7827 | 1.0000 | 0.0000 |
| distribution_shift_claim | 11 | 1 | 12 | yes | 0.7827 | 1.0000 | 0.0000 |
| distribution_shift_claim | 13 | 3 | 12 | yes | 0.7827 | 1.0000 | 0.0000 |
| distribution_shift_claim | 19 | 4 | 12 | yes | 0.7827 | 1.0000 | 0.0000 |
| distribution_shift_claim | 25 | 0 | 12 | yes | 0.7827 | 1.0000 | 0.0000 |
| identity_fraud | 7 | 2 | 10 | yes | 0.8180 | 1.0000 | 0.0000 |
| identity_fraud | 11 | 1 | 10 | yes | 0.8180 | 1.0000 | 0.0000 |
| identity_fraud | 13 | 3 | 10 | yes | 0.8180 | 1.0000 | 0.0000 |
| identity_fraud | 19 | 4 | 10 | yes | 0.8180 | 1.0000 | 0.0000 |
| identity_fraud | 25 | 0 | 10 | yes | 0.8180 | 1.0000 | 0.0000 |

Average Reward: 0.8092
Completion Rate: 100.00%

> **Note on identical rewards within a task.** Each task above shows the same
> reward across all 5 seeds because this run is a *scripted baseline* — the
> evaluation client follows a fixed strategy per `task_id` (e.g. always call
> `validate_document` then `flag_fraud_signal` then `deny_claim` for
> `contradictory_claim`). The seeds vary the generated documents (different
> claimants, amounts, fraud-signal strengths — see the `Variant` column), but
> the scripted strategy is invariant to that variation, so the env returns the
> same scalar reward every time. This is intentional: the scripted baseline
> exists to demonstrate that the environment's reward surface is deterministic
> and reproducible across seeds, *not* to show learning.
>
> The trained GRPO model produces variable rewards across seeds — see the
> held-out component-shift evaluation in
> [`reports/component_shift_summary.json`](component_shift_summary.json) and
> the README "Held-out evaluation" section.
