from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from .models import (
    ClaimStatus,
    InsuranceClaimAction,
    InsuranceClaimObservation,
    InsuranceClaimState,
)
from .tasks import (
    TASKS,
    RuntimeTask,
    build_runtime_task,
    build_initial_payload,
    compute_reward_breakdown,
    get_evidence_keyword_hints,
    get_task_definition,
)


class InsuranceClaimEnvironment(Environment[InsuranceClaimAction, InsuranceClaimObservation, InsuranceClaimState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # NOW ACTUALLY TRUE - session-managed via main.py

    def __init__(self):
        self._state = InsuranceClaimState(episode_id=str(uuid4()), step_count=0)
        self._payload: Dict[str, Any] = {}
        self._action_history: List[Dict[str, Any]] = []
        self._flags_raised: List[str] = []
        self._found_signals: List[str] = []
        self._false_flags: int = 0
        self._investigation_targets: List[str] = []
        self._evidence_hits: int = 0
        self._evidence_total: int = 0
        self._exploit_penalty: float = 0.0
        self._request_info_streak: int = 0
        self._last_progress_step: int = 0
        self._runtime_task: RuntimeTask | None = None
        self._last_message = "Environment initialized"
        self._queried_claims: set[str] = set()
        self._visible_linked_claims: list = []

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None, episode_id: Optional[str] = None) -> InsuranceClaimObservation:
        selected_task = task_id or "clean_claim"
        task = build_runtime_task(selected_task, seed=seed)
        self._runtime_task = task

        self._payload = build_initial_payload(task)
        self._action_history = []
        self._flags_raised = []
        self._found_signals = []
        self._false_flags = 0
        self._investigation_targets = []
        self._evidence_hits = 0
        self._evidence_total = 0
        self._exploit_penalty = 0.0
        self._request_info_streak = 0
        self._last_progress_step = 0
        self._queried_claims = set()
        self._visible_linked_claims = deepcopy(self._payload.get("linked_claims", []))
        self._last_message = (
            f"Task '{task.task_id}' loaded (variant={task.variant_id}). Start investigation."
        )

        self._state = InsuranceClaimState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            claim_id=task.claim_id,
            step_number=0,
            max_steps=task.max_steps,
            status=ClaimStatus.OPEN,
            flags_raised=[],
            found_signals=[],
            penalty_total=0.0,
            done=False,
            last_action_error=None,
            payout_estimate_inr=None,
            final_decision=None,
            final_score=0.0,
        )
        return self._build_observation(message=self._last_message)

    def step(self, action: InsuranceClaimAction) -> InsuranceClaimObservation:
        if self._state.task_id == "":
            return self.reset(task_id="clean_claim")

        if self._state.done:
            return self._build_observation(message="Episode already complete. Call reset() to start a new episode.")

        self._state.step_count += 1
        self._state.step_number += 1
        self._state.status = ClaimStatus.INVESTIGATING
        self._state.last_action_error = None

        try:
            message = self._apply_action(action)
            self._last_message = message
        except ValueError as exc:
            self._state.last_action_error = str(exc)
            self._state.penalty_total += 0.05
            self._last_message = f"Invalid action: {exc}"

        self._action_history.append(
            {
                "step": self._state.step_number,
                "action_type": action.action_type,
                "parameters": deepcopy(action.parameters),
                "reasoning": action.reasoning,
            }
        )

        if not self._state.done and (self._state.step_number - self._last_progress_step) >= 4:
            self._exploit_penalty += 0.01

        if self._state.step_number >= self._state.max_steps and not self._state.done:
            self._state.done = True
            self._state.status = ClaimStatus.CLOSED
            self._last_message = "Max steps reached before final adjudication. Episode closed."

        observation = self._build_observation(message=self._last_message)
        self._state.final_score = float(observation.reward)
        return observation

    @property
    def state(self) -> InsuranceClaimState:
        return self._state

    def _apply_action(self, action: InsuranceClaimAction) -> str:
        task = self._runtime_task or build_runtime_task(self._state.task_id)

        if action.action_type == "request_information":
            self._request_info_streak += 1
            if self._request_info_streak > 2:
                self._exploit_penalty += 0.03
            self._state.penalty_total += 0.02
            return "Additional information requested. Useful but consumes time and SLA budget."

        self._request_info_streak = 0

        if action.action_type == "validate_document":
            doc_id = str(action.parameters.get("doc_id", "")).strip()
            if not doc_id:
                raise ValueError("'doc_id' is required for validate_document")

            doc = next((d for d in self._payload["documents"] if d.get("doc_id") == doc_id), None)
            if doc is None:
                raise ValueError(f"Unknown doc_id '{doc_id}'")

            discovered = self._discover_signals_from_document(doc_id, task.task_id)
            if discovered:
                for signal in discovered:
                    if signal not in self._found_signals:
                        self._found_signals.append(signal)
                self._last_progress_step = self._state.step_number
                return f"Validated {doc_id}. Potential inconsistencies detected: {', '.join(discovered)}"
            return f"Validated {doc_id}. No direct inconsistency detected."

        if action.action_type == "flag_fraud_signal":
            flag_id = str(action.parameters.get("flag_id", "")).strip()
            evidence = str(action.parameters.get("evidence", "")).strip()
            if not flag_id:
                raise ValueError("'flag_id' is required for flag_fraud_signal")
            if not evidence:
                raise ValueError("'evidence' is required for flag_fraud_signal")

            if flag_id in self._flags_raised:
                self._exploit_penalty += 0.05

            if flag_id not in self._flags_raised:
                self._flags_raised.append(flag_id)

            self._evidence_total += 1

            if flag_id in task.expected_signals:
                hints = get_evidence_keyword_hints(task.task_id, flag_id)
                evidence_lc = evidence.lower()
                if not hints or any(h in evidence_lc for h in hints):
                    self._evidence_hits += 1
                else:
                    self._exploit_penalty += 0.02

                if flag_id not in self._found_signals:
                    self._found_signals.append(flag_id)
                self._last_progress_step = self._state.step_number
                return f"Fraud signal '{flag_id}' logged with evidence."

            self._false_flags += 1
            return f"Fraud signal '{flag_id}' logged, but does not match ground-truth indicators."

        if action.action_type == "estimate_payout":
            amount = action.parameters.get("amount_inr")
            if amount is None:
                raise ValueError("'amount_inr' is required for estimate_payout")
            try:
                payout = float(amount)
            except (TypeError, ValueError) as exc:
                raise ValueError("'amount_inr' must be numeric") from exc
            self._state.payout_estimate_inr = payout
            return f"Payout estimate set to INR {payout:.2f}."

        if action.action_type == "query_linked_claim":
            claim_id = str(action.parameters.get("claim_id", "")).strip()
            if not claim_id:
                raise ValueError("'claim_id' is required for query_linked_claim")
            full_linked = self._payload.get("_full_linked_claims", self._payload.get("linked_claims", []))
            match = next((c for c in full_linked if c.get("claim_id") == claim_id), None)
            if match is None:
                raise ValueError(f"Linked claim '{claim_id}' not found")
            # Reveal full detail in the visible linked claims list for this session
            already_visible = any(
                c.get("claim_id") == claim_id and len(c) > 2
                for c in self._visible_linked_claims
            )
            if not already_visible:
                self._visible_linked_claims = [
                    deepcopy(match) if c.get("claim_id") == claim_id else c
                    for c in self._visible_linked_claims
                ]
            self._queried_claims.add(claim_id)
            self._last_progress_step = self._state.step_number
            return f"Linked claim detail retrieved for {claim_id}: {match}"

        if action.action_type in {"approve_claim", "deny_claim", "request_investigation"}:
            self._state.final_decision = action.action_type
            self._state.done = True
            self._state.status = ClaimStatus.DECIDED

            if action.action_type == "request_investigation":
                targets = action.parameters.get("target_claim_ids", [])
                if isinstance(targets, list):
                    self._investigation_targets = [str(t) for t in targets]
                else:
                    raise ValueError("'target_claim_ids' must be a list for request_investigation")

            reason = str(action.parameters.get("reason", "")).strip()
            if not reason and action.action_type != "approve_claim":
                self._state.penalty_total += 0.03

            self._state.status = ClaimStatus.CLOSED
            return f"Final decision submitted: {action.action_type}."

        raise ValueError(f"Unsupported action_type '{action.action_type}'")

    def _discover_signals_from_document(self, doc_id: str, task_id: str) -> List[str]:
        if task_id == "clean_claim":
            return []

        mapping: Dict[str, Dict[str, List[str]]] = {
            "contradictory_claim": {
                "DOC-10": ["date_mismatch"],
                "DOC-11": ["date_mismatch"],
                "DOC-12": ["cost_inflation"],
                "DOC-13": ["signature_mismatch"],
            },
            "coordinated_fraud": {
                "DOC-21": ["shared_repair_shop_far"],
                "DOC-22": ["near_identical_descriptions"],
                "DOC-23": ["recent_policy_cluster"],
            },
        }
        signal_map = mapping.get(task_id, {})
        signals = list(signal_map.get(doc_id, []))

        if task_id == "coordinated_fraud" and doc_id in {"DOC-21", "DOC-22", "DOC-23"}:
            # This cross-claim inconsistency is visible across linked claim metadata.
            signals.append("shared_emergency_contact")

        # Keep signal order deterministic and unique.
        seen: set[str] = set()
        unique_signals: List[str] = []
        for signal in signals:
            if signal not in seen:
                seen.add(signal)
                unique_signals.append(signal)
        return unique_signals

    def _build_observation(self, message: str) -> InsuranceClaimObservation:
        task = self._runtime_task or build_runtime_task(self._state.task_id)
        if len(task.expected_signals) == 0:
            evidence_quality_score = 1.0 if self._false_flags == 0 else 0.0
        else:
            evidence_quality_score = (
                float(self._evidence_hits) / float(self._evidence_total)
                if self._evidence_total > 0
                else 0.0
            )

        reward_breakdown = compute_reward_breakdown(
            task_id=task.task_id,
            expected_signals=task.expected_signals,
            found_signals=self._found_signals,
            false_flags=self._false_flags,
            step_number=self._state.step_number,
            max_steps=self._state.max_steps,
            final_decision=self._state.final_decision,
            allowed_decisions=task.allowed_final_decisions,
            payout_estimate_inr=self._state.payout_estimate_inr,
            payout_band=task.payout_band,
            investigation_targets=self._investigation_targets,
            evidence_quality_score=evidence_quality_score,
            exploit_penalty=min(self._exploit_penalty, 0.5),
            penalty_total=self._state.penalty_total,
            queried_claims=self._queried_claims,
        )

        return InsuranceClaimObservation(
            claim_id=self._payload["claim_id"],
            task_id=self._payload["task_id"],
            claimant=deepcopy(self._payload["claimant"]),
            incident=deepcopy(self._payload["incident"]),
            documents=deepcopy(self._payload["documents"]),
            linked_claims=deepcopy(self._visible_linked_claims),
            action_history=deepcopy(self._action_history),
            available_actions=deepcopy(self._payload["available_actions"]),
            step_number=self._state.step_number,
            max_steps=self._state.max_steps,
            flags_raised=deepcopy(self._flags_raised),
            status=self._state.status,
            message=message,
            done=self._state.done,
            reward=reward_breakdown.total,
            metadata={
                "last_action_error": self._state.last_action_error,
                "investigation_targets": self._investigation_targets,
                "variant_id": self._payload.get("variant_id", 0),
                "evidence_hits": self._evidence_hits,
                "evidence_total": self._evidence_total,
                "exploit_penalty": round(self._exploit_penalty, 4),
            },
            reward_breakdown=reward_breakdown,
        )


def available_task_ids() -> List[str]:
    return list(TASKS.keys())
