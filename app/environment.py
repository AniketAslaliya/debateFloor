from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .models import (
    ClaimStatus,
    InsuranceClaimAction,
    InsuranceClaimObservation,
    InsuranceClaimState,
)
from .tasks import (
    ACTION_COSTS,
    TASKS,
    RuntimeTask,
    build_runtime_task,
    build_initial_payload,
    compute_reward_breakdown,
    get_compare_signals,
    get_evidence_keyword_hints,
    get_task_definition,
)
from server.calibration_grader import calibration_reward as compute_calibration_reward

# Map Literal confidence levels to float for Brier-score compatibility
_CONFIDENCE_TO_FLOAT = {"HIGH": 0.9, "MED": 0.6, "LOW": 0.3}

# Correct terminal action for each task — used by calibration grader
_TASK_GROUND_TRUTH = {
    "clean_claim":              "approve_claim",
    "contradictory_claim":      "deny_claim",
    "coordinated_fraud":        "escalate_to_human",
    "identity_fraud":           "deny_claim",
    "distribution_shift_claim": "escalate_to_human",
}


class InsuranceClaimEnvironment:
    SUPPORTS_CONCURRENT_SESSIONS: bool = True  # NOW ACTUALLY TRUE - session-managed via main.py

    def __init__(self):
        self._state = InsuranceClaimState(episode_id=str(uuid4()), step_count=0)
        self._payload: Dict[str, Any] = {}
        self._action_history: List[Dict[str, Any]] = []
        self._flags_raised: List[str] = []
        self._found_signals: List[str] = []
        self._discovered_signals: List[str] = []
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
        self._policy_history_checked: bool = False
        self._identity_verified: bool = False
        self._agent_confidence: Optional[float] = None
        self._agent_confidence_str: Optional[str] = None  # "HIGH" | "MED" | "LOW"
        self._calibration_score: Optional[float] = None   # from 3x2 matrix
        self._episode_history: List[Dict] = []            # for anti-gaming detection
        self._budget_remaining: int = 0
        self._compared_pairs: set[tuple] = set()
        self._debate_transcript: Optional[Dict[str, Any]] = None
        self._debate_convened: bool = False

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None, episode_id: Optional[str] = None) -> InsuranceClaimObservation:
        selected_task = task_id or "clean_claim"
        task = build_runtime_task(selected_task, seed=seed)
        self._runtime_task = task

        self._payload = build_initial_payload(task)
        self._action_history = []
        self._flags_raised = []
        self._found_signals = []
        self._discovered_signals = []
        self._false_flags = 0
        self._investigation_targets = []
        self._evidence_hits = 0
        self._evidence_total = 0
        self._exploit_penalty = 0.0
        self._request_info_streak = 0
        self._last_progress_step = 0
        self._queried_claims = set()
        self._visible_linked_claims = deepcopy(self._payload.get("linked_claims", []))
        self._policy_history_checked = False
        self._identity_verified = False
        self._agent_confidence = None
        self._agent_confidence_str = None
        self._calibration_score = None
        self._budget_remaining = self._payload.get("investigation_budget", 0)
        self._compared_pairs = set()
        self._debate_transcript = None
        self._debate_convened = False
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
            discovered_signals=[],
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

        # Deduct investigation budget; overage adds 0.02 penalty per unit
        cost = ACTION_COSTS.get(action.action_type, 1)
        self._budget_remaining -= cost
        if self._budget_remaining < 0:
            self._state.penalty_total += 0.02  # per unit over budget

        if action.action_type == "request_information":
            self._request_info_streak += 1
            if self._request_info_streak > 2:
                self._exploit_penalty += 0.03
            if self._request_info_streak > 1:
                self._state.penalty_total += 0.02
            return "Additional information requested. Useful but consumes time and SLA budget."

        self._request_info_streak = 0

        if action.action_type == "lookup_policy_history":
            task = self._runtime_task or build_runtime_task(self._state.task_id)
            if self._policy_history_checked:
                # Second lookup is an exploit — no new info
                self._exploit_penalty += 0.03
                return "Policy history already retrieved. No new information available."
            self._policy_history_checked = True
            history = task.policy_history
            # For contradictory_claim: looking up history reveals the prior similar claim signal
            if task.task_id == "contradictory_claim":
                self._record_discovered_signals(["prior_similar_claim"])
            # For identity_fraud: policy_age_days being very low reveals recent_policy_purchase
            if task.task_id == "identity_fraud":
                if history.get("policy_age_days", 999) <= 30:
                    self._record_discovered_signals(["recent_policy_purchase"])
            return (
                f"Policy history retrieved: {history['prior_claims']} prior claims. "
                f"Customer for {history['years_as_customer']} years. "
                f"Policy age: {history['policy_age_days']} days. "
                f"Risk score: {history['risk_score']}. Note: {history['note']}"
            )

        if action.action_type == "verify_identity":
            task = self._runtime_task or build_runtime_task(self._state.task_id)
            if task.task_id != "identity_fraud":
                raise ValueError("'verify_identity' is only available for the identity_fraud task")
            if self._identity_verified:
                self._exploit_penalty += 0.03
                return "Identity verification already performed. No new information."
            self._identity_verified = True
            self._record_discovered_signals(["identity_mismatch", "hospital_no_record"])
            return (
                "Identity verification FAILED. National registry has no record matching "
                "claimant name 'Aarav Mehta' with ID suffix 7821. "
                "Hospital records show admission under a different name ('Aarav Kumar') with DOB mismatch. "
                "KYC status at policy inception: PENDING — identity was never confirmed."
            )

        if action.action_type == "compare_documents":
            task = self._runtime_task or build_runtime_task(self._state.task_id)
            doc_id_a = str(action.parameters.get("doc_id_a", "")).strip()
            doc_id_b = str(action.parameters.get("doc_id_b", "")).strip()
            if not doc_id_a or not doc_id_b:
                raise ValueError("'doc_id_a' and 'doc_id_b' are required for compare_documents")
            if doc_id_a == doc_id_b:
                raise ValueError("'doc_id_a' and 'doc_id_b' must be different documents")

            all_doc_ids = {d["doc_id"] for d in self._payload["documents"]}
            for did in (doc_id_a, doc_id_b):
                if did not in all_doc_ids:
                    raise ValueError(f"Unknown doc_id '{did}'")

            pair = (doc_id_a, doc_id_b)
            pair_rev = (doc_id_b, doc_id_a)
            if pair in self._compared_pairs or pair_rev in self._compared_pairs:
                self._exploit_penalty += 0.03
                return f"Documents {doc_id_a} and {doc_id_b} were already compared. No new findings."

            self._compared_pairs.add(pair)
            signals = get_compare_signals(task.task_id, doc_id_a, doc_id_b)
            if signals:
                self._record_discovered_signals(signals)
                return (
                    f"Cross-document comparison of {doc_id_a} vs {doc_id_b} revealed "
                    f"inconsistencies: {', '.join(signals)}."
                )
            return f"Cross-document comparison of {doc_id_a} vs {doc_id_b}: documents are consistent."

        if action.action_type == "validate_document":
            doc_id = str(action.parameters.get("doc_id", "")).strip()
            if not doc_id:
                raise ValueError("'doc_id' is required for validate_document")

            doc = next((d for d in self._payload["documents"] if d.get("doc_id") == doc_id), None)
            if doc is None:
                raise ValueError(f"Unknown doc_id '{doc_id}'")

            discovered = self._discover_signals_from_document(doc_id, task.task_id)
            if discovered:
                self._record_discovered_signals(discovered)
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
                if flag_id not in self._discovered_signals:
                    self._state.penalty_total += 0.08
                    self._exploit_penalty += 0.02
                    return (
                        f"Fraud signal '{flag_id}' was raised before it was discovered. "
                        "Investigate first, then flag with grounded evidence."
                    )
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

            # Dynamic ring expansion: after querying 2 existing claims, the 4th
            # hidden claim (CLM-GROUP-304) surfaces in linked_claims.
            expansion_hint = ""
            if len(self._queried_claims) >= 2:
                full_linked = self._payload.get("_full_linked_claims", [])
                hidden = [
                    c for c in full_linked
                    if c.get("_hidden_until_queries", 0) <= len(self._queried_claims)
                    and not any(v.get("claim_id") == c["claim_id"] for v in self._visible_linked_claims)
                ]
                for new_claim in hidden:
                    stub = {"claim_id": new_claim["claim_id"], "claimant": new_claim["claimant"]}
                    self._visible_linked_claims.append(stub)
                    expansion_hint = (
                        f" NEW: A previously unknown linked claim {new_claim['claim_id']} "
                        f"({new_claim['claimant']}) has surfaced. Query it for full details."
                    )

            # After querying 2+ linked claims, the shared emergency contact becomes detectable.
            hint = ""
            if len(self._queried_claims) >= 2:
                queried_data = [
                    c for c in self._visible_linked_claims
                    if c.get("claim_id") in self._queried_claims and len(c) > 2
                ]
                contacts = [c.get("emergency_contact") for c in queried_data if c.get("emergency_contact")]
                unique_contacts = set(contacts)
                if len(contacts) > 1 and len(unique_contacts) == 1:
                    hint = f" Cross-claim pattern detected: all queried claims share emergency_contact={contacts[0]}."

            # Querying CLM-GROUP-304 reveals clustered_policy_broker signal
            if match.get("broker_id") and claim_id == "CLM-GROUP-304":
                self._record_discovered_signals(["clustered_policy_broker"])
                hint += " All queried claims share broker_id=BRK-441 (clustered_policy_broker signal)."

            return f"Linked claim detail retrieved for {claim_id}: {match}{hint}{expansion_hint}"

        if action.action_type in {
            "approve_claim", "deny_claim", "request_investigation", "escalate_to_human"
        }:
            # Normalise escalate_to_human → request_investigation for legacy grader
            canonical_decision = (
                "request_investigation"
                if action.action_type == "escalate_to_human"
                else action.action_type
            )
            self._state.final_decision = canonical_decision
            self._state.done = True
            self._state.status = ClaimStatus.DECIDED

            # Capture Literal confidence and convert for Brier-score compatibility
            if action.confidence is not None:
                conf_str = str(action.confidence)
                self._agent_confidence_str = conf_str
                self._agent_confidence = _CONFIDENCE_TO_FLOAT.get(conf_str)

                # Compute DebateFloor calibration reward via 3x2 matrix
                ground_truth = _TASK_GROUND_TRUTH.get(self._state.task_id, "deny_claim")
                # Map escalate_to_human ground truth to canonical for comparison
                effective_decision = action.action_type
                effective_ground_truth = (
                    "escalate_to_human"
                    if ground_truth == "request_investigation"
                    else ground_truth
                )
                self._calibration_score = compute_calibration_reward(
                    effective_decision, conf_str, effective_ground_truth,
                    self._episode_history,
                )
                # Record this episode for future gaming detection
                self._episode_history.append({"confidence": conf_str})

            if canonical_decision == "request_investigation":
                targets = action.parameters.get("target_claim_ids", [])
                if isinstance(targets, list):
                    self._investigation_targets = [str(t) for t in targets]
                else:
                    raise ValueError("'target_claim_ids' must be a list for request_investigation")

            reason = str(action.parameters.get("reason", "")).strip()
            if not reason and action.action_type not in {"approve_claim", "escalate_to_human"}:
                self._state.penalty_total += 0.03

            self._state.status = ClaimStatus.CLOSED
            return f"Final decision submitted: {action.action_type}."

        if action.action_type == "query_historical_data":
            # Alias for lookup_policy_history — used by distribution_shift_claim task
            if self._policy_history_checked:
                self._exploit_penalty += 0.03
                return "Historical data already retrieved. No new information available."
            self._policy_history_checked = True
            task = self._runtime_task or build_runtime_task(self._state.task_id)
            if task.task_id in {"contradictory_claim", "distribution_shift_claim"}:
                self._record_discovered_signals(["prior_similar_claim"])
            if task.task_id == "identity_fraud":
                history = task.policy_history
                if history.get("policy_age_days", 999) <= 30:
                    self._record_discovered_signals(["recent_policy_purchase"])
            return (
                "Historical data retrieved. Cross-claim patterns and policy history available. "
                "Prior claim activity and linked policy data surfaced."
            )

        if action.action_type == "verify_provider_registration":
            task = self._runtime_task or build_runtime_task(self._state.task_id)
            if task.task_id not in {"distribution_shift_claim"}:
                raise ValueError("'verify_provider_registration' is only available for distribution_shift_claim")
            self._record_discovered_signals(["unregistered_provider", "invalid_gst_registration"])
            return "Provider registration check: hospital not found in IRDAI registry. GST number invalid."

        if action.action_type == "convene_debate_panel":
            if self._debate_convened:
                self._exploit_penalty += 0.03
                return "Debate panel already convened this episode. Proceed to terminal decision."
            self._debate_convened = True
            self._debate_transcript = self._generate_debate_transcript()
            self._last_progress_step = self._state.step_number
            return (
                f"Debate panel convened. "
                f"Prosecutor: {self._debate_transcript['prosecutor_argument'][:80]}... "
                f"Defender: {self._debate_transcript['defender_argument'][:80]}... "
                f"Panel verdict: {self._debate_transcript['panel_verdict']}. "
                "Review transcript in observation.debate_transcript, then make your final decision."
            )

        raise ValueError(f"Unsupported action_type '{action.action_type}'")

    def _generate_debate_transcript(self) -> Dict[str, Any]:
        """Generate a structured prosecutor vs defender debate based on investigation state."""
        task = self._runtime_task
        found = self._found_signals
        discovered = self._discovered_signals
        claimant_name = self._payload.get("claimant", {}).get("name", "the claimant")
        incident_type = self._payload.get("incident", {}).get("type", "the incident")

        # Prosecutor builds case from discovered and flagged signals
        if found:
            fraud_signals_str = ", ".join(found)
            prosecutor = (
                f"PROSECUTOR: The evidence strongly suggests fraud. "
                f"Investigation has uncovered {len(found)} fraud signal(s): {fraud_signals_str}. "
                f"These signals are consistent with {task.task_id.replace('_', ' ')} fraud patterns. "
                f"I recommend denial or escalation — approving this claim would reward deliberate deception."
            )
            prosecutor_strength = "STRONG" if len(found) >= 2 else "MODERATE"
        elif discovered:
            prosecutor = (
                f"PROSECUTOR: Suspicious indicators have been discovered: {', '.join(discovered)}. "
                f"While not yet formally flagged, these anomalies warrant serious scrutiny. "
                f"The claim by {claimant_name} regarding {incident_type} shows red flags."
            )
            prosecutor_strength = "WEAK"
        else:
            prosecutor = (
                f"PROSECUTOR: No fraud signals have been found yet, but the investigation "
                f"may be incomplete. More documents should be validated before approval. "
                f"Insufficient investigation is itself a risk."
            )
            prosecutor_strength = "INSUFFICIENT"

        # Defender builds case from clean documents and policy context
        doc_count = len(self._payload.get("documents", []))
        policy_age = self._payload.get("_policy_history", {}).get("policy_age_days", 0)
        if task and task.task_id == "clean_claim":
            defender = (
                f"DEFENDER: All {doc_count} documents are internally consistent. "
                f"Claimant {claimant_name} has a clean policy history. "
                f"No fraud indicators found. This is a legitimate claim — denial would be unjust."
            )
            defender_strength = "STRONG"
        elif found and len(found) >= len(task.expected_signals if task else []) * 0.6:
            defender = (
                f"DEFENDER: While anomalies exist, the core claim documentation ({doc_count} docs) "
                f"has not been fully discredited. Some apparent inconsistencies may have innocent explanations. "
                f"Burden of proof requires clear evidence, not suspicion."
            )
            defender_strength = "WEAK"
        else:
            defender = (
                f"DEFENDER: The claim has {doc_count} supporting documents submitted on time. "
                f"Without confirmed fraud signals, denial would expose the insurer to legal challenge. "
                f"Claimant {claimant_name} deserves due process. Standard processing is warranted."
            )
            defender_strength = "MODERATE"

        # Panel verdict: which side has stronger case
        strength_rank = {"STRONG": 3, "MODERATE": 2, "WEAK": 1, "INSUFFICIENT": 0}
        p_rank = strength_rank.get(prosecutor_strength, 0)
        d_rank = strength_rank.get(defender_strength, 0)

        if p_rank > d_rank:
            verdict = f"Panel leans PROSECUTION ({prosecutor_strength} case). Recommended action: deny_claim or escalate_to_human."
            lean = "prosecution"
        elif d_rank > p_rank:
            verdict = f"Panel leans DEFENSE ({defender_strength} case). Recommended action: approve_claim."
            lean = "defense"
        else:
            verdict = "Panel is SPLIT — both sides have comparable arguments. Judge must use independent judgment and declare LOW confidence."
            lean = "split"

        return {
            "prosecutor_argument": prosecutor,
            "prosecutor_strength": prosecutor_strength,
            "defender_argument": defender,
            "defender_strength": defender_strength,
            "panel_verdict": verdict,
            "panel_lean": lean,
            "signals_at_debate": list(found),
            "step_convened": self._state.step_number,
        }

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
            "identity_fraud": {
                "DOC-31": ["identity_mismatch"],
                "DOC-32": ["hospital_no_record"],
                # DOC-33 (policy_inception) does NOT reveal recent_policy_purchase here;
                # that signal is only discoverable via lookup_policy_history.
                "DOC-34": ["dob_inconsistency"],
            },
        }
        signal_map = mapping.get(task_id, {})
        signals = list(signal_map.get(doc_id, []))

        # NOTE: shared_emergency_contact is NOT discoverable from primary documents.
        # It can only be found by calling query_linked_claim on at least 2 linked claims,
        # then flag_fraud_signal with evidence from the queried data. This enforces
        # genuine multi-hop reasoning rather than single-step observation reading.

        # Keep signal order deterministic and unique.
        seen: set[str] = set()
        unique_signals: List[str] = []
        for signal in signals:
            if signal not in seen:
                seen.add(signal)
                unique_signals.append(signal)
        return unique_signals

    def _record_discovered_signals(self, signals: List[str]) -> None:
        progressed = False
        for signal in signals:
            if signal not in self._discovered_signals:
                self._discovered_signals.append(signal)
                progressed = True
            if signal not in self._found_signals:
                self._found_signals.append(signal)
        if progressed:
            self._last_progress_step = self._state.step_number

    def _build_observation(self, message: str) -> InsuranceClaimObservation:
        task = self._runtime_task or build_runtime_task(self._state.task_id)
        self._state.flags_raised = deepcopy(self._flags_raised)
        self._state.discovered_signals = deepcopy(self._discovered_signals)
        self._state.found_signals = deepcopy(self._found_signals)
        if self._state.step_number == 0:
            # No actions taken yet — reward must be 0.0 so the trajectory is meaningful
            evidence_quality_score = 0.0
        elif len(task.expected_signals) == 0:
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
            agent_confidence=self._agent_confidence,
            ground_truth_confidence=task.ground_truth_confidence,
        )

        # Override calibration_score with DebateFloor 3x2 matrix value when available
        if self._calibration_score is not None:
            reward_breakdown.calibration_score = self._calibration_score

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
            investigation_budget=self._payload.get("investigation_budget", 0),
            budget_remaining=self._budget_remaining,
            flags_raised=deepcopy(self._flags_raised),
            discovered_signals=deepcopy(self._discovered_signals),
            status=self._state.status,
            message=message,
            confidence_required=True,
            done=self._state.done,
            reward=reward_breakdown.total,
            metadata={
                "last_action_error": self._state.last_action_error,
                "investigation_targets": self._investigation_targets,
                "variant_id": self._payload.get("variant_id", 0),
                "evidence_hits": self._evidence_hits,
                "evidence_total": self._evidence_total,
                "exploit_penalty": round(self._exploit_penalty, 4),
                "policy_history_checked": self._policy_history_checked,
                "identity_verified": self._identity_verified,
                "agent_confidence": self._agent_confidence_str,
                "calibration_score": self._calibration_score,
                "budget_remaining": self._budget_remaining,
                "discovered_signals": deepcopy(self._discovered_signals),
                "compared_pairs": [list(p) for p in self._compared_pairs],
            },
            reward_breakdown=reward_breakdown,
            debate_transcript=deepcopy(self._debate_transcript),
        )


def available_task_ids() -> List[str]:
    return list(TASKS.keys())
