export const TASK_DESCRIPTIONS = {
  clean_claim: "Easy: all documents agree. Correct action: approve_claim with HIGH confidence.",
  contradictory_claim: "Medium: documents disagree. Correct action: deny_claim with MED confidence.",
  distribution_shift_claim: "Hard: looks normal at first. Correct action: escalate_to_human with LOW confidence.",
};

export const TASK_STRATEGIES = {
  clean_claim: [
    { action_type: "validate_document", parameters: { doc_id: "DOC-1" }, reasoning: "Verify primary claim document." },
    { action_type: "validate_document", parameters: { doc_id: "DOC-2" }, reasoning: "Verify garage estimate." },
    { action_type: "estimate_payout", parameters: { amount_inr: 150000 }, reasoning: "Standard auto claim payout." },
    { action_type: "approve_claim", parameters: { reason: "All documents consistent." }, reasoning: "Clean claim — HIGH confidence.", confidence: "HIGH" },
  ],
  contradictory_claim: [
    { action_type: "validate_document", parameters: { doc_id: "DOC-10" }, reasoning: "Check claim form date." },
    { action_type: "validate_document", parameters: { doc_id: "DOC-11" }, reasoning: "Check hospital admission." },
    { action_type: "validate_document", parameters: { doc_id: "DOC-12" }, reasoning: "Check billing summary for inflation." },
    { action_type: "query_historical_data", parameters: {}, reasoning: "Check prior claim history." },
    { action_type: "flag_fraud_signal", parameters: { flag_id: "date_mismatch", evidence: "Claim form date differs from hospital admission date." }, reasoning: "Date inconsistency flagged." },
    { action_type: "flag_fraud_signal", parameters: { flag_id: "cost_inflation", evidence: "Billing is 2.4x the standard rate for this procedure." }, reasoning: "Cost inflation detected." },
    { action_type: "convene_debate_panel", parameters: {}, reasoning: "Seek adversarial perspectives before final decision." },
    { action_type: "deny_claim", parameters: { reason: "Procedure mismatch and cost inflation confirmed by debate panel." }, reasoning: "Panel leans prosecution — MED confidence appropriate.", confidence: "MED" },
  ],
  distribution_shift_claim: [
    { action_type: "validate_document", parameters: { doc_id: "DOC-41" }, reasoning: "Initial document check." },
    { action_type: "query_historical_data", parameters: {}, reasoning: "Must check cross-claim patterns." },
    { action_type: "query_linked_claim", parameters: { claim_id: "CLM-DIST-602" }, reasoning: "Investigate linked claim for ring pattern." },
    { action_type: "query_linked_claim", parameters: { claim_id: "CLM-DIST-603" }, reasoning: "Second linked claim — same broker." },
    { action_type: "flag_fraud_signal", parameters: { flag_id: "clustered_policy_broker", evidence: "3 claimants share broker BRK-882 and same repair shop." }, reasoning: "Coordinated ring detected." },
    { action_type: "escalate_to_human", parameters: { reason: "Cross-claim fraud ring — expert review required." }, reasoning: "Full ring scope unclear — LOW confidence correct.", confidence: "LOW" },
  ],
};
