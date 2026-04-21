"""
tests/test_generator.py
Tests for server/claim_generator.py — run with: pytest tests/test_generator.py -v
"""

import pytest
from server.claim_generator import (
    generate_claim,
    generate_episode_pool,
    FRAUD_TYPES,
    COVERAGE_TYPES,
    ClaimScenario,
)


class TestDeterminism:

    def test_same_seed_returns_same_claim(self):
        a = generate_claim(42, "medical_inflation", "health", "medium")
        b = generate_claim(42, "medical_inflation", "health", "medium")
        assert a.claim_id == b.claim_id
        assert a.claimant["name"] == b.claimant["name"]
        assert a.payout_amount_inr == b.payout_amount_inr
        assert a.ground_truth == b.ground_truth

    def test_different_seeds_return_different_claims(self):
        a = generate_claim(1, "medical_inflation", "health", "medium")
        b = generate_claim(2, "medical_inflation", "health", "medium")
        assert a.claim_id != b.claim_id

    def test_claim_id_encodes_seed_and_fraud_type(self):
        c = generate_claim(99, "staged_accident", "auto", "easy")
        assert "0099" in c.claim_id
        assert "STA" in c.claim_id


class TestAllFraudTypes:

    @pytest.mark.parametrize("fraud_type", FRAUD_TYPES)
    def test_all_fraud_types_generate_correctly(self, fraud_type):
        c = generate_claim(0, fraud_type, "health", "medium")
        assert isinstance(c, ClaimScenario)
        assert c.fraud_type == fraud_type
        assert c.ground_truth in {"approve_claim", "deny_claim", "escalate_to_human"}
        assert len(c.documents) >= 2
        assert len(c.available_actions) >= 5

    @pytest.mark.parametrize("fraud_type", FRAUD_TYPES)
    def test_fraud_types_have_signals(self, fraud_type):
        c = generate_claim(0, fraud_type, "health", "easy")
        assert len(c.expected_fraud_signals) > 0, f"{fraud_type} should have fraud signals on easy"

    def test_clean_claim_approves_and_no_signals(self):
        c = generate_claim(0, "none", "auto", "easy")
        assert c.ground_truth == "approve_claim"
        assert c.expected_fraud_signals == []

    def test_coordinated_ring_has_linked_claims(self):
        c = generate_claim(0, "coordinated_ring", "auto", "medium")
        assert len(c.linked_claims) >= 3

    def test_coordinated_ring_escalates(self):
        c = generate_claim(0, "coordinated_ring", "health", "medium")
        assert c.ground_truth == "escalate_to_human"

    def test_identity_fraud_has_verify_action(self):
        c = generate_claim(0, "identity_fraud", "health", "medium")
        assert "verify_identity" in c.available_actions


class TestDifficulty:

    def test_easy_has_low_ambiguity(self):
        c = generate_claim(0, "medical_inflation", "health", "easy")
        assert c.ambiguity_score < 0.3

    def test_hard_has_high_ambiguity(self):
        c = generate_claim(0, "medical_inflation", "health", "hard")
        assert c.ambiguity_score > 0.6

    def test_easy_max_steps_10(self):
        c = generate_claim(0, "staged_accident", "auto", "easy")
        assert c.max_steps == 10

    def test_medium_max_steps_18(self):
        c = generate_claim(0, "staged_accident", "auto", "medium")
        assert c.max_steps == 18

    def test_hard_max_steps_28(self):
        c = generate_claim(0, "staged_accident", "auto", "hard")
        assert c.max_steps == 28


class TestCoverageTypes:

    @pytest.mark.parametrize("coverage", COVERAGE_TYPES)
    def test_all_coverage_types_generate(self, coverage):
        c = generate_claim(0, "medical_inflation", coverage, "medium")
        assert c.coverage_type == coverage
        assert c.payout_amount_inr > 0


class TestEpisodePool:

    def test_500_unique_episodes_no_duplicates(self):
        pool = generate_episode_pool(count=500)
        assert len(pool) == 500
        ids = [e.claim_id for e in pool]
        assert len(set(ids)) == 500, "All 500 episodes must have unique claim IDs"

    def test_pool_covers_all_fraud_types(self):
        pool = generate_episode_pool(count=100)
        found_types = {e.fraud_type for e in pool}
        assert found_types == set(FRAUD_TYPES)


class TestValidation:

    def test_invalid_fraud_type_raises(self):
        with pytest.raises(ValueError):
            generate_claim(0, "nonexistent_fraud", "health", "medium")

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError):
            generate_claim(0, "medical_inflation", "crypto", "medium")

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            generate_claim(0, "medical_inflation", "health", "extreme")

    def test_ambiguity_always_in_0_1_range(self):
        for seed in range(50):
            c = generate_claim(seed, "coordinated_ring", "auto", "hard")
            assert 0.0 <= c.ambiguity_score <= 1.0
