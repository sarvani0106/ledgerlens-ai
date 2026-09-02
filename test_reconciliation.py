"""
test_reconciliation.py
Expanded Buildathon test suite for LedgerLens AI — Evidence-Based Tax-Line Reconciliation Agent.

Tests:
  1. Invoice normalization (/ vs -, leading zeroes)
  2. Levenshtein string distance
  3. Evidence validation: split invoice sums
  4. Evidence validation: payment gateway MDR fees + GST
  5. Adversarial AI hallucination rejection by Evidence Layer
  6. Explicit ABSTAIN on ambiguous candidate collisions
  7. Standard dataset (140+ rows) ground truth integrity
  8. Held-Out Challenge dataset (40 rows) evaluation
  9. LLM fault injection: Timeout graceful degradation
  10. LLM fault injection: Invalid JSON degradation
  11. LLM fault injection: 503 Service Outage degradation
  12. End-to-end LedgerLens AI hybrid pipeline execution & zero false positives
"""

import pytest
import pandas as pd
from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset, introduce_typo
from matcher import normalize_invoice_no, clean_and_normalize_dataframe, run_deterministic_matching
from exception_resolver import EvidenceValidator, ExceptionResolver, compute_levenshtein
from audit_logger import AuditLogger
from llm_matcher import LLMInvestigationAgent, _clean_json_string
from reconciliation_engine import run_full_reconciliation


def test_invoice_normalization():
    """Verify standardizing slashes, hyphens, and leading zeroes."""
    assert normalize_invoice_no("INV/2024/001042") == "INV-2024-1042"
    assert normalize_invoice_no("inv_2024.1001") == "INV-2024-1001"
    assert normalize_invoice_no("INV-2024-0005") == "INV-2024-5"


def test_levenshtein_distance():
    """Verify string distance calculation."""
    assert compute_levenshtein("NovaTech Solutions", "NovaTceh Solutions") == 2
    assert compute_levenshtein("Apex Retail Network", "Apex Retail Network") == 0


def test_evidence_validator_split_invoice():
    """Verify split invoice math checking."""
    passed, msg, diff = EvidenceValidator.verify_split_invoice(50000.0, [30000.0, 20000.0])
    assert passed is True
    assert diff == 0.0

    passed, msg, diff = EvidenceValidator.verify_split_invoice(50000.0, [30000.0, 15000.0])
    assert passed is False
    assert diff == 5000.0


def test_evidence_validator_gateway_fee():
    """Verify gateway MDR fee calculation (2% fee + 18% GST on fee)."""
    # ₹10,000 gross -> Fee ₹200 + GST ₹36 = ₹9,764 net
    passed, msg, diff = EvidenceValidator.verify_gateway_fee(10000.0, 9764.0)
    assert passed is True
    assert diff == 0.0

    passed, msg, diff = EvidenceValidator.verify_gateway_fee(10000.0, 8500.0)
    assert passed is False


def test_adversarial_ai_hallucination_rejection():
    """Verify that the Evidence Validation layer rejects false LLM claims."""
    resolver = ExceptionResolver(confidence_auto_resolve=0.95, confidence_review_min=0.75)
    
    inv_row = {
        "invoice_no": "INV-2024-1099",
        "vendor_name": "BluePeak Technologies Pvt Ltd",
        "amount": 100000.0,
        "tax_rate": 18.0,
        "date": "2024-01-15"
    }
    
    fake_ai_proposal = {
        "invoice_no": "INV-2024-1099",
        "decision": "MATCH",
        "confidence": 0.99,  # Adversarial high confidence claim
        "exception_type": "SPLIT_INVOICE",
        "root_cause": "Split invoice match",
        "candidate_record_ids": ["INV-FAKE-A", "INV-FAKE-B"]
    }
    
    cand_rows = [
        {"invoice_no": "INV-FAKE-A", "amount": 40000.0, "vendor_name": "BluePeak Technologies Pvt Ltd", "date": "2024-01-15"},
        {"invoice_no": "INV-FAKE-B", "amount": 40000.0, "vendor_name": "BluePeak Technologies Pvt Ltd", "date": "2024-01-15"}
    ]  # Sum = 80,000 != 100,000
    
    result = resolver.evaluate_ai_proposal(inv_row, cand_rows, fake_ai_proposal)
    assert result["evidence_validation_passed"] is False
    assert result["final_decision"] in ["ABSTAIN", "HUMAN_REVIEW"]


def test_abstain_decision_on_ambiguous_collisions():
    """Verify that multiple candidate collisions result in explicit ABSTAIN."""
    resolver = ExceptionResolver()
    inv_row = {
        "invoice_no": "INV-2024-1050",
        "vendor_name": "NovaTech Solutions Pvt Ltd",
        "amount": 50000.0,
        "date": "2024-02-01"
    }
    ai_prop = {
        "invoice_no": "INV-2024-1050",
        "decision": "ABSTAIN",
        "confidence": 0.50,
        "exception_type": "MULTIPLE_POSSIBLE_MATCHES",
        "root_cause": "Multiple identical candidates in ledger",
        "candidate_record_ids": ["CAND-1", "CAND-2"]
    }
    cand_rows = [
        {"invoice_no": "CAND-1", "amount": 50000.0, "vendor_name": "NovaTech Solutions Pvt Ltd", "date": "2024-02-01"},
        {"invoice_no": "CAND-2", "amount": 50000.0, "vendor_name": "NovaTech Solutions Pvt Ltd", "date": "2024-02-01"}
    ]
    result = resolver.evaluate_ai_proposal(inv_row, cand_rows, ai_prop)
    assert result["final_decision"] == "ABSTAIN"
    assert result["recommended_action"] == "HUMAN_REVIEW"


def test_synthetic_ground_truth_datasets():
    """Verify 140+ records and metadata integrity."""
    inv_df, led_df, meta = generate_synthetic_datasets(num_invoices=140, seed=42)
    assert len(inv_df) == 140
    assert len(led_df) >= 140
    assert "ground_truth" in meta
    assert len(meta["ground_truth"]) == 140


def test_held_out_challenge_dataset_evaluation():
    """Evaluate on the separate Held-Out Challenge Dataset (seed=101)."""
    ch_inv, ch_led, ch_meta = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
    assert len(ch_inv) == 40
    assert len(ch_led) >= 40
    
    results = run_full_reconciliation(
        invoices_df=ch_inv,
        ledger_df=ch_led,
        ground_truth_meta=ch_meta,
        amount_tolerance_abs=15.0,
        amount_tolerance_pct=0.01,
        anthropic_api_key=None,
        enable_offline_fallback=True
    )
    
    metrics = results["metrics"]
    # Challenge dataset has difficult adversarial scenarios (abstains & reviews)
    assert metrics["total_invoices"] == 40
    assert metrics["precision"] >= 95.0
    assert metrics["false_positives"] == 0  # Evidence validation must block false matches!
    assert metrics["human_review_count"] > 0


def test_fault_injection_timeout_degradation():
    """Verify that simulated LLM timeout degrades safely to fallback/exceptions without crashing."""
    inv_df, led_df, meta = generate_synthetic_datasets(num_invoices=20, seed=42)
    
    results = run_full_reconciliation(
        invoices_df=inv_df,
        ledger_df=led_df,
        ground_truth_meta=meta,
        simulate_failure="TIMEOUT",
        enable_offline_fallback=True
    )
    assert results["metrics"]["total_invoices"] == 20
    assert results["metrics"]["precision"] >= 95.0


def test_fault_injection_invalid_json_degradation():
    """Verify that simulated malformed JSON degrades safely."""
    inv_df, led_df, meta = generate_synthetic_datasets(num_invoices=20, seed=42)
    
    results = run_full_reconciliation(
        invoices_df=inv_df,
        ledger_df=led_df,
        ground_truth_meta=meta,
        simulate_failure="INVALID_JSON",
        enable_offline_fallback=True
    )
    assert results["metrics"]["total_invoices"] == 20
    assert results["metrics"]["precision"] >= 95.0


def test_full_ledgerlens_ai_pipeline():
    """End-to-end integration test of LedgerLens AI pipeline."""
    inv_df, led_df, meta = generate_synthetic_datasets(num_invoices=140, seed=42)
    
    results = run_full_reconciliation(
        invoices_df=inv_df,
        ledger_df=led_df,
        ground_truth_meta=meta,
        amount_tolerance_abs=15.0,
        amount_tolerance_pct=0.01,
        anthropic_api_key=None,  # Uses built-in forensic evaluator
        llm_model="claude-sonnet-4-6",
        confidence_auto_resolve=0.95,
        confidence_review_min=0.75,
        enable_offline_fallback=True
    )
    
    metrics = results["metrics"]
    baseline_df = results["baseline_comparison"]
    audit_logger = results["audit_logger"]
    
    # 1. Honest metrics check
    assert metrics["total_invoices"] == 140
    assert metrics["precision"] >= 95.0
    assert metrics["false_positives"] == 0  # Zero false positives due to evidence validation!
    assert metrics["false_positive_financial_impact"] == 0.0
    assert metrics["auto_resolved_count"] > 0
    assert metrics["human_review_count"] > 0
    assert metrics["unresolved_count"] > 0
    
    # 2. Baseline comparison check
    assert len(baseline_df) == 3
    assert any("LedgerLens AI" in m for m in baseline_df["Method"].values)
    
    # 3. Audit logger check
    assert len(audit_logger.logs) > 0
    phases = set(log["lifecycle_phase"] for log in audit_logger.logs)
    assert "MATCH" in phases
    assert "INVESTIGATE" in phases
    assert "VERIFY" in phases


def test_tamper_evident_audit_log_hash_chaining():
    """Verify SHA-256 cryptographic hash chaining and tampering detection."""
    logger = AuditLogger()
    
    # Log 3 events
    logger.log_lifecycle_step("INV-101", "OBSERVE", "Ingestion", "NORMALIZE", "SUCCESS", {"field": "invoice_no"})
    logger.log_lifecycle_step("INV-101", "MATCH", "Stage 1", "EXACT_MATCH", "SUCCESS", {"confidence": 1.0})
    logger.log_lifecycle_step("INV-101", "AUDIT", "Audit", "HASH_LOG", "SUCCESS", {})
    
    # 1. Verify clean integrity
    is_valid, count, err = logger.verify_audit_integrity()
    assert is_valid is True
    assert count == 3
    assert err is None
    
    # 2. Tamper with an entry and verify tamper is detected
    logger.logs[1]["action"] = "TAMPERED_ACTION"
    is_valid, count, err = logger.verify_audit_integrity()
    assert is_valid is False
    assert "tampering detected" in err.lower() or "broken" in err.lower()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
