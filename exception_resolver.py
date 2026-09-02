"""
exception_resolver.py
Exception Resolution & Programmatic Evidence Validation Engine for LedgerLens AI.

Key Responsibilities:
  1. Deep investigation and root-cause classification of unresolved records.
  2. Programmatic evidence verification (split invoice sums, payment gateway fees, GST rates, string edit distance).
  3. Enforcing Confidence & Safety Policy:
     - Confidence >= 0.95 AND Evidence Valid -> AUTO_RESOLVE
     - Confidence 0.75 - 0.94 OR Partial Evidence -> HUMAN_REVIEW
     - Conflicting evidence or ambiguous collisions -> ABSTAIN (routed to Human Review)
     - Confidence < 0.75 OR missing records -> LEAVE_UNRESOLVED
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd


# 15 Standardized Exception Categories
EXCEPTION_CATEGORIES = {
    "EXACT_MATCH": "Exact match on normalized GSTIN and Invoice Number",
    "INVOICE_FORMATTING_VARIATION": "Invoice number formatting variance (slashes, hyphens, leading zeroes)",
    "INVOICE_NUMBER_TYPO": "Minor typographical error in invoice number (1-2 char edit distance)",
    "GSTIN_TYPO_OR_FORMAT": "Typo or format variation in GSTIN string",
    "DATE_DIFFERENCE": "Invoice date vs ledger posting date variance within allowable business window",
    "ROUNDING_DIFFERENCE": "Minor amount variance due to fractional rupee rounding or cash discount (<= ₹15)",
    "GST_CALCULATION_DIFFERENCE": "Discrepancy in GST rate application or cess calculation",
    "SPLIT_INVOICE": "Single seller invoice distributed across multiple buyer ledger line items",
    "PARTIAL_PAYMENT_OR_FEE_DEDUCTION": "Payment gateway MDR fee and GST deduction (e.g., 2% fee + 18% GST on fee)",
    "MISSING_SOURCE_RECORD": "Seller invoice completely absent from buyer purchase ledger",
    "MISSING_GST_RECORD": "Buyer ledger record absent from seller GST return (GSTR-2B discrepancy)",
    "DUPLICATE_RECORD": "Duplicate ledger booking detected for single seller invoice",
    "MULTIPLE_POSSIBLE_MATCHES": "Multiple ambiguous candidate collisions with equal plausibility",
    "INSUFFICIENT_EVIDENCE": "Insufficient mathematical or textual evidence to confirm match",
    "UNKNOWN_ANOMALY": "Unclassified financial discrepancy requiring forensic audit"
}


def compute_levenshtein(s1: str, s2: str) -> int:
    """Standard Levenshtein string distance."""
    s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class EvidenceValidator:
    """
    Programmatic validation layer that verifies proposed matches mathematically and structurally.
    The LLM is NEVER trusted blindly; every claim must be validated here.
    """

    @staticmethod
    def verify_split_invoice(invoice_amount: float, candidate_amounts: List[float], tolerance: float = 1.0) -> Tuple[bool, str, float]:
        """
        Verify that the sum of split ledger entries equals the invoice amount within tolerance.
        """
        if not candidate_amounts:
            return False, "No split candidates provided", 0.0
        split_sum = round(sum(candidate_amounts), 2)
        diff = round(abs(split_sum - invoice_amount), 2)
        if diff <= tolerance:
            return True, f"Split sum ₹{split_sum:,.2f} exactly matches invoice total ₹{invoice_amount:,.2f} (diff: ₹{diff:.2f})", diff
        return False, f"Split sum ₹{split_sum:,.2f} does not match invoice ₹{invoice_amount:,.2f} (diff: ₹{diff:.2f} exceeds ₹{tolerance:.2f})", diff

    @staticmethod
    def verify_gateway_fee(
        gross_amount: float,
        settlement_amount: float,
        fee_rate: float = 0.02,       # 2% standard payment gateway fee
        gst_on_fee_rate: float = 0.18  # 18% GST on payment gateway fee
    ) -> Tuple[bool, str, float]:
        """
        Verify if the net settlement matches Gross - Gateway Fee - GST on Fee.
        Example: ₹10,000 gross -> Fee = ₹200, GST = ₹36 -> Expected Settlement = ₹9,764.
        """
        expected_fee = round(gross_amount * fee_rate, 2)
        expected_gst_on_fee = round(expected_fee * gst_on_fee_rate, 2)
        expected_net = round(gross_amount - expected_fee - expected_gst_on_fee, 2)
        diff = round(abs(expected_net - settlement_amount), 2)
        
        if diff <= 1.0:
            return True, f"Verified payment gateway deduction: Gross ₹{gross_amount:,.2f} - Fee(2%) ₹{expected_fee:,.2f} - GST(18%) ₹{expected_gst_on_fee:,.2f} = Net ₹{expected_net:,.2f}", diff
        return False, f"Gateway fee calculation mismatch: expected ₹{expected_net:,.2f}, got ₹{settlement_amount:,.2f} (diff: ₹{diff:.2f})", diff

    @staticmethod
    def verify_typo_or_format(s1: str, s2: str, max_dist: int = 2) -> Tuple[bool, str, int]:
        """Verify string typographical distance."""
        dist = compute_levenshtein(s1, s2)
        if dist <= max_dist:
            return True, f"Text distance ({dist}) within acceptable threshold (<= {max_dist})", dist
        return False, f"Text distance ({dist}) exceeds allowable threshold ({max_dist})", dist

    @staticmethod
    def verify_date_window(date_str1: str, date_str2: str, max_days: int = 45) -> Tuple[bool, str, int]:
        """Verify date proximity between invoice date and ledger date."""
        try:
            d1 = datetime.strptime(str(date_str1)[:10], "%Y-%m-%d")
            d2 = datetime.strptime(str(date_str2)[:10], "%Y-%m-%d")
            delta_days = abs((d1 - d2).days)
            if delta_days <= max_days:
                return True, f"Date difference of {delta_days} days is within allowable window ({max_days} days)", delta_days
            return False, f"Date difference of {delta_days} days exceeds allowable window ({max_days} days)", delta_days
        except Exception as e:
            return False, f"Could not parse dates ('{date_str1}', '{date_str2}'): {e}", 999


class ExceptionResolver:
    """
    Evaluates ambiguous records, performs programmatic evidence validation,
    applies confidence policy, and emits final actionable decisions.
    """

    def __init__(self, confidence_auto_resolve: float = 0.95, confidence_review_min: float = 0.75):
        self.conf_auto = confidence_auto_resolve
        self.conf_review = confidence_review_min
        self.validator = EvidenceValidator()

    def evaluate_ai_proposal(
        self,
        invoice_row: Dict[str, Any],
        candidate_rows: List[Dict[str, Any]],
        ai_proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Subject the AI proposal to rigorous evidence validation.
        
        Returns:
            dict containing:
              - final_decision: AUTO_RESOLVE | HUMAN_REVIEW | ABSTAIN | LEAVE_UNRESOLVED
              - validated_confidence: float
              - exception_category: str
              - root_cause: str
              - evidence_validation_passed: bool
              - evidence_validation_details: list
              - recommended_action: str
              - audit_notes: str
        """
        raw_decision = ai_proposal.get("decision", "NO_MATCH").upper()
        raw_conf = float(ai_proposal.get("confidence", 0.0))
        exception_type = ai_proposal.get("exception_type", "UNKNOWN_ANOMALY")
        root_cause = ai_proposal.get("root_cause", "AI Investigation")
        
        validation_tests = []
        is_evidence_valid = True
        
        inv_amt = float(invoice_row.get("amount", 0.0))
        inv_no = str(invoice_row.get("invoice_no", "")).strip().upper()
        inv_vendor = str(invoice_row.get("vendor_name", "")).strip()
        inv_date = str(invoice_row.get("date", ""))
        inv_gstin = str(invoice_row.get("gstin", "")).strip().upper()

        # Test 0: Identity — GSTIN conflicts are never auto-resolved
        gstin_conflicts = []
        for c in candidate_rows:
            cand_gstin = str(c.get("gstin", "")).strip().upper()
            if cand_gstin and inv_gstin and cand_gstin != inv_gstin:
                gstin_conflicts.append(cand_gstin)
        if gstin_conflicts:
            is_evidence_valid = False
            validation_tests.append({
                "test": "GSTIN_IDENTITY_CHECK",
                "passed": False,
                "reason": f"GSTIN conflict: invoice {inv_gstin} vs candidate(s) {gstin_conflicts}. Different legal entities must not be matched."
            })
        elif candidate_rows:
            validation_tests.append({
                "test": "GSTIN_IDENTITY_CHECK",
                "passed": True,
                "reason": "Candidate GSTIN matches invoice GSTIN."
            })

        # Ambiguous multi-candidate matches (except verified splits) must abstain
        is_split_claim = "split" in exception_type.lower() or "split" in root_cause.lower()
        if len(candidate_rows) >= 2 and not is_split_claim:
            is_evidence_valid = False
            validation_tests.append({
                "test": "AMBIGUOUS_CANDIDATE_CHECK",
                "passed": False,
                "reason": f"{len(candidate_rows)} equally plausible candidates; refusing to pick the first one."
            })

        # Test 1: Check candidate existence
        if not candidate_rows:
            if raw_decision == "MATCH":
                is_evidence_valid = False
                validation_tests.append({
                    "test": "CANDIDATE_EXISTENCE",
                    "passed": False,
                    "reason": "AI proposed a MATCH but zero matching candidate records exist in ledger."
                })
            else:
                validation_tests.append({
                    "test": "CANDIDATE_EXISTENCE",
                    "passed": True,
                    "reason": "No candidate records found in ledger (confirmed missing/unmatched)."
                })

        # Test 2: Split Invoice Math Verification
        elif len(candidate_rows) >= 2 or "split" in exception_type.lower() or "split" in root_cause.lower():
            cand_amts = [float(c.get("amount", 0.0)) for c in candidate_rows]
            passed, msg, diff = self.validator.verify_split_invoice(inv_amt, cand_amts)
            validation_tests.append({
                "test": "SPLIT_INVOICE_SUM_CHECK",
                "passed": passed,
                "reason": msg,
                "diff": diff
            })
            if not passed:
                is_evidence_valid = False

        # Test 3: Payment Gateway Fee Verification
        elif "fee" in exception_type.lower() or "gateway" in root_cause.lower() or "settlement" in root_cause.lower():
            cand_amt = float(candidate_rows[0].get("amount", 0.0))
            passed, msg, diff = self.validator.verify_gateway_fee(inv_amt, cand_amt)
            validation_tests.append({
                "test": "GATEWAY_FEE_ARITHMETIC_CHECK",
                "passed": passed,
                "reason": msg,
                "diff": diff
            })
            if not passed:
                is_evidence_valid = False

        # Test 4: Vendor Typo / String Distance Verification
        elif "typo" in exception_type.lower() or "typo" in root_cause.lower() or "vendor" in exception_type.lower():
            cand_vendor = str(candidate_rows[0].get("vendor_name", ""))
            passed, msg, dist = self.validator.verify_typo_or_format(inv_vendor, cand_vendor, max_dist=3)
            validation_tests.append({
                "test": "VENDOR_STRING_DISTANCE_CHECK",
                "passed": passed,
                "reason": msg,
                "distance": dist
            })
            if not passed:
                is_evidence_valid = False

        # Test 5: Date Window Verification
        if candidate_rows:
            cand_date = str(candidate_rows[0].get("date", ""))
            passed, msg, delta = self.validator.verify_date_window(inv_date, cand_date, max_days=60)
            validation_tests.append({
                "test": "DATE_WINDOW_CHECK",
                "passed": passed,
                "reason": msg,
                "days_diff": delta
            })
            if not passed:
                is_evidence_valid = False

        # Determine Final Action & Decision using Safety Policy
        if raw_decision == "ABSTAIN":
            final_decision = "ABSTAIN"
            rec_action = "HUMAN_REVIEW"
            status = "ESCALATED_TO_HUMAN"
            validated_conf = 0.50
            audit_notes = f"ABSTAINED: Ambiguous candidate collision or conflicting evidence ({root_cause}). Escalated to human review queue."

        elif not is_evidence_valid:
            final_decision = "ABSTAIN"
            rec_action = "HUMAN_REVIEW"
            status = "ESCALATED_TO_HUMAN"
            validated_conf = min(raw_conf, 0.60)
            audit_notes = f"ABSTAINED: Evidence validation failed ({'; '.join([t['reason'] for t in validation_tests if not t['passed']])}). Escalated for manual verification."
            
        elif raw_conf >= self.conf_auto and is_evidence_valid and raw_decision == "MATCH":
            # Extra safety: unexplained amount gaps cannot auto-resolve
            if candidate_rows and not is_split_claim:
                gap = abs(float(candidate_rows[0].get("amount", 0.0)) - inv_amt)
                gw_pass = False
                if gap > 1.0:
                    gw_pass, _, _ = self.validator.verify_gateway_fee(inv_amt, float(candidate_rows[0].get("amount", 0.0)))
                    if not gw_pass:
                        final_decision = "ABSTAIN"
                        rec_action = "HUMAN_REVIEW"
                        status = "ESCALATED_TO_HUMAN"
                        validated_conf = min(raw_conf, 0.60)
                        audit_notes = "ABSTAINED: Amount difference is not explained by gateway MDR/GST math or split sums."
                        return {
                            "final_decision": final_decision,
                            "status": status,
                            "validated_confidence": validated_conf,
                            "exception_category": exception_type,
                            "root_cause": root_cause,
                            "evidence_validation_passed": False,
                            "evidence_validation_tests": validation_tests + [{
                                "test": "UNEXPLAINED_AMOUNT_GAP",
                                "passed": False,
                                "reason": audit_notes,
                                "diff": gap
                            }],
                            "recommended_action": rec_action,
                            "audit_notes": audit_notes
                        }

            final_decision = "AUTO_RESOLVE"
            rec_action = "AUTO_RESOLVE"
            status = "RESOLVED"
            validated_conf = raw_conf
            audit_notes = f"AUTO-RESOLVED: High confidence ({raw_conf:.2%}) supported by verified mathematical & structural evidence."
            
        elif raw_conf >= self.conf_review and is_evidence_valid and raw_decision == "MATCH":
            final_decision = "HUMAN_REVIEW"
            rec_action = "HUMAN_REVIEW"
            status = "PENDING_REVIEW"
            validated_conf = raw_conf
            audit_notes = f"HUMAN REVIEW: Plausible match with confidence ({raw_conf:.2%}) requiring controller sign-off."
            
        elif raw_decision == "REVIEW":
            final_decision = "HUMAN_REVIEW"
            rec_action = "HUMAN_REVIEW"
            status = "PENDING_REVIEW"
            validated_conf = raw_conf
            audit_notes = f"HUMAN REVIEW: AI flagged ambiguity: {root_cause}"
            
        elif raw_decision == "NO_MATCH" or not candidate_rows:
            final_decision = "LEAVE_UNRESOLVED"
            rec_action = "LEAVE_UNRESOLVED"
            status = "UNRESOLVED_EXCEPTION"
            validated_conf = 1.0 if not candidate_rows else raw_conf
            audit_notes = f"UNRESOLVED: No matching counterparty record found in buyer ledger ({exception_type})."
            
        else:
            final_decision = "ABSTAIN"
            rec_action = "HUMAN_REVIEW"
            status = "ESCALATED_TO_HUMAN"
            validated_conf = 0.50
            audit_notes = f"ABSTAINED: Insufficient confidence ({raw_conf:.2%}) or inconclusive evidence."

        return {
            "final_decision": final_decision,
            "status": status,
            "validated_confidence": validated_conf,
            "exception_category": exception_type,
            "root_cause": root_cause,
            "evidence_validation_passed": is_evidence_valid,
            "evidence_validation_tests": validation_tests,
            "recommended_action": rec_action,
            "audit_notes": audit_notes
        }
