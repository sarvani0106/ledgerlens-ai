"""
llm_matcher.py
Stage 3 AI Investigation Agent for Ambiguous Financial Leftovers.

Features:
  - Integration with Anthropic Claude (claude-sonnet-4-6)
  - Strict JSON structured output with evidence objects
  - Built-in offline forensic evaluator for zero-config demos
  - Failure Injection Simulation modes (TIMEOUT, INVALID_JSON, UNAVAILABLE, HALLUCINATION)
  - Programmatic Evidence Validation routing for zero-hallucination guarantees
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from audit_logger import AuditLogger
from exception_resolver import ExceptionResolver, compute_levenshtein


def _clean_json_string(text: str) -> str:
    """Strip markdown code blocks or stray formatting from LLM response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class LLMInvestigationAgent:
    """
    AI Investigation Agent for analyzing complex and ambiguous financial discrepancies.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-sonnet-4-6",
        confidence_auto_resolve: float = 0.95,
        confidence_review_min: float = 0.75,
        enable_offline_fallback: bool = True,
        simulate_failure: Optional[str] = None  # None | "TIMEOUT" | "INVALID_JSON" | "UNAVAILABLE" | "HALLUCINATION"
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model_name = model_name
        self.resolver = ExceptionResolver(confidence_auto_resolve, confidence_review_min)
        self.enable_offline_fallback = enable_offline_fallback
        self.simulate_failure = simulate_failure

    def _build_prompt(self, unmatched_invoices: pd.DataFrame, unmatched_ledger: pd.DataFrame) -> str:
        """Construct structured prompt for Claude Sonnet."""
        inv_payload = unmatched_invoices[["gstin", "invoice_no", "vendor_name", "amount", "tax_rate", "date"]].to_dict(orient="records")
        led_payload = unmatched_ledger[["gstin", "invoice_no", "vendor_name", "amount", "tax_rate", "date"]].to_dict(orient="records")

        prompt = f"""You are a specialized AI Finance Controller performing forensic tax-line reconciliation.
Investigate the leftover unmatched seller invoices against buyer purchase ledger entries.

OBJECTIVES & DISCREPANCY SCENARIOS TO INVESTIGATE:
1. Vendor Typos: Swapped letters or spelling variations with matching GSTIN/amount.
2. Split Invoices: Single invoice split into 2+ ledger entries (e.g. -A and -B suffixes) summing to invoice total.
3. Payment Gateway Fee: Gross amount minus 2% MDR fee and 18% GST on fee (e.g. ₹10,000 gross becomes ₹9,764 net settlement).
4. GST Rate Difference: Seller billed at 18% vs buyer booked at 12% (or vice versa) on same base amount.
5. Ambiguous Collisions / Multiple Candidates: Multiple plausible matches -> Set decision to "ABSTAIN" or "REVIEW".
6. Missing Records: If no valid counterparty exists, set decision to "NO_MATCH".

STRICT OUTPUT FORMAT:
Respond with ONLY a JSON array of investigation objects matching this exact schema:
[
  {{
    "invoice_no": "INV-2024-XXXX",
    "decision": "MATCH | NO_MATCH | REVIEW | ABSTAIN",
    "confidence": 0.95,
    "exception_type": "SPLIT_INVOICE | PARTIAL_PAYMENT_OR_FEE_DEDUCTION | INVOICE_NUMBER_TYPO | GST_CALCULATION_DIFFERENCE | MULTIPLE_POSSIBLE_MATCHES | MISSING_SOURCE_RECORD",
    "root_cause": "One concise line describing the factual discrepancy and evidence",
    "candidate_record_ids": ["LEDGER-INV-1", "LEDGER-INV-2"],
    "evidence": [
      {{
        "field": "amount",
        "source_value": "50000.00",
        "target_value": "30000.00 + 20000.00",
        "reason": "Split payments sum matches invoice amount"
      }}
    ],
    "recommended_action": "AUTO_RESOLVE | HUMAN_REVIEW | ABSTAIN | LEAVE_UNRESOLVED"
  }}
]

DATA TO RECONCILE:

Unmatched Seller Invoices ({len(inv_payload)} items):
{json.dumps(inv_payload, indent=2)}

Unmatched Buyer Ledger Records ({len(led_payload)} items):
{json.dumps(led_payload, indent=2)}
"""
        return prompt

    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API (claude-sonnet-4-6) with failure simulation hooks."""
        if self.simulate_failure == "TIMEOUT":
            raise TimeoutError("Anthropic API call timed out after 30000ms (Simulated Fault Injection).")
        elif self.simulate_failure == "UNAVAILABLE":
            raise ConnectionError("503 Service Unavailable: Anthropic Claude API cluster unreachable (Simulated Outage).")
        elif self.simulate_failure == "INVALID_JSON":
            return "MALFORMED_RESPONSE: { invoice_no: 'INV-1001', incomplete_json..."

        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model_name,
            max_tokens=4000,
            temperature=0.0,
            system="You are an expert AI Finance Controller. Output strict JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        full_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                full_text += block.text
        return full_text

    def _offline_heuristic_investigator(
        self,
        unmatched_invoices: pd.DataFrame,
        unmatched_ledger: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Intelligent forensic investigation engine for offline evaluation.
        """
        proposals = []
        used_led_ids = set()

        # Check for Fault Injection simulation
        if self.simulate_failure == "HALLUCINATION":
            # Adversarial simulation: AI proposes a high-confidence match on invalid amounts
            if not unmatched_invoices.empty and not unmatched_ledger.empty:
                inv = unmatched_invoices.iloc[0]
                led = unmatched_ledger.iloc[0]
                proposals.append({
                    "invoice_no": inv["invoice_no"],
                    "decision": "MATCH",
                    "confidence": 0.99,  # High confidence claim
                    "exception_type": "SPLIT_INVOICE",
                    "root_cause": "Adversarial Hallucination Injection: LLM falsely claiming match.",
                    "candidate_record_ids": [led["invoice_no"]],
                    "evidence": [
                        {"field": "amount", "source_value": str(inv["amount"]), "target_value": str(led["amount"]), "reason": "Hallucinated claim that amounts match"}
                    ],
                    "recommended_action": "AUTO_RESOLVE"
                })

        for _, inv in unmatched_invoices.iterrows():
            inv_no = str(inv["invoice_no"])
            if any(p["invoice_no"] == inv_no for p in proposals):
                continue

            inv_amt = float(inv["amount"])
            gstin = str(inv["gstin"]).strip().upper()
            vendor = str(inv["vendor_name"])
            inv_date = str(inv.get("date", ""))

            remaining = unmatched_ledger[~unmatched_ledger["invoice_no"].isin(used_led_ids)]

            # 0. Same invoice number with conflicting GSTIN / vendor / amount -> ABSTAIN
            same_number = remaining[remaining["invoice_no"].astype(str) == inv_no]
            if not same_number.empty:
                gstin_mismatch = same_number["gstin"].astype(str).str.strip().str.upper() != gstin
                if gstin_mismatch.any():
                    proposals.append({
                        "invoice_no": inv_no,
                        "decision": "ABSTAIN",
                        "confidence": 0.40,
                        "exception_type": "INSUFFICIENT_EVIDENCE",
                        "root_cause": "Invoice number collides with a ledger row from a different GSTIN/legal entity. Conflicting identity evidence.",
                        "candidate_record_ids": same_number["invoice_no"].astype(str).tolist(),
                        "evidence": [
                            {"field": "gstin", "source_value": gstin, "target_value": ", ".join(same_number["gstin"].astype(str).tolist()), "reason": "GSTIN conflict on shared invoice number"}
                        ],
                        "recommended_action": "ABSTAIN"
                    })
                    continue

            same_identity = remaining[
                (remaining["gstin"].astype(str).str.strip().str.upper() == gstin)
                & (remaining["invoice_no"].astype(str) == inv_no)
            ]

            # 1. Split invoices: only fragments of THIS invoice (INV-...-A / -B), same GSTIN
            splits = remaining[
                (remaining["gstin"].astype(str).str.strip().str.upper() == gstin)
                & remaining["invoice_no"].astype(str).str.startswith(f"{inv_no}-")
            ]
            if len(splits) >= 2:
                split_sum = round(float(splits["amount"].sum()), 2)
                if abs(split_sum - inv_amt) <= 1.0:
                    cand_ids = splits["invoice_no"].astype(str).tolist()
                    for cid in cand_ids:
                        used_led_ids.add(cid)
                    proposals.append({
                        "invoice_no": inv_no,
                        "decision": "MATCH",
                        "confidence": 0.98,
                        "exception_type": "SPLIT_INVOICE",
                        "root_cause": f"Seller invoice Rs{inv_amt:,.2f} distributed across {len(cand_ids)} ledger rows ({', '.join(cand_ids)}) summing to Rs{split_sum:,.2f}.",
                        "candidate_record_ids": cand_ids,
                        "evidence": [
                            {"field": "amount", "source_value": str(inv_amt), "target_value": f"{' + '.join([str(s) for s in splits['amount'].tolist()])} = {split_sum}", "reason": "Split lines sum matches invoice total"}
                        ],
                        "recommended_action": "AUTO_RESOLVE"
                    })
                    continue
                proposals.append({
                    "invoice_no": inv_no,
                    "decision": "ABSTAIN",
                    "confidence": 0.45,
                    "exception_type": "SPLIT_INVOICE",
                    "root_cause": f"Split fragments found but sum Rs{split_sum:,.2f} does not equal invoice Rs{inv_amt:,.2f}.",
                    "candidate_record_ids": splits["invoice_no"].astype(str).tolist(),
                    "evidence": [
                        {"field": "amount", "source_value": str(inv_amt), "target_value": str(split_sum), "reason": "Incomplete or conflicting split sum"}
                    ],
                    "recommended_action": "ABSTAIN"
                })
                continue
            if len(splits) == 1:
                fragment_amt = float(splits.iloc[0]["amount"])
                proposals.append({
                    "invoice_no": inv_no,
                    "decision": "ABSTAIN",
                    "confidence": 0.45,
                    "exception_type": "SPLIT_INVOICE",
                    "root_cause": f"Incomplete split: only fragment {splits.iloc[0]['invoice_no']} (Rs{fragment_amt:,.2f}) found; remaining amount unexplained.",
                    "candidate_record_ids": splits["invoice_no"].astype(str).tolist(),
                    "evidence": [
                        {"field": "amount", "source_value": str(inv_amt), "target_value": str(fragment_amt), "reason": "Missing split fragment"}
                    ],
                    "recommended_action": "ABSTAIN"
                })
                continue

            # 2. Payment Gateway MDR Fee Deduction — requires GSTIN + invoice identity
            expected_fee = round(inv_amt * 0.02, 2)
            expected_gst_fee = round(expected_fee * 0.18, 2)
            expected_net = round(inv_amt - expected_fee - expected_gst_fee, 2)
            gw_hits = same_identity[abs(same_identity["amount"].astype(float) - expected_net) <= 1.0]
            if len(gw_hits) == 1:
                led = gw_hits.iloc[0]
                l_no = str(led["invoice_no"])
                used_led_ids.add(l_no)
                proposals.append({
                    "invoice_no": inv_no,
                    "decision": "MATCH",
                    "confidence": 0.96,
                    "exception_type": "PARTIAL_PAYMENT_OR_FEE_DEDUCTION",
                    "root_cause": f"Payment gateway settlement deduction: Gross Rs{inv_amt:,.2f} - 2% MDR fee (Rs{expected_fee:,.2f}) - 18% GST on fee (Rs{expected_gst_fee:,.2f}) = Net Rs{expected_net:,.2f}.",
                    "candidate_record_ids": [l_no],
                    "evidence": [
                        {"field": "amount", "source_value": f"Rs{inv_amt:,.2f} (Gross)", "target_value": f"Rs{led['amount']:,.2f} (Net)", "reason": "Verified standard 2% gateway MDR + 18% GST formula"}
                    ],
                    "recommended_action": "AUTO_RESOLVE"
                })
                continue

            # 3. GST rate difference on the same invoice identity (before treating as unexplained partial)
            if len(same_identity) == 1:
                led = same_identity.iloc[0]
                l_no = str(led["invoice_no"])
                try:
                    base_amt = round(inv_amt / (1 + float(inv["tax_rate"]) / 100.0), 2)
                    led_base = round(float(led["amount"]) / (1 + float(led["tax_rate"]) / 100.0), 2)
                except Exception:
                    base_amt, led_base = inv_amt, float(led["amount"])
                if abs(base_amt - led_base) <= 2.0 and abs(float(inv["tax_rate"]) - float(led["tax_rate"])) > 0.01:
                    proposals.append({
                        "invoice_no": inv_no,
                        "decision": "REVIEW",
                        "confidence": 0.82,
                        "exception_type": "GST_CALCULATION_DIFFERENCE",
                        "root_cause": f"GST Rate Discrepancy: Seller billed {inv['tax_rate']}% (Rs{inv_amt:,.2f}) vs Buyer booked {led['tax_rate']}% (Rs{led['amount']:,.2f}) on identical base amount Rs{base_amt:,.2f}.",
                        "candidate_record_ids": [l_no],
                        "evidence": [
                            {"field": "tax_rate", "source_value": f"{inv['tax_rate']}%", "target_value": f"{led['tax_rate']}%", "reason": "Conflicting tax rate entries on same base value"}
                        ],
                        "recommended_action": "HUMAN_REVIEW"
                    })
                    continue

            # 4. Same identity, unexplained partial / conflicting amount -> ABSTAIN (do not guess)
            if len(same_identity) == 1:
                led = same_identity.iloc[0]
                amt_gap = abs(float(led["amount"]) - inv_amt)
                if amt_gap > 1.0:
                    proposals.append({
                        "invoice_no": inv_no,
                        "decision": "ABSTAIN",
                        "confidence": 0.50,
                        "exception_type": "PARTIAL_PAYMENT_OR_FEE_DEDUCTION",
                        "root_cause": f"Same invoice/GSTIN identity but amount gap Rs{amt_gap:,.2f} is not explained by gateway MDR+GST math. Refusing to auto-resolve a partial payment.",
                        "candidate_record_ids": [str(led["invoice_no"])],
                        "evidence": [
                            {"field": "amount", "source_value": str(inv_amt), "target_value": str(led["amount"]), "reason": "Unexplained partial payment / non-gateway deduction"}
                        ],
                        "recommended_action": "ABSTAIN"
                    })
                    continue

            # 4. Multiple ambiguous candidates (same GSTIN + amount, different invoice numbers)
            amb_mask = (
                (remaining["gstin"].astype(str).str.strip().str.upper() == gstin)
                & (abs(remaining["amount"].astype(float) - inv_amt) < 0.05)
                & (remaining["invoice_no"].astype(str) != inv_no)
                & (~remaining["invoice_no"].astype(str).str.startswith(f"{inv_no}-"))
            )
            amb_candidates = remaining.loc[amb_mask, "invoice_no"].astype(str).tolist()
            if len(amb_candidates) >= 2:
                proposals.append({
                    "invoice_no": inv_no,
                    "decision": "ABSTAIN",
                    "confidence": 0.50,
                    "exception_type": "MULTIPLE_POSSIBLE_MATCHES",
                    "root_cause": f"Multiple ambiguous ledger records ({', '.join(amb_candidates)}) share identical amount Rs{inv_amt:,.2f} and GSTIN.",
                    "candidate_record_ids": amb_candidates,
                    "evidence": [
                        {"field": "candidates", "source_value": inv_no, "target_value": str(amb_candidates), "reason": "Ambiguous candidate collision"}
                    ],
                    "recommended_action": "ABSTAIN"
                })
                continue

            # 5. Vendor name typos — same GSTIN, same amount, unique candidate, invoice identity or tiny invoice edit
            typo_pool = remaining[
                (remaining["gstin"].astype(str).str.strip().str.upper() == gstin)
                & (abs(remaining["amount"].astype(float) - inv_amt) < 0.05)
            ]
            if len(typo_pool) == 1:
                led = typo_pool.iloc[0]
                l_no = str(led["invoice_no"])
                dist = compute_levenshtein(vendor, led["vendor_name"])
                inv_dist = compute_levenshtein(inv_no, l_no)
                if dist <= 3 or inv_no == l_no:
                    if inv_no == l_no or inv_dist <= 2:
                        used_led_ids.add(l_no)
                        proposals.append({
                            "invoice_no": inv_no,
                            "decision": "MATCH",
                            "confidence": 0.96,
                            "exception_type": "INVOICE_NUMBER_TYPO" if inv_no != l_no else "GSTIN_TYPO_OR_FORMAT",
                            "root_cause": f"Vendor name typographical variation ('{vendor}' vs '{led['vendor_name']}') with identical GSTIN and amount Rs{inv_amt:,.2f}.",
                            "candidate_record_ids": [l_no],
                            "evidence": [
                                {"field": "vendor_name", "source_value": vendor, "target_value": led["vendor_name"], "reason": f"Levenshtein edit distance = {dist}"}
                            ],
                            "recommended_action": "AUTO_RESOLVE"
                        })
                        continue

            # 7. Similar GSTIN (edit distance 1) different legal entity -> do not match
            similar_gstin_hits = []
            for _, led in remaining.iterrows():
                other = str(led["gstin"]).strip().upper()
                if other != gstin and compute_levenshtein(gstin, other) == 1 and str(led["invoice_no"]) == inv_no:
                    similar_gstin_hits.append(str(led["invoice_no"]))
            if similar_gstin_hits:
                proposals.append({
                    "invoice_no": inv_no,
                    "decision": "NO_MATCH",
                    "confidence": 0.90,
                    "exception_type": "MISSING_SOURCE_RECORD",
                    "root_cause": "Near-identical GSTIN belongs to a different legal entity; source record for this GSTIN is missing.",
                    "candidate_record_ids": [],
                    "evidence": [
                        {"field": "gstin", "source_value": gstin, "target_value": "similar-but-different GSTIN", "reason": "SIMILAR_GSTIN_DIFFERENT_ENTITY"}
                    ],
                    "recommended_action": "LEAVE_UNRESOLVED"
                })
                continue

            # 8. Default: Missing Source Record
            proposals.append({
                "invoice_no": inv_no,
                "decision": "NO_MATCH",
                "confidence": 0.99,
                "exception_type": "MISSING_SOURCE_RECORD",
                "root_cause": f"Invoice {inv_no} has no counterpart entry in buyer purchase ledger.",
                "candidate_record_ids": [],
                "evidence": [],
                "recommended_action": "LEAVE_UNRESOLVED"
            })

        return proposals

    def investigate_and_resolve(
        self,
        unmatched_invoices: pd.DataFrame,
        unmatched_ledger: pd.DataFrame,
        audit_logger: Optional[AuditLogger] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
        """
        Runs Stage 3 AI Investigation followed immediately by Programmatic Evidence Validation.
        """
        if unmatched_invoices.empty:
            empty_df = pd.DataFrame(columns=[
                "invoice_no", "matched_ledger_ids", "gstin", "seller_vendor", "ledger_vendor",
                "invoice_amount", "ledger_amount", "amount_diff", "tax_rate",
                "invoice_date", "ledger_date", "match_stage", "match_type", "confidence",
                "resolution_status", "recommended_action", "evidence_validation_passed", "notes"
            ])
            return empty_df, unmatched_invoices, unmatched_ledger, []

        raw_proposals = []
        if self.api_key and not self.simulate_failure:
            try:
                prompt = self._build_prompt(unmatched_invoices, unmatched_ledger)
                raw_resp = self._call_anthropic_api(prompt)
                raw_proposals = json.loads(_clean_json_string(raw_resp))
            except Exception as e:
                if audit_logger:
                    audit_logger.log_lifecycle_step(
                        record_id="STAGE-3-AI",
                        lifecycle_phase="INVESTIGATE",
                        stage_name="Stage 3 (AI)",
                        action="API_CALL_ERROR",
                        status="ERROR",
                        details={"error": str(e), "fallback": "Using local heuristic investigator"}
                    )
                raw_proposals = self._offline_heuristic_investigator(unmatched_invoices, unmatched_ledger)
        else:
            raw_proposals = self._offline_heuristic_investigator(unmatched_invoices, unmatched_ledger)

        inv_dict = {row["invoice_no"]: row.to_dict() for _, row in unmatched_invoices.iterrows()}
        led_dict = {row["invoice_no"]: row.to_dict() for _, row in unmatched_ledger.iterrows()}

        resolved_records = []
        consumed_inv_nos = set()
        consumed_led_ids = set()
        investigation_reports = []

        for prop in raw_proposals:
            inv_no = prop.get("invoice_no")
            if inv_no not in inv_dict:
                continue

            inv_row = inv_dict[inv_no]
            cand_ids = prop.get("candidate_record_ids", [])
            cand_rows = [led_dict[cid] for cid in cand_ids if cid in led_dict]

            # Step 1: Log INVESTIGATE
            if audit_logger:
                audit_logger.log_lifecycle_step(
                    record_id=inv_no,
                    lifecycle_phase="INVESTIGATE",
                    stage_name="Stage 3 (AI)",
                    action="AI_HYPOTHESIS_GENERATION",
                    status="SUCCESS",
                    confidence=prop.get("confidence"),
                    candidate_ids=cand_ids,
                    details={"hypothesis": prop.get("root_cause"), "exception_type": prop.get("exception_type")}
                )

            # Step 2: Programmatic Evidence Validation
            eval_result = self.resolver.evaluate_ai_proposal(
                invoice_row=inv_row,
                candidate_rows=cand_rows,
                ai_proposal=prop
            )

            # Step 3: Log VERIFY
            if audit_logger:
                audit_logger.log_lifecycle_step(
                    record_id=inv_no,
                    lifecycle_phase="VERIFY",
                    stage_name="Evidence Validation Layer",
                    action="PROGRAMMATIC_EVIDENCE_CHECK",
                    status="SUCCESS" if eval_result["evidence_validation_passed"] else "REJECTED",
                    confidence=eval_result["validated_confidence"],
                    candidate_ids=cand_ids,
                    evidence_summary=eval_result["evidence_validation_tests"],
                    details={"tests": eval_result["evidence_validation_tests"]}
                )

            # Step 4: Log DECIDE & ACT
            if audit_logger:
                audit_logger.log_lifecycle_step(
                    record_id=inv_no,
                    lifecycle_phase="DECIDE",
                    stage_name="Confidence & Safety Policy",
                    action="FINAL_RESOLUTION_DECISION",
                    status=eval_result["status"],
                    confidence=eval_result["validated_confidence"],
                    details={"decision": eval_result["final_decision"], "notes": eval_result["audit_notes"]}
                )
                audit_logger.log_lifecycle_step(
                    record_id=inv_no,
                    lifecycle_phase="ACT",
                    stage_name="Resolution Routing",
                    action=eval_result["recommended_action"],
                    status=eval_result["status"],
                    confidence=eval_result["validated_confidence"],
                    candidate_ids=cand_ids,
                    details={"final_decision": eval_result["final_decision"]}
                )
                audit_logger.log_lifecycle_step(
                    record_id=inv_no,
                    lifecycle_phase="AUDIT",
                    stage_name="Audit",
                    action="HASH_LOG",
                    status="SUCCESS",
                    confidence=eval_result["validated_confidence"],
                    details={"exception_type": eval_result["exception_category"]}
                )

            total_led_amt = round(sum(float(c.get("amount", 0.0)) for c in cand_rows), 2) if cand_rows else 0.0
            amt_diff = round(total_led_amt - float(inv_row["amount"]), 2) if cand_rows else -float(inv_row["amount"])
            led_vendors = ", ".join(list(set(str(c.get("vendor_name", "")) for c in cand_rows))) if cand_rows else "N/A"
            led_dates = ", ".join(list(set(str(c.get("date", "")) for c in cand_rows))) if cand_rows else "N/A"

            rec = {
                "invoice_no": inv_no,
                "matched_ledger_ids": cand_ids,
                "gstin": inv_row["gstin"],
                "seller_vendor": inv_row["vendor_name"],
                "ledger_vendor": led_vendors,
                "invoice_amount": float(inv_row["amount"]),
                "ledger_amount": total_led_amt,
                "amount_diff": amt_diff,
                "tax_rate": float(inv_row["tax_rate"]),
                "invoice_date": str(inv_row["date"]),
                "ledger_date": led_dates,
                "match_stage": "Stage 3 (AI Investigation)",
                "match_type": eval_result["exception_category"],
                "confidence": eval_result["validated_confidence"],
                "resolution_status": eval_result["final_decision"],
                "recommended_action": eval_result["recommended_action"],
                "evidence_validation_passed": eval_result["evidence_validation_passed"],
                "notes": eval_result["audit_notes"],
                "root_cause": eval_result["root_cause"],
                "evidence_tests": eval_result["evidence_validation_tests"],
                "raw_ai_proposal": prop
            }
            resolved_records.append(rec)
            investigation_reports.append(rec)

            if eval_result["final_decision"] == "AUTO_RESOLVE":
                consumed_inv_nos.add(inv_no)
                for cid in cand_ids:
                    consumed_led_ids.add(cid)

        resolved_df = pd.DataFrame(resolved_records)
        rem_inv = unmatched_invoices[~unmatched_invoices["invoice_no"].isin(consumed_inv_nos)].copy()
        rem_led = unmatched_ledger[~unmatched_ledger["invoice_no"].isin(consumed_led_ids)].copy()

        return resolved_df, rem_inv, rem_led, investigation_reports
