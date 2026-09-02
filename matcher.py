"""
matcher.py
Deterministic Ingestion, Normalization, and Rule-Based Matching Layer for LedgerLens AI.

Lifecycle Phases handled:
  1. OBSERVE: Data validation and canonical normalization.
  2. MATCH:
     - Stage 1: Exact Key Matching (Normalized GSTIN + Invoice No) -> 100% confidence
     - Stage 2: Rule-Based & Tolerance Matching (Formatting, Rupee Tolerance, Date Windows)
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
from audit_logger import AuditLogger


def normalize_invoice_no(raw_no: Any) -> str:
    """
    Standardize invoice strings:
    - Uppercase & strip
    - Replace slashes, dots, underscores with hyphens
    - Strip leading zeroes from numeric components (e.g. INV-2024-001005 -> INV-2024-1005)
    """
    if pd.isna(raw_no):
        return ""
    s = str(raw_no).strip().upper()
    s = re.sub(r"[/._\s]+", "-", s)
    # Remove leading zeroes inside numeric segments like INV-2024-001042 -> INV-2024-1042
    parts = s.split("-")
    norm_parts = []
    for p in parts:
        if p.isdigit():
            norm_parts.append(str(int(p)))
        else:
            norm_parts.append(p)
    return "-".join(norm_parts)


def clean_and_normalize_dataframe(df: pd.DataFrame, source_name: str = "source") -> pd.DataFrame:
    """Validate schema and apply canonical normalization."""
    cleaned = df.copy()
    cleaned.columns = [c.strip().lower() for c in cleaned.columns]
    
    expected = ["gstin", "invoice_no", "vendor_name", "amount", "tax_rate", "date"]
    for col in expected:
        if col not in cleaned.columns:
            raise ValueError(f"Missing expected column '{col}' in {source_name} dataset")

    cleaned["_raw_invoice_no"] = cleaned["invoice_no"].astype(str)
    cleaned["_raw_gstin"] = cleaned["gstin"].astype(str)
    
    cleaned["gstin"] = cleaned["gstin"].astype(str).str.strip().str.upper()
    cleaned["norm_invoice_no"] = cleaned["invoice_no"].apply(normalize_invoice_no)
    cleaned["invoice_no"] = cleaned["invoice_no"].astype(str).str.strip().str.upper()
    cleaned["vendor_name"] = cleaned["vendor_name"].astype(str).str.strip()
    
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce").fillna(0.0).round(2)
    cleaned["tax_rate"] = pd.to_numeric(cleaned["tax_rate"], errors="coerce").fillna(0.0).round(2)
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    
    cleaned["_row_id"] = [f"{source_name}_{i}" for i in range(len(cleaned))]
    return cleaned


def run_deterministic_matching(
    invoices_df: pd.DataFrame,
    ledger_df: pd.DataFrame,
    amount_tolerance_abs: float = 15.0,
    amount_tolerance_pct: float = 0.01,
    audit_logger: Optional[AuditLogger] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Executes Stage 1 (Exact) and Stage 2 (Rule & Tolerance) matching.
    """
    inv_clean = clean_and_normalize_dataframe(invoices_df, "inv")
    led_clean = clean_and_normalize_dataframe(ledger_df, "led")

    if audit_logger:
        audit_logger.log_lifecycle_step(
            record_id="DATASET",
            lifecycle_phase="OBSERVE",
            stage_name="Ingestion",
            action="NORMALIZE",
            status="SUCCESS",
            details={
                "invoices": int(len(inv_clean)),
                "ledger": int(len(led_clean)),
                "columns_validated": ["gstin", "invoice_no", "vendor_name", "amount", "tax_rate", "date"],
            },
        )

    matched_records: List[Dict[str, Any]] = []
    used_inv_ids = set()
    used_led_ids = set()
    
    # -------------------------------------------------------------------------
    # STAGE 1: Exact Key Matching on (normalized_gstin, normalized_invoice_no)
    # -------------------------------------------------------------------------
    ledger_exact_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for _, l_row in led_clean.iterrows():
        key = (l_row["gstin"], l_row["norm_invoice_no"])
        ledger_exact_lookup.setdefault(key, []).append(l_row.to_dict())
        
    for _, inv_row in inv_clean.iterrows():
        inv_id = inv_row["_row_id"]
        key = (inv_row["gstin"], inv_row["norm_invoice_no"])
        
        candidates = ledger_exact_lookup.get(key, [])
        amount_ok = []
        for cand in candidates:
            if cand["_row_id"] in used_led_ids:
                continue
            # Exact identity is not enough: amount must also match.
            # Same invoice number with a large unexplained remainder (partial payment)
            # must not auto-resolve at Stage 1.
            if abs(float(cand["amount"]) - float(inv_row["amount"])) < 0.01:
                amount_ok.append(cand)

        # Multiple unused ledger rows with the same identity + amount is ambiguous.
        matched_cand = amount_ok[0] if len(amount_ok) == 1 else None

        if matched_cand is not None:
            used_inv_ids.add(inv_id)
            used_led_ids.add(matched_cand["_row_id"])
            amt_diff = round(matched_cand["amount"] - inv_row["amount"], 2)
            
            rec = {
                "invoice_no": inv_row["invoice_no"],
                "matched_ledger_ids": [matched_cand["invoice_no"]],
                "gstin": inv_row["gstin"],
                "seller_vendor": inv_row["vendor_name"],
                "ledger_vendor": matched_cand["vendor_name"],
                "invoice_amount": inv_row["amount"],
                "ledger_amount": matched_cand["amount"],
                "amount_diff": amt_diff,
                "tax_rate": inv_row["tax_rate"],
                "invoice_date": inv_row["date"],
                "ledger_date": matched_cand["date"],
                "match_stage": "Stage 1 (Exact)",
                "match_type": "EXACT_MATCH",
                "confidence": 1.0,
                "resolution_status": "AUTO_RESOLVED",
                "recommended_action": "AUTO_RESOLVE",
                "evidence_validation_passed": True,
                "notes": "Exact 1:1 match on normalized GSTIN and Invoice Number."
            }
            matched_records.append(rec)
            
            if audit_logger:
                audit_logger.log_lifecycle_step(
                    record_id=inv_row["invoice_no"],
                    lifecycle_phase="MATCH",
                    stage_name="Stage 1 (Exact)",
                    action="EXACT_KEY_MATCH",
                    status="SUCCESS",
                    confidence=1.0,
                    candidate_ids=[matched_cand["invoice_no"]],
                    details={"invoice_amount": inv_row["amount"], "ledger_amount": matched_cand["amount"]}
                )

    # Filter leftovers for Stage 2
    rem_inv = inv_clean[~inv_clean["_row_id"].isin(used_inv_ids)].copy()
    rem_led = led_clean[~led_clean["_row_id"].isin(used_led_ids)].copy()

    # -------------------------------------------------------------------------
    # STAGE 2: Rule & Tolerance Matching
    # Rule 2a: Formatting variations (/ vs -) with exact GSTIN & amount
    # Rule 2b: Rupee Tolerance (<= Rs 15 or 1%) with exact GSTIN & invoice no
    # Rule 2c: Date Window (within 30 days) with exact GSTIN & amount
    # -------------------------------------------------------------------------
    ledger_gstin_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for _, l_row in rem_led.iterrows():
        ledger_gstin_lookup.setdefault(l_row["gstin"], []).append(l_row.to_dict())

    for _, inv_row in rem_inv.iterrows():
        inv_id = inv_row["_row_id"]
        gstin = inv_row["gstin"]
        inv_amt = inv_row["amount"]
        inv_no = inv_row["invoice_no"]
        norm_no = inv_row["norm_invoice_no"]
        inv_date = inv_row["date"]
        
        candidates = ledger_gstin_lookup.get(gstin, [])
        plausible = []

        for cand in candidates:
            if cand["_row_id"] in used_led_ids:
                continue

            cand_amt = cand["amount"]
            cand_no = cand["invoice_no"]
            cand_norm = cand["norm_invoice_no"]
            cand_date = cand["date"]
            diff = abs(cand_amt - inv_amt)
            identity_match = (norm_no == cand_norm)

            # Identity is required. GSTIN+similar-amount without invoice identity is not a match.
            if not identity_match:
                continue

            days_diff = None
            try:
                d1 = datetime.strptime(str(inv_date)[:10], "%Y-%m-%d")
                d2 = datetime.strptime(str(cand_date)[:10], "%Y-%m-%d")
                days_diff = abs((d1 - d2).days)
            except Exception:
                days_diff = None

            # Rule 2a: Formatting variation with exact amount
            if diff < 0.05:
                if days_diff is not None and days_diff <= 30:
                    plausible.append((cand, "INVOICE_FORMATTING_VARIATION" if inv_no != cand_no else "DATE_DIFFERENCE", 0.99 if inv_no != cand_no else 0.95))
                    continue
                if days_diff is None:
                    plausible.append((cand, "INVOICE_FORMATTING_VARIATION", 0.99))
                    continue

            # Rule 2b: Small rupee tolerance with matching invoice identity
            tol = max(amount_tolerance_abs, inv_amt * amount_tolerance_pct)
            if diff <= tol:
                # Compounded date lag + rounding is plausible but not safe to auto-resolve.
                conf = 0.90 if (days_diff is not None and days_diff > 30) else 0.96
                plausible.append((cand, "ROUNDING_DIFFERENCE", conf))
                continue

        # Multiple equally plausible ledger rows -> do not pick the first one.
        if len(plausible) != 1:
            continue

        best_cand, best_rule, best_conf = plausible[0]

        if best_cand is not None:
            used_inv_ids.add(inv_id)
            used_led_ids.add(best_cand["_row_id"])
            amt_diff = round(best_cand["amount"] - inv_amt, 2)
            
            rec = {
                "invoice_no": inv_row["invoice_no"],
                "matched_ledger_ids": [best_cand["invoice_no"]],
                "gstin": inv_row["gstin"],
                "seller_vendor": inv_row["vendor_name"],
                "ledger_vendor": best_cand["vendor_name"],
                "invoice_amount": inv_row["amount"],
                "ledger_amount": best_cand["amount"],
                "amount_diff": amt_diff,
                "tax_rate": inv_row["tax_rate"],
                "invoice_date": inv_row["date"],
                "ledger_date": best_cand["date"],
                "match_stage": "Stage 2 (Tolerance/Rules)",
                "match_type": best_rule,
                "confidence": best_conf,
                "resolution_status": "AUTO_RESOLVED" if best_conf >= 0.95 else "HUMAN_REVIEW",
                "recommended_action": "AUTO_RESOLVE" if best_conf >= 0.95 else "HUMAN_REVIEW",
                "evidence_validation_passed": True,
                "notes": f"Deterministic rule match ({best_rule}) with amount variance ₹{amt_diff:+.2f}."
            }
            matched_records.append(rec)
            
            if audit_logger:
                audit_logger.log_lifecycle_step(
                    record_id=inv_row["invoice_no"],
                    lifecycle_phase="MATCH",
                    stage_name="Stage 2 (Tolerance/Rules)",
                    action=best_rule,
                    status="SUCCESS",
                    confidence=best_conf,
                    candidate_ids=[best_cand["invoice_no"]],
                    details={"rule": best_rule, "amount_diff": amt_diff}
                )

    # Remaining leftovers for Stage 3 (AI Investigation)
    unmatched_inv = inv_clean[~inv_clean["_row_id"].isin(used_inv_ids)].copy().drop(columns=["_row_id", "_raw_invoice_no", "_raw_gstin", "norm_invoice_no"])
    unmatched_led = led_clean[~led_clean["_row_id"].isin(used_led_ids)].copy().drop(columns=["_row_id", "_raw_invoice_no", "_raw_gstin", "norm_invoice_no"])

    matched_df = pd.DataFrame(matched_records)
    if matched_df.empty:
        matched_df = pd.DataFrame(columns=[
            "invoice_no", "matched_ledger_ids", "gstin", "seller_vendor", "ledger_vendor",
            "invoice_amount", "ledger_amount", "amount_diff", "tax_rate",
            "invoice_date", "ledger_date", "match_stage", "match_type", "confidence",
            "resolution_status", "recommended_action", "evidence_validation_passed", "notes"
        ])

    stage_stats = {
        "stage1_exact_count": len(matched_df[matched_df["match_stage"] == "Stage 1 (Exact)"]),
        "stage2_rules_count": len(matched_df[matched_df["match_stage"] == "Stage 2 (Tolerance/Rules)"]),
        "total_deterministic_matched": len(matched_df),
        "unmatched_invoices_for_stage3": len(unmatched_inv),
        "unmatched_ledger_for_stage3": len(unmatched_led)
    }

    return matched_df, unmatched_inv, unmatched_led, stage_stats
