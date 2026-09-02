"""
generate_data.py
Enhanced Synthetic Financial Dataset Generator for LedgerLens AI.

Generates:
  1. Standard Benchmark Dataset (140+ rows, seed=42):
     Balanced enterprise GST reconciliation scenarios.
  2. Held-Out Challenge Dataset (40 rows, seed=101):
     Difficult adversarial scenarios (similar GSTIN collisions, missing split fragments,
     partial payments vs gateway fees, conflicting dates, ambiguous collisions, and intentional ABSTAIN cases).

Includes hidden ground-truth metadata for honest Precision, Recall, and Confusion Matrix calculation.
All entities, GSTINs, and transactions are 100% synthetic for safe public release.
"""

import os
import random
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List
import pandas as pd

DATASET_VERSION = "v2.0-Buildathon-Release"

__all__ = [
    "generate_synthetic_datasets",
    "generate_held_out_challenge_dataset",
    "save_all_datasets",
    "save_synthetic_data",
    "introduce_typo",
    "DATASET_VERSION",
    "SAMPLE_VENDORS",
    "TAX_RATES"
]

# Completely fictional enterprise vendor identities with synthetic GSTINs
SAMPLE_VENDORS = [
    {"name": "NovaTech Solutions Pvt Ltd", "gstin": "27SYNTC1234A1Z5", "state": "Maharashtra"},
    {"name": "Vertex Business Systems Pvt Ltd", "gstin": "29SYNVR5678B1Z6", "state": "Karnataka"},
    {"name": "BluePeak Technologies Pvt Ltd", "gstin": "27SYNBP9012C1Z7", "state": "Maharashtra"},
    {"name": "Apex Retail Network Pvt Ltd", "gstin": "27SYNAP3456D1Z8", "state": "Maharashtra"},
    {"name": "Orion Industrial Services Pvt Ltd", "gstin": "27SYNOI7890E1Z9", "state": "Maharashtra"},
    {"name": "Silverline Components Pvt Ltd", "gstin": "29SYNSL2345F1Z0", "state": "Karnataka"},
    {"name": "Quantum Office Supplies Pvt Ltd", "gstin": "27SYNQO6789G1Z1", "state": "Maharashtra"},
    {"name": "GreenGrid Infrastructure Pvt Ltd", "gstin": "27SYNGG0123H1Z2", "state": "Maharashtra"},
    {"name": "Astra Global Logistics Pvt Ltd", "gstin": "24SYNAE4567I1Z3", "state": "Gujarat"},
    {"name": "Horizon Hospitality & FMCG Ltd", "gstin": "19SYNHH8901J1Z4", "state": "West Bengal"},
    {"name": "Zenith Consumer Products Pvt Ltd", "gstin": "27SYNZC2345K1Z5", "state": "Maharashtra"},
    {"name": "Matrix Digital Services Pvt Ltd", "gstin": "27SYNMD6789L1Z6", "state": "Maharashtra"},
    {"name": "Sterling Auto Components Ltd", "gstin": "27SYNSA0123M1Z7", "state": "Maharashtra"},
    {"name": "Solaris Healthcare & Pharma Ltd", "gstin": "24SYNSH4567N1Z8", "state": "Gujarat"},
    {"name": "Prism Paints & Coatings Pvt Ltd", "gstin": "27SYNPP8901P1Z9", "state": "Maharashtra"},
]

TAX_RATES = [5.0, 12.0, 18.0, 28.0]


def introduce_typo(text: str) -> str:
    """Swap two adjacent alphabetic characters to simulate human entry typo."""
    words = text.split()
    eligible_words = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not eligible_words:
        chars = list(text)
        for i in range(len(chars) - 1):
            if chars[i].isalpha() and chars[i + 1].isalpha() and chars[i].lower() != chars[i + 1].lower():
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                return "".join(chars)
        return text

    word_idx = random.choice(eligible_words)
    target_word = list(words[word_idx])
    swap_pos = random.randint(1, len(target_word) - 2)
    target_word[swap_pos], target_word[swap_pos + 1] = target_word[swap_pos + 1], target_word[swap_pos]
    words[word_idx] = "".join(target_word)
    return " ".join(words)


def generate_synthetic_datasets(num_invoices: int = 140, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate Standard Benchmark dataset with ground truth.
    """
    random.seed(seed)
    start_date = datetime(2024, 1, 1)

    invoices = []
    ledger = []
    ground_truth: Dict[str, Dict[str, Any]] = {}

    categories = (
        ["EXACT"] * 68 +
        ["FORMATTING"] * 11 +
        ["TYPO"] * 11 +
        ["ROUNDING"] * 10 +
        ["GST_DIFF"] * 7 +
        ["DATE_DIFF"] * 8 +
        ["SPLIT"] * 7 +
        ["GATEWAY_FEE"] * 6 +
        ["MISSING"] * 5 +
        ["DUPLICATE"] * 3 +
        ["AMBIGUOUS_COLLISION"] * 4
    )
    
    if len(categories) != num_invoices:
        ratio = num_invoices / len(categories)
        categories = (
            ["EXACT"] * max(1, int(68 * ratio)) +
            ["FORMATTING"] * max(1, int(11 * ratio)) +
            ["TYPO"] * max(1, int(11 * ratio)) +
            ["ROUNDING"] * max(1, int(10 * ratio)) +
            ["GST_DIFF"] * max(1, int(7 * ratio)) +
            ["DATE_DIFF"] * max(1, int(8 * ratio)) +
            ["SPLIT"] * max(1, int(7 * ratio)) +
            ["GATEWAY_FEE"] * max(1, int(6 * ratio)) +
            ["MISSING"] * max(1, int(5 * ratio)) +
            ["DUPLICATE"] * max(1, int(3 * ratio)) +
            ["AMBIGUOUS_COLLISION"] * max(1, int(4 * ratio))
        )
        while len(categories) < num_invoices:
            categories.append("EXACT")
        categories = categories[:num_invoices]

    random.shuffle(categories)

    for idx, category in enumerate(categories, start=1):
        vendor = random.choice(SAMPLE_VENDORS)
        inv_no = f"INV-2024-{1000 + idx}"
        date_obj = start_date + timedelta(days=random.randint(0, 90))
        date_str = date_obj.strftime("%Y-%m-%d")
        tax_rate = random.choice(TAX_RATES)
        base_amount = round(random.uniform(5000, 250000), 2)
        total_amount = round(base_amount * (1 + tax_rate / 100.0), 2)

        inv_row = {
            "gstin": vendor["gstin"],
            "invoice_no": inv_no,
            "vendor_name": vendor["name"],
            "amount": total_amount,
            "tax_rate": tax_rate,
            "date": date_str
        }
        invoices.append(inv_row)

        gt_entry = {
            "invoice_no": inv_no,
            "amount": total_amount,
            "category": category,
            "expected_decision": "MATCH",
            "expected_action": "AUTO_RESOLVE",
            "expected_ledger_ids": [inv_no],
            "notes": ""
        }

        if category == "EXACT":
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": total_amount,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["notes"] = "Exact 1:1 match."

        elif category == "FORMATTING":
            fmt_variant = inv_no.replace("-", "/")
            if random.random() > 0.5:
                fmt_variant = f"INV/2024/00{1000 + idx}"
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": fmt_variant,
                "vendor_name": vendor["name"],
                "amount": total_amount,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["expected_ledger_ids"] = [fmt_variant]
            gt_entry["notes"] = f"Format variation ({inv_no} vs {fmt_variant})."

        elif category == "TYPO":
            typo_name = introduce_typo(vendor["name"])
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": typo_name,
                "amount": total_amount,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["notes"] = f"Vendor typo: '{vendor['name']}' vs '{typo_name}'."

        elif category == "ROUNDING":
            delta = random.choice([-14.50, -9.00, -3.50, -0.75, 0.50, 2.50, 7.80, 12.00, 14.50])
            ledger_amt = round(max(100.0, total_amount + delta), 2)
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": ledger_amt,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["notes"] = f"Rounding difference: ₹{delta:+.2f} variance."

        elif category == "GST_DIFF":
            alt_rate = 12.0 if tax_rate == 18.0 else 18.0
            alt_total = round(base_amount * (1 + alt_rate / 100.0), 2)
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": alt_total,
                "tax_rate": alt_rate,
                "date": date_str
            })
            gt_entry["expected_action"] = "HUMAN_REVIEW"
            gt_entry["notes"] = f"GST rate mismatch ({tax_rate}% vs {alt_rate}%)."

        elif category == "DATE_DIFF":
            lag_days = random.randint(5, 20)
            lagged_date = (date_obj + timedelta(days=lag_days)).strftime("%Y-%m-%d")
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": total_amount,
                "tax_rate": tax_rate,
                "date": lagged_date
            })
            gt_entry["notes"] = f"Date posting lag of {lag_days} days."

        elif category == "SPLIT":
            split_pct = random.choice([0.40, 0.50, 0.60, 0.35])
            p1_amt = round(total_amount * split_pct, 2)
            p2_amt = round(total_amount - p1_amt, 2)
            led_a = f"{inv_no}-A"
            led_b = f"{inv_no}-B"
            ledger.extend([
                {"gstin": vendor["gstin"], "invoice_no": led_a, "vendor_name": vendor["name"], "amount": p1_amt, "tax_rate": tax_rate, "date": date_str},
                {"gstin": vendor["gstin"], "invoice_no": led_b, "vendor_name": vendor["name"], "amount": p2_amt, "tax_rate": tax_rate, "date": (date_obj + timedelta(days=3)).strftime("%Y-%m-%d")}
            ])
            gt_entry["expected_ledger_ids"] = [led_a, led_b]
            gt_entry["notes"] = f"Split into {led_a} + {led_b}."

        elif category == "GATEWAY_FEE":
            fee = round(total_amount * 0.02, 2)
            gst_fee = round(fee * 0.18, 2)
            net_settlement = round(total_amount - fee - gst_fee, 2)
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": net_settlement,
                "tax_rate": tax_rate,
                "date": (date_obj + timedelta(days=2)).strftime("%Y-%m-%d")
            })
            gt_entry["notes"] = f"Gateway MDR deduction (Gross ₹{total_amount} -> Net ₹{net_settlement})."

        elif category == "MISSING":
            gt_entry["expected_decision"] = "NO_MATCH"
            gt_entry["expected_action"] = "LEAVE_UNRESOLVED"
            gt_entry["expected_ledger_ids"] = []
            gt_entry["notes"] = "Missing from buyer purchase ledger."

        elif category == "DUPLICATE":
            ledger.extend([
                {"gstin": vendor["gstin"], "invoice_no": inv_no, "vendor_name": vendor["name"], "amount": total_amount, "tax_rate": tax_rate, "date": date_str},
                {"gstin": vendor["gstin"], "invoice_no": f"{inv_no}-DUP", "vendor_name": vendor["name"], "amount": total_amount, "tax_rate": tax_rate, "date": (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")}
            ])
            gt_entry["notes"] = "Duplicate booking in buyer ledger."

        elif category == "AMBIGUOUS_COLLISION":
            ledger.extend([
                {"gstin": vendor["gstin"], "invoice_no": f"AMB-CAND-1-{idx}", "vendor_name": vendor["name"], "amount": total_amount, "tax_rate": tax_rate, "date": date_str},
                {"gstin": vendor["gstin"], "invoice_no": f"AMB-CAND-2-{idx}", "vendor_name": vendor["name"], "amount": total_amount, "tax_rate": tax_rate, "date": date_str}
            ])
            gt_entry["expected_decision"] = "ABSTAIN"
            gt_entry["expected_action"] = "HUMAN_REVIEW"
            gt_entry["expected_ledger_ids"] = [f"AMB-CAND-1-{idx}", f"AMB-CAND-2-{idx}"]
            gt_entry["notes"] = "Multiple ambiguous candidates with identical GSTIN and amount."

        ground_truth[inv_no] = gt_entry

    orphan_records = [
        {"gstin": "27SYNTC1234A1Z5", "invoice_no": "EXP-MKTG-2024-91", "vendor_name": "NovaTech Solutions Pvt Ltd", "amount": 18500.0, "tax_rate": 18.0, "date": "2024-02-14"},
        {"gstin": "29SYNDO9999Z1ZX", "invoice_no": "PUR-OFFICE-8821", "vendor_name": "Direct Office Hub", "amount": 4250.0, "tax_rate": 12.0, "date": "2024-03-10"},
        {"gstin": "27SYNAP3456D1Z8", "invoice_no": "EXP-LOGISTICS-404", "vendor_name": "Apex Retail Network Pvt Ltd", "amount": 32000.0, "tax_rate": 18.0, "date": "2024-01-28"},
        {"gstin": "33SYNST1234A1Z1", "invoice_no": "UNBILLED-VENDOR-11", "vendor_name": "Southern Freight Transport", "amount": 12400.0, "tax_rate": 5.0, "date": "2024-03-22"},
    ]
    ledger.extend(orphan_records)

    invoices_df = pd.DataFrame(invoices)
    ledger_df = pd.DataFrame(ledger)
    invoices_df["amount"] = invoices_df["amount"].astype(float)
    invoices_df["tax_rate"] = invoices_df["tax_rate"].astype(float)
    ledger_df["amount"] = ledger_df["amount"].astype(float)
    ledger_df["tax_rate"] = ledger_df["tax_rate"].astype(float)

    meta = {
        "dataset_type": "Standard Benchmark Dataset",
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "ground_truth": ground_truth,
        "total_invoices": len(invoices_df),
        "total_ledger": len(ledger_df),
        "orphans_count": len(orphan_records),
        "categories_breakdown": {cat: categories.count(cat) for cat in set(categories)}
    }
    return invoices_df, ledger_df, meta


def generate_held_out_challenge_dataset(num_invoices: int = 40, seed: int = 101) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate the Held-Out Challenge Dataset with 40 difficult adversarial scenarios:
      1. Similar GSTIN values belonging to different legal entities
      2. Split invoices with ONE missing fragment (partial sum)
      3. Partial payment vs gateway fee confusion (random partial amount)
      4. Amounts within tolerance but completely different entities/dates
      5. Conflicting evidence between amount, date, GSTIN, and invoice number
      6. Duplicate-looking transactions that are actually distinct purchases
      7. Intentional ABSTAIN collision cases
      8. Missing source records
      9. Compounded date lag + rounding difference
      10. Clean exact benchmark records
    """
    random.seed(seed)
    start_date = datetime(2024, 2, 1)

    invoices = []
    ledger = []
    ground_truth: Dict[str, Dict[str, Any]] = {}

    challenge_scenarios = [
        "SIMILAR_GSTIN_DIFFERENT_ENTITY",
        "SPLIT_WITH_MISSING_FRAGMENT",
        "PARTIAL_PAYMENT_NON_GATEWAY",
        "TOLERANCE_COLLISION_DIFFERENT_ENTITY",
        "CONFLICTING_EVIDENCE_ABSTAIN",
        "LEGITIMATE_SEPARATE_DUPLICATE_LOOKING",
        "AMBIGUOUS_MULTI_CANDIDATE_ABSTAIN",
        "MISSING_SOURCE_RECORD",
        "DATE_AND_ROUNDING_COMPOUNDED",
        "EXACT_CLEAN_BENCHMARK"
    ] * 4  # 40 items total

    for idx, scenario in enumerate(challenge_scenarios, start=1):
        vendor = random.choice(SAMPLE_VENDORS)
        inv_no = f"CHALLENGE-2024-{2000 + idx}"
        date_obj = start_date + timedelta(days=random.randint(0, 45))
        date_str = date_obj.strftime("%Y-%m-%d")
        tax_rate = 18.0
        base_amt = round(random.uniform(10000, 150000), 2)
        total_amt = round(base_amt * 1.18, 2)

        inv_row = {
            "gstin": vendor["gstin"],
            "invoice_no": inv_no,
            "vendor_name": vendor["name"],
            "amount": total_amt,
            "tax_rate": tax_rate,
            "date": date_str
        }
        invoices.append(inv_row)

        gt_entry = {
            "invoice_no": inv_no,
            "amount": total_amt,
            "category": scenario,
            "expected_decision": "ABSTAIN",
            "expected_action": "HUMAN_REVIEW",
            "expected_ledger_ids": [],
            "notes": ""
        }

        # 1. Similar GSTIN but different legal entity (e.g. 27SYNTC1234A1Z5 vs 27SYNTC1234A2Z5)
        if scenario == "SIMILAR_GSTIN_DIFFERENT_ENTITY":
            fake_gstin = vendor["gstin"][:-2] + "2" + vendor["gstin"][-1]
            ledger.append({
                "gstin": fake_gstin,
                "invoice_no": inv_no,
                "vendor_name": "NovaTech Subsidiary Unit",
                "amount": total_amt,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["expected_decision"] = "NO_MATCH"
            gt_entry["expected_action"] = "LEAVE_UNRESOLVED"
            gt_entry["notes"] = "Different GSTIN entity (different legal branch)."

        # 2. Split invoice with one missing fragment (e.g. ₹100,000 invoice, only ₹40,000 found)
        elif scenario == "SPLIT_WITH_MISSING_FRAGMENT":
            fragment_amt = round(total_amt * 0.40, 2)
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": f"{inv_no}-A",
                "vendor_name": vendor["name"],
                "amount": fragment_amt,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["expected_decision"] = "ABSTAIN"
            gt_entry["expected_action"] = "HUMAN_REVIEW"
            gt_entry["notes"] = f"Incomplete split: only Part A (₹{fragment_amt}) found; Part B missing."

        # 3. Partial payment (not standard gateway fee formula)
        elif scenario == "PARTIAL_PAYMENT_NON_GATEWAY":
            part_amt = round(total_amt * 0.70, 2)
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": part_amt,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["expected_decision"] = "ABSTAIN"
            gt_entry["expected_action"] = "HUMAN_REVIEW"
            gt_entry["notes"] = f"Arbitrary partial payment of ₹{part_amt} on ₹{total_amt} invoice."

        # 4. Amount within tolerance but representing completely different entity
        elif scenario == "TOLERANCE_COLLISION_DIFFERENT_ENTITY":
            other_vendor = [v for v in SAMPLE_VENDORS if v["gstin"] != vendor["gstin"]][0]
            ledger.append({
                "gstin": other_vendor["gstin"],
                "invoice_no": f"DIFF-PUR-{idx}",
                "vendor_name": other_vendor["name"],
                "amount": total_amt + 5.0,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["expected_decision"] = "NO_MATCH"
            gt_entry["expected_action"] = "LEAVE_UNRESOLVED"
            gt_entry["notes"] = "Unrelated transaction with different vendor."

        # 5. Conflicting evidence (matching invoice number but conflicting GSTIN & Amount)
        elif scenario == "CONFLICTING_EVIDENCE_ABSTAIN":
            ledger.append({
                "gstin": "29SYNCF9999Z1ZX",
                "invoice_no": inv_no,
                "vendor_name": "Conflicting Vendor Entity",
                "amount": total_amt * 1.5,
                "tax_rate": 12.0,
                "date": (date_obj + timedelta(days=60)).strftime("%Y-%m-%d")
            })
            gt_entry["expected_decision"] = "ABSTAIN"
            gt_entry["expected_action"] = "HUMAN_REVIEW"
            gt_entry["notes"] = "Conflicting vendor, GSTIN, amount, and date."

        # 6. Legitimate separate transactions that look like duplicates
        elif scenario == "LEGITIMATE_SEPARATE_DUPLICATE_LOOKING":
            ledger.extend([
                {"gstin": vendor["gstin"], "invoice_no": inv_no, "vendor_name": vendor["name"], "amount": total_amt, "tax_rate": tax_rate, "date": date_str},
                {"gstin": vendor["gstin"], "invoice_no": f"{inv_no}-BATCH2", "vendor_name": vendor["name"], "amount": total_amt, "tax_rate": tax_rate, "date": (date_obj + timedelta(days=25)).strftime("%Y-%m-%d")}
            ])
            gt_entry["expected_decision"] = "MATCH"
            gt_entry["expected_action"] = "AUTO_RESOLVE"
            gt_entry["expected_ledger_ids"] = [inv_no]
            gt_entry["notes"] = "Legitimate distinct recurring billing batch."

        # 7. Ambiguous multi-candidate collision
        elif scenario == "AMBIGUOUS_MULTI_CANDIDATE_ABSTAIN":
            ledger.extend([
                {"gstin": vendor["gstin"], "invoice_no": f"CHALLENGE-AMB-1-{idx}", "vendor_name": vendor["name"], "amount": total_amt, "tax_rate": tax_rate, "date": date_str},
                {"gstin": vendor["gstin"], "invoice_no": f"CHALLENGE-AMB-2-{idx}", "vendor_name": vendor["name"], "amount": total_amt, "tax_rate": tax_rate, "date": date_str}
            ])
            gt_entry["expected_decision"] = "ABSTAIN"
            gt_entry["expected_action"] = "HUMAN_REVIEW"
            gt_entry["expected_ledger_ids"] = [f"CHALLENGE-AMB-1-{idx}", f"CHALLENGE-AMB-2-{idx}"]
            gt_entry["notes"] = "Multiple ambiguous candidates with equal plausibility."

        # 8. Missing source record
        elif scenario == "MISSING_SOURCE_RECORD":
            gt_entry["expected_decision"] = "NO_MATCH"
            gt_entry["expected_action"] = "LEAVE_UNRESOLVED"
            gt_entry["notes"] = "Omitted from buyer ledger."

        # 9. Compounded date lag (40 days) + rounding variance
        elif scenario == "DATE_AND_ROUNDING_COMPOUNDED":
            lag_date = (date_obj + timedelta(days=40)).strftime("%Y-%m-%d")
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": total_amt + 12.0,
                "tax_rate": tax_rate,
                "date": lag_date
            })
            gt_entry["expected_decision"] = "MATCH"
            gt_entry["expected_action"] = "HUMAN_REVIEW"
            gt_entry["expected_ledger_ids"] = [inv_no]
            gt_entry["notes"] = "Compounded date variance (40 days) and rounding diff (₹12)."

        # 10. Clean exact match benchmark
        elif scenario == "EXACT_CLEAN_BENCHMARK":
            ledger.append({
                "gstin": vendor["gstin"],
                "invoice_no": inv_no,
                "vendor_name": vendor["name"],
                "amount": total_amt,
                "tax_rate": tax_rate,
                "date": date_str
            })
            gt_entry["expected_decision"] = "MATCH"
            gt_entry["expected_action"] = "AUTO_RESOLVE"
            gt_entry["expected_ledger_ids"] = [inv_no]
            gt_entry["notes"] = "Clean exact benchmark record."

        ground_truth[inv_no] = gt_entry

    challenge_invoices_df = pd.DataFrame(invoices)
    challenge_ledger_df = pd.DataFrame(ledger)
    challenge_invoices_df["amount"] = challenge_invoices_df["amount"].astype(float)
    challenge_invoices_df["tax_rate"] = challenge_invoices_df["tax_rate"].astype(float)
    challenge_ledger_df["amount"] = challenge_ledger_df["amount"].astype(float)
    challenge_ledger_df["tax_rate"] = challenge_ledger_df["tax_rate"].astype(float)

    meta = {
        "dataset_type": "Held-Out Adversarial Challenge Dataset",
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "ground_truth": ground_truth,
        "total_invoices": len(challenge_invoices_df),
        "total_ledger": len(challenge_ledger_df),
        "categories_breakdown": {sc: challenge_scenarios.count(sc) for sc in set(challenge_scenarios)}
    }
    return challenge_invoices_df, challenge_ledger_df, meta


def save_all_datasets(output_dir: str = "data"):
    """Generate and persist both Standard and Challenge datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Standard Dataset
    std_inv, std_led, std_meta = generate_synthetic_datasets(num_invoices=140, seed=42)
    std_inv.to_csv(os.path.join(output_dir, "invoices.csv"), index=False)
    std_led.to_csv(os.path.join(output_dir, "ledger.csv"), index=False)
    
    # 2. Held-Out Challenge Dataset
    ch_inv, ch_led, ch_meta = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
    ch_inv.to_csv(os.path.join(output_dir, "challenge_invoices.csv"), index=False)
    ch_led.to_csv(os.path.join(output_dir, "challenge_ledger.csv"), index=False)
    
    print(f"[SUCCESS] Datasets saved at '{output_dir}':")
    print(f"  - Standard Invoices: {len(std_inv)} rows | Ledger: {len(std_led)} rows")
    print(f"  - Challenge Invoices: {len(ch_inv)} rows | Ledger: {len(ch_led)} rows")


# Backward compatibility alias
save_synthetic_data = save_all_datasets


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    save_all_datasets(data_dir)
