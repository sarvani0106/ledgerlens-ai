"""
Diagnostic script to identify false positive cases in the Held-Out Challenge Dataset.
"""
from generate_data import generate_held_out_challenge_dataset
from reconciliation_engine import run_full_reconciliation

ch_inv, ch_led, ch_meta = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
results = run_full_reconciliation(
    invoices_df=ch_inv,
    ledger_df=ch_led,
    ground_truth_meta=ch_meta,
    amount_tolerance_abs=15.0,
    amount_tolerance_pct=0.01,
    anthropic_api_key=None,
    enable_offline_fallback=True
)

metrics = results['metrics']
gt_map = ch_meta.get('ground_truth', {})
auto_df = results['all_auto_resolved_df']
review_df = results['human_review_df']
unresolved_df = results['unresolved_invoices_df']

print("=== OVERALL METRICS ===")
print(f"Precision:    {metrics['precision']}%")
print(f"False Positives: {metrics['false_positives']}")
print(f"Auto-resolved:   {metrics['auto_resolved_count']}")
print(f"Human Review:    {metrics['human_review_count']}")
print(f"Unresolved:      {metrics['unresolved_count']}")
print()

print("=== FALSE POSITIVE CASES (Auto-resolved where GT != MATCH) ===")
fp_count = 0
if not auto_df.empty:
    for _, row in auto_df.iterrows():
        inv_no = row['invoice_no']
        gt = gt_map.get(inv_no, {})
        expected = gt.get('expected_decision', '?')
        if expected != 'MATCH':
            fp_count += 1
            print(f"  [{fp_count}] Invoice: {inv_no}")
            print(f"       GT Expected:   {expected}")
            print(f"       Stage:         {row.get('match_stage')}")
            print(f"       Match Type:    {row.get('match_type')}")
            print(f"       Confidence:    {row.get('confidence')}")
            print(f"       GT Category:   {gt.get('category')}")
            print(f"       GT Notes:      {str(gt.get('notes')).replace(chr(8377), 'Rs')}")
            print()

print(f"Total FPs: {fp_count}")
print()
print("=== ALL AUTO-RESOLVED ===")
if not auto_df.empty:
    for _, row in auto_df.iterrows():
        inv_no = row['invoice_no']
        gt = gt_map.get(inv_no, {})
        expected = gt.get('expected_decision', '?')
        correct = "OK" if expected == 'MATCH' else "FP!"
        print(f"  [{correct}] {inv_no} | GT={expected} | Stage={row.get('match_stage')} | Conf={row.get('confidence')}")
