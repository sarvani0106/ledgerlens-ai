"""
Direct generator to save all datasets to data/ folder.
"""
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("Generating Standard Benchmark Dataset (seed=42)...")
    std_inv, std_led, _ = generate_synthetic_datasets(num_invoices=140, seed=42)
    std_inv.to_csv(os.path.join(data_dir, "invoices.csv"), index=False)
    std_led.to_csv(os.path.join(data_dir, "ledger.csv"), index=False)

    print("Generating Held-Out Challenge Dataset (seed=101)...")
    ch_inv, ch_led, _ = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
    ch_inv.to_csv(os.path.join(data_dir, "challenge_invoices.csv"), index=False)
    ch_led.to_csv(os.path.join(data_dir, "challenge_ledger.csv"), index=False)

    print("Done! All 4 CSVs updated with synthetic data.")

if __name__ == "__main__":
    main()
