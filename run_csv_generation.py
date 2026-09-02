"""
Generate and overwrite data/ CSV files with 100% synthetic fictional data.
"""
import os
import pandas as pd
from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset

def run():
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Standard Dataset
    std_inv, std_led, _ = generate_synthetic_datasets(num_invoices=140, seed=42)
    std_inv_path = os.path.join(target_dir, "invoices.csv")
    std_led_path = os.path.join(target_dir, "ledger.csv")
    std_inv.to_csv(std_inv_path, index=False)
    std_led.to_csv(std_led_path, index=False)
    
    # 2. Challenge Dataset
    ch_inv, ch_led, _ = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
    ch_inv_path = os.path.join(target_dir, "challenge_invoices.csv")
    ch_led_path = os.path.join(target_dir, "challenge_ledger.csv")
    ch_inv.to_csv(ch_inv_path, index=False)
    ch_led.to_csv(ch_led_path, index=False)
    
    print(f"Generated {std_inv_path} ({len(std_inv)} records)")
    print(f"Generated {std_led_path} ({len(std_led)} records)")
    print(f"Generated {ch_inv_path} ({len(ch_inv)} records)")
    print(f"Generated {ch_led_path} ({len(ch_led)} records)")

if __name__ == "__main__":
    run()
