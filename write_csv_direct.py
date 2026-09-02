import os
import pandas as pd
from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Standard Dataset
    inv_df, led_df, _ = generate_synthetic_datasets(num_invoices=140, seed=42)
    inv_path = os.path.join(data_dir, "invoices.csv")
    led_path = os.path.join(data_dir, "ledger.csv")
    inv_df.to_csv(inv_path, index=False)
    led_df.to_csv(led_path, index=False)

    # 2. Challenge Dataset
    ch_inv_df, ch_led_df, _ = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
    ch_inv_path = os.path.join(data_dir, "challenge_invoices.csv")
    ch_led_path = os.path.join(data_dir, "challenge_ledger.csv")
    ch_inv_df.to_csv(ch_inv_path, index=False)
    ch_led_df.to_csv(ch_led_path, index=False)

    print("Successfully generated all 4 CSVs:")
    print(f"  {inv_path} ({len(inv_df)} rows)")
    print(f"  {led_path} ({len(led_df)} rows)")
    print(f"  {ch_inv_path} ({len(ch_inv_df)} rows)")
    print(f"  {ch_led_path} ({len(ch_led_df)} rows)")

if __name__ == "__main__":
    main()
