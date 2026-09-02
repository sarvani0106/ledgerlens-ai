import os
import pandas as pd
from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# 1. Standard Dataset (140 rows, seed=42)
inv_df, led_df, meta = generate_synthetic_datasets(num_invoices=140, seed=42)
inv_df.to_csv(os.path.join(data_dir, "invoices.csv"), index=False)
led_df.to_csv(os.path.join(data_dir, "ledger.csv"), index=False)

# 2. Held-Out Challenge Dataset (40 rows, seed=101)
ch_inv_df, ch_led_df, ch_meta = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
ch_inv_df.to_csv(os.path.join(data_dir, "challenge_invoices.csv"), index=False)
ch_led_df.to_csv(os.path.join(data_dir, "challenge_ledger.csv"), index=False)

print("Standard Invoices:", len(inv_df))
print("Standard Ledger:", len(led_df))
print("Challenge Invoices:", len(ch_inv_df))
print("Challenge Ledger:", len(ch_led_df))
