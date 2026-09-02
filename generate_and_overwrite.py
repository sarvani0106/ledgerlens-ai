import os
import pandas as pd
from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(data_dir, exist_ok=True)

# 1. Standard Dataset
std_inv, std_led, _ = generate_synthetic_datasets(num_invoices=140, seed=42)
std_inv.to_csv(os.path.join(data_dir, "invoices.csv"), index=False)
std_led.to_csv(os.path.join(data_dir, "ledger.csv"), index=False)

# 2. Held-Out Challenge Dataset
ch_inv, ch_led, _ = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
ch_inv.to_csv(os.path.join(data_dir, "challenge_invoices.csv"), index=False)
ch_led.to_csv(os.path.join(data_dir, "challenge_ledger.csv"), index=False)

print("SUCCESS: 4 synthetic CSV files written to data/ folder.")
