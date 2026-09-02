import os
import pandas as pd
from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset

os.makedirs("data", exist_ok=True)

# 1. Standard Dataset
std_inv, std_led, _ = generate_synthetic_datasets(num_invoices=140, seed=42)
std_inv.to_csv(os.path.join("data", "invoices.csv"), index=False)
std_led.to_csv(os.path.join("data", "ledger.csv"), index=False)

# 2. Challenge Dataset
ch_inv, ch_led, _ = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
ch_inv.to_csv(os.path.join("data", "challenge_invoices.csv"), index=False)
ch_led.to_csv(os.path.join("data", "challenge_ledger.csv"), index=False)

print("Export completed successfully.")
