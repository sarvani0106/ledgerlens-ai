"""
data/build_datasets.py
Builds and saves the 4 CSV files in the data directory using generate_data.py.
"""
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from generate_data import generate_synthetic_datasets, generate_held_out_challenge_dataset

def build():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Standard Dataset
    std_inv, std_led, _ = generate_synthetic_datasets(num_invoices=140, seed=42)
    std_inv.to_csv(os.path.join(data_dir, "invoices.csv"), index=False)
    std_led.to_csv(os.path.join(data_dir, "ledger.csv"), index=False)
    
    # 2. Challenge Dataset
    ch_inv, ch_led, _ = generate_held_out_challenge_dataset(num_invoices=40, seed=101)
    ch_inv.to_csv(os.path.join(data_dir, "challenge_invoices.csv"), index=False)
    ch_led.to_csv(os.path.join(data_dir, "challenge_ledger.csv"), index=False)
    
    print("Standard Invoices saved:", len(std_inv))
    print("Standard Ledger saved:", len(std_led))
    print("Challenge Invoices saved:", len(ch_inv))
    print("Challenge Ledger saved:", len(ch_led))

if __name__ == "__main__":
    build()
