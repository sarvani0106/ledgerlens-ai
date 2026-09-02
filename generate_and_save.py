"""
generate_and_save.py
Executes save_all_datasets from generate_data.py to generate and persist the 4 synthetic CSV files.
"""
import os
from generate_data import save_all_datasets

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    save_all_datasets(data_dir)
