"""
Concatenate multiple Parquet files in exports/parquet into a single Parquet file,
sorted by 'id'.
"""

import os
import polars as pl

# -------------------------------
# Configuration
# -------------------------------

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directory containing Parquet files
PARQUET_DIR = os.path.join(SCRIPT_DIR, "../../exports/parquet")

# Output Parquet file
OUTPUT_FILE = os.path.join(PARQUET_DIR, "full_data.parquet")

# -------------------------------
# Main script
# -------------------------------

def main():
    if not os.path.exists(PARQUET_DIR):
        print(f"Error: Parquet directory does not exist: {PARQUET_DIR}")
        return

    # Find all Parquet files in the directory
    parquet_files = [os.path.join(PARQUET_DIR, f) for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet")]

    if not parquet_files:
        print(f"No Parquet files found in {PARQUET_DIR}")
        return

    print(f"Found {len(parquet_files)} Parquet files. Loading...")

    # Load all Parquet files into a single DataFrame
    dfs = []
    for f in parquet_files:
        print(f"Loading {f}...")
        df = pl.read_parquet(f)
        dfs.append(df)

    full_df = pl.concat(dfs).sort("id")
    print(f"Concatenated DataFrame has {full_df.height} rows and {full_df.width} columns.")

    # Save to a single Parquet file
    full_df.write_parquet(OUTPUT_FILE)
    print(f"Saved concatenated DataFrame to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
