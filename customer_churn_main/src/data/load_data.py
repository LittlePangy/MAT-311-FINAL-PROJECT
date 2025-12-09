"""
The purpose of this module is to load in the data from the CSV files.
"""
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the credit card transaction dataset."""
    return pd.read_csv(file_path)


if __name__ == "__main__":
    df = load_dataset("data/raw/card_transdata.csv")
    print(df.head())
