"""
This module provides functionality for splitting a 
dataset into a training, testing, and validation set.
"""


import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train, validation and test sets."""
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-8:
        raise ValueError("Fractions must sum to 1.0")

    train_df, temp_df = train_test_split(
        df, train_size=train_frac, random_state=seed, stratify=df["Churn"]
    )
    val_size = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_size, random_state=seed, stratify=temp_df["Churn"]
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    pass
