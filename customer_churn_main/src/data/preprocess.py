"""
This module provides functionality for cleaning
and preprocessing a dataset for analysis.
"""

import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def one_hot_encode_column(
    df: pd.DataFrame, column_name: str, column_labels: dict
) -> pd.DataFrame:
    """One-hot encode the specified column in the DataFrame."""
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder_df = pd.DataFrame(encoder.fit_transform(df[[column_name]]).toarray())
    df = df.join(encoder_df)
    df.rename(columns=column_labels, inplace=True)
    df.drop(column_name, axis=1, inplace=True)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset through imputation and encoding."""
    cleanup_df = df.copy()

    # Impute missing values for Tenure and Last Interaction with mean
    cleanup_df["Tenure"] = cleanup_df["Tenure"].fillna(cleanup_df["Tenure"].mean())
    cleanup_df["Last Interaction"] = cleanup_df["Last Interaction"].fillna(
        cleanup_df["Last Interaction"].mean()
    )

    # Perform one-hot encoding on Gender, Subscription Type, Contract Length, and Customer Status columns
    cleanup_df = one_hot_encode_column(
        cleanup_df, "Subscription Type", {0: "Basic", 1: "Premium", 2: "Standard"}
    )
    cleanup_df = one_hot_encode_column(
        cleanup_df, "Contract Length", {0: "Monthly", 1: "Quarterly", 2: "Yearly"}
    )
    cleanup_df = one_hot_encode_column(
        cleanup_df,
        "Gender",
        {0: "Female", 1: "Male"},
    )

    # Replace 'none' and nan with 0 in Support Calls column
    cleanup_df["Support Calls"] = cleanup_df["Support Calls"].replace("none", 0)
    cleanup_df["Support Calls"] = cleanup_df["Support Calls"].fillna(0)
    cleanup_df["Support Calls"] = cleanup_df["Support Calls"].astype(int)

    # Cleaning Payment Delay, Last Interaction, and Tenure Columns
    cleanup_df["Last Interaction"] = cleanup_df["Last Interaction"].fillna(
        cleanup_df["Last Interaction"].mean()
    )
    cleanup_df["Tenure"] = cleanup_df["Tenure"].fillna(cleanup_df["Tenure"].mean())

    # Convert dates to datetime (use any placeholder year)
    due = pd.to_datetime(cleanup_df["Last Due Date"], format="%m-%d")
    paid = pd.to_datetime(cleanup_df["Last Payment Date"], format="%m-%d")

    # Compute delay in days
    computed_delay = (paid - due).dt.days

    # Only fill missing values in Payment Delay
    cleanup_df.loc[cleanup_df["Payment Delay"].isna(), "Payment Delay"] = (
        computed_delay[cleanup_df["Payment Delay"].isna()]
    )

    # Drop unused columns
    cleanup_df = cleanup_df.drop(["Customer Status"], axis=1, errors="ignore")
    cleanup_df = cleanup_df.drop(["Last Due Date", "Last Payment Date"], axis=1)

    # # Scale numeric columns
    scaler = StandardScaler()
    numeric_columns = [
        "Tenure",
        "Last Interaction",
        "Payment Delay",
        "Age",
        "Total Spend",
        "Usage Frequency",
    ]
    cleanup_df[numeric_columns] = scaler.fit_transform(cleanup_df[numeric_columns])

    return cleanup_df


if __name__ == "__main__":
    # Load the raw dataset
    raw = pd.read_csv("data/raw/train.csv")
    # Clean the dataset
    cleaned = clean_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "data/processed/train_clean.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
