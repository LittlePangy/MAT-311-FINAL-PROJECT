"""Utility helpers for model training and evaluation."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def split_data(df: pd.DataFrame):
    """Split dataframe into train/validation/test sets."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=123, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_roc_curve(y_true, y_score, label: str) -> float:
    """Plot a ROC curve and return the AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label} (AUC={auc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return auc
