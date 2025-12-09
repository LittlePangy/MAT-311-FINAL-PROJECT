"""
Module for training a Random Forest model
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_rand_forest_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train and return a Random Forest classifier."""
    model = RandomForestClassifier(max_depth=13, min_samples_leaf=3, min_samples_split=9, n_estimators=472, n_jobs=-1, random_state=42) # BEST ONE SO FAR
    model.fit(X_train, y_train)
    return model
