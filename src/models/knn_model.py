"""
Module for training a k-nearest neighbors model
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series, num_neighbors: int = 25) -> KNeighborsClassifier:
    """Train and return a 3-NN classifier."""
    model = KNeighborsClassifier(n_neighbors=num_neighbors)
    model.fit(X_train, y_train)
    return model
