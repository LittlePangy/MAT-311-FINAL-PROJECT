"""
This module is the main execution script for training and evaluating models 
for predicting customer churn and evaluating their performance.
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.load_data import load_dataset

from src.data.preprocess import clean_dataset
# from src.visualization.eda import plot_eda
from src.models.train_model import split_data, plot_roc_curve
from src.models.knn_model import train_knn_model
from src.models.dumb_model import train_dumb_model
from src.models.rand_forest_model import train_rand_forest_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)


def main() -> None:
    print("---Loading data...")
    raw_df = load_dataset("data/raw/train.csv")
    
    # Print shape of the raw dataset
    print(f"Raw dataset shape: {raw_df.shape}")

    print("---Cleaning data...")
    clean_df = clean_dataset(raw_df)

    print(f"Cleaned dataset shape: {clean_df.shape}")

    print("---Creating EDA visuals...")
    # plot_eda(clean_df)

    print("---Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean_df.drop("CustomerID", axis=1))

    print("---Training models...")
    dumb_model = train_dumb_model(X_train, y_train)
    knn_model = train_knn_model(X_train, y_train)
    rand_forest_model = train_rand_forest_model(X_train, y_train)

    print("---Evaluating on validation set...")
    y_val_pred_dumb = dumb_model.predict(X_val)
    y_val_pred_knn = knn_model.predict(X_val)
    y_val_pred_rand_forest = rand_forest_model.predict(X_val)

    val_prob_knn = knn_model.predict_proba(X_val)[:, 1]
    val_prob_dumb = dumb_model.predict_proba(X_val)[:, 1]
    val_prob_rand_forest = rand_forest_model.predict_proba(X_val)[:, 1]

    prediction_csv = pd.DataFrame({"CustomerID": clean_df.iloc[X_val.index]["CustomerID"], "Churn": val_prob_knn})
    prediction_csv.to_csv("data/pls_wrk_prediction.csv", index=False)

    # Output accuracy, precision, recall, specificity, and F1 scores for each of the 3 models on the validation set

    
    plot_confusion_matrices(y_val, y_val_pred_dumb, y_val_pred_knn, y_val_pred_rand_forest)
    plot_performance_comparison(y_val, y_val_pred_dumb, y_val_pred_knn, y_val_pred_rand_forest)

    def plot_roc_curves(y_true, y_prob, label):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc_score:.3f})")
        return auc_score
    
    plt.figure()

    plot_roc_curves(y_val, val_prob_dumb, "Never Churn")
    plot_roc_curves(y_val, val_prob_knn, "3-NN")
    plot_roc_curves(y_val, val_prob_rand_forest, "Random Forest")

    plt.plot([0, 1], [0, 1], "k--")  # baseline

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()

    # Plot all ROC curves on the same graph
    # auc_dumb = plot_roc_curve(y_val, val_prob_dumb, "Never Fraud")
    # auc_knn = plot_roc_curve(y_val, val_prob_knn, "3-NN")
    # auc_rand_forest = plot_roc_curve(y_val, val_prob_rand_forest, "Random Forest")

    # Determine best model based on AUC
    auc_dumb = auc(*roc_curve(y_val, val_prob_dumb)[:2])
    auc_knn = auc(*roc_curve(y_val, val_prob_knn)[:2])
    auc_rand_forest = auc(*roc_curve(y_val, val_prob_rand_forest)[:2])  
    best_model = knn_model if auc_knn >= auc_dumb else (dumb_model if auc_dumb >= auc_rand_forest else rand_forest_model)
    best_label = "3-NN" if best_model is knn_model else ("Never Fraud" if best_model is dumb_model else "Random Forest")

    print(f"---Testing best model ({best_label})...")
    y_test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, test_prob, f"Test {best_label}")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Best Model Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
