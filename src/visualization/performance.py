"""
This module contains functions for evaluating the performance of different models.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrices(y_test, y_pred_baseline, y_pred_knn, y_pred_rand_forest) -> None:
    """Plot confusion matrices for both models."""
    conf_baseline = confusion_matrix(y_test, y_pred_baseline)
    conf_knn = confusion_matrix(y_test, y_pred_knn)
    conf_rand_forest = confusion_matrix(y_test, y_pred_rand_forest)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.heatmap(conf_baseline, annot=True, fmt='d', cmap='Reds', ax=axes[0])
    axes[0].set_title('Never Fraud')
    sns.heatmap(conf_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('3-NN')
    sns.heatmap(conf_rand_forest, annot=True, fmt='d', cmap='Greens', ax=axes[2])
    axes[2].set_title('Random Forest')
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(y_test, y_pred_baseline, y_pred_knn, y_pred_rand_forest) -> None:
    """Create a bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity','F1 Score']
    baseline_scores = [
        accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline, zero_division=0),
        recall_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_baseline, pos_label=0),
        f1_score(y_test, y_pred_baseline)
    ]
    knn_scores = [
        accuracy_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_knn, pos_label=0),
        f1_score(y_test, y_pred_knn)
    ]
    rand_forest_scores = [
        accuracy_score(y_test, y_pred_rand_forest),
        precision_score(y_test, y_pred_rand_forest),
        recall_score(y_test, y_pred_rand_forest),
        recall_score(y_test, y_pred_rand_forest, pos_label=0), 
        f1_score(y_test, y_pred_rand_forest)
    ]
    df = pd.DataFrame({'Metric': metrics, 'k-NN': knn_scores, 'RF': rand_forest_scores,'Never Fraud': baseline_scores})
    df.plot(x='Metric', kind='bar', figsize=(8, 5))
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.show()
