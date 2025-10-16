import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plots a confusion matrix heatmap.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - model_name: Name of the model (for title)
    - save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name="Model", save_path=None):
    """
    Plots the ROC Curve.

    Parameters:
    - y_true: True labels
    - y_proba: Predicted probabilities (for positive class)
    - model_name: Name of the model (for title)
    - save_path: Path to save the plot (optional)
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.show()
