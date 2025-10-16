from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

def evaluate_model(y_true, y_pred, y_proba=None):
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    if y_proba is not None:
        print("ROC-AUC Score:", roc_auc_score(y_true, y_proba))
    print("F1 Score:", f1_score(y_true, y_pred))

    