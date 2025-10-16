from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    results = {}
    results['Model'] = model_name
    results['F1 Score'] = f1_score(y_true, y_pred)
    
    if y_proba is not None:
        results['ROC-AUC'] = roc_auc_score(y_true, y_proba)
    else:
        results['ROC-AUC'] = None
    
    report = classification_report(y_true, y_pred, output_dict=True)
    results['Precision'] = report['weighted avg']['precision']
    results['Recall'] = report['weighted avg']['recall']
    results['Accuracy'] = report['accuracy']
    results['Confusion Matrix'] = confusion_matrix(y_true, y_pred).tolist()  # Convert to list for easier serialization
    return results