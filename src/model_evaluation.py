# src/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

def evaluate_model(y_test, y_pred, y_proba):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    return metrics

def save_metrics(metrics, path="reports/metrics.json"):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
