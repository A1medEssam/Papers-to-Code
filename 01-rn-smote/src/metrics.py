#Metrics matching RN-SMOTE paper Table 2, computed for the minority (1) class.

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
import numpy as np


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # minority recall
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # majority recall
    gm = np.sqrt(sensitivity * specificity)
    return dict(
        GM=gm,
        Kappa=cohen_kappa_score(y_true, y_pred),
        MCC=matthews_corrcoef(y_true, y_pred),
        F1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        Precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        Recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        Accuracy=accuracy_score(y_true, y_pred),
    )
