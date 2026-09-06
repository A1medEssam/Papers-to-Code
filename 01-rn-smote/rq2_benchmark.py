"""Reproduces the paper's section 4.6 comparison: RN-SMOTE vs. SMOTE-LOF
on PIMA, Haberman, and Glass, using NB and SVM (the paper's own choice of
classifiers for this specific comparison), scored on Accuracy, Precision, amd
Recall, and F1.
"""
import sys
import warnings

sys.path.insert(0, "/home/a1medessam/Desktop/papers2code/rnsmote_project/src")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import datasets
from metrics import compute_metrics
from rn_smote import rn_smote, _safe_k_neighbors
from smote_lof import smote_lof
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings("ignore")

CLASSIFIERS = {"NB": GaussianNB(), "SVM": SVC(random_state=42)}
CONDITIONS = ["SMOTE-LOF", "RN-SMOTE"]
DATASETS = ["pima", "haberman", "glass"]


def resample(condition, X_train, y_train, random_state):
    if condition == "SMOTE-LOF":
        return smote_lof(X_train, y_train, random_state=random_state)
    if condition == "RN-SMOTE":
        return rn_smote(X_train, y_train, random_state=random_state)
    raise ValueError(condition)


def run(name, n_repeats=10, n_splits=5, random_state=42):
    X, y = datasets.load(name)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=random_state)
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for condition in CONDITIONS:
            try:
                X_res, y_res = resample(condition, X_train, y_train, random_state)
            except Exception as e:
                warnings.warn(f"{name}/{condition} fold {fold_i}: {e}")
                continue
            for clf_name, clf in CLASSIFIERS.items():
                model = clone(clf)
                model.fit(X_res, y_res)
                y_pred = model.predict(X_test)
                m = compute_metrics(y_test, y_pred)
                m.update(dataset=name, condition=condition, classifier=clf_name)
                rows.append(m)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_rows = pd.concat([run(name) for name in DATASETS], ignore_index=True)
    all_rows.to_csv("/home/a1medessam/Desktop/papers2code/rnsmote_project/results_smote_lof_raw.csv", index=False)
    summary = (all_rows.groupby(["dataset", "condition", "classifier"])
               [["Accuracy", "Precision", "Recall", "F1"]].mean().reset_index())
    summary.to_csv("/home/a1medessam/Desktop/papers2code/rnsmote_project/results_smote_lof_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
