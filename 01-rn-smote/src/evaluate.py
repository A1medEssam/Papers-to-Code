import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from collections import Counter

from metrics import compute_metrics
from rn_smote import _safe_k_neighbors, rn_smote

warnings.filterwarnings("ignore")

CONDITIONS = ["Original", "SMOTE", "RN-SMOTE"]


def make_classifiers(random_state=42):
    return {
        "CART": DecisionTreeClassifier(random_state=random_state),
        "GB": GradientBoostingClassifier(random_state=random_state),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=random_state, verbosity=0),
        "SGD-LR": SGDClassifier(loss="log_loss", random_state=random_state),
        "SVM": SVC(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
    }


def resample(condition, X_train, y_train, random_state):
    if condition == "Original":
        return X_train, y_train
    if condition == "SMOTE":
        minority_label = min(Counter(y_train), key=Counter(y_train).get)
        k = _safe_k_neighbors(y_train, minority_label)
        return SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X_train, y_train)
    if condition == "RN-SMOTE":
        return rn_smote(X_train, y_train, random_state=random_state)
    raise ValueError(condition)


def evaluate_dataset(name, X, y, n_repeats=10, n_splits=10, random_state=42,
                      classifiers=None):
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    clf_bank = make_classifiers(random_state)
    if classifiers is not None:
        clf_bank = {k: v for k, v in clf_bank.items() if k in classifiers}

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for condition in CONDITIONS:
            try:
                X_res, y_res = resample(condition, X_train, y_train, random_state)
            except Exception as e:
                warnings.warn(f"{name}/{condition} resample failed on fold {fold_i}: {e}")
                continue

            for clf_name, clf in clf_bank.items():
                model = clone(clf)
                try:
                    model.fit(X_res, y_res)
                    y_pred = model.predict(X_test)
                except Exception as e:
                    warnings.warn(f"{name}/{condition}/{clf_name} failed on fold {fold_i}: {e}")
                    continue
                m = compute_metrics(y_test, y_pred)
                m.update(dataset=name, condition=condition, classifier=clf_name, fold=fold_i)
                rows.append(m)

    return pd.DataFrame(rows)


def summarize(df):
    metric_cols = ["GM", "Kappa", "MCC", "F1", "Precision", "Recall"]
    return (
        df.groupby(["dataset", "condition", "classifier"])[metric_cols]
        .mean()
        .reset_index()
    )
