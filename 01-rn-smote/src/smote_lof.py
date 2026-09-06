"""SMOTE first, then Local Outlier Factor flags noisy points within the
now-balanced minority class, and those are dropped. Unlike RN-SMOTE, the
paper's description of SMOTE-LOF does not include a second rebalancing
SMOTE pass -- the noise-flagged minority points are simply removed.
"""
from collections import Counter

from sklearn.neighbors import LocalOutlierFactor

from rn_smote import _safe_k_neighbors
from imblearn.over_sampling import SMOTE
import numpy as np


def smote_lof(X_train, y_train, random_state=42, n_neighbors=20):
    counts = Counter(y_train)
    minority_label = min(counts, key=counts.get)

    k = _safe_k_neighbors(y_train, minority_label)
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    minor_mask = y_smote == minority_label
    X_minor = X_smote[minor_mask]

    lof_neighbors = min(n_neighbors, len(X_minor) - 1)
    if lof_neighbors < 2:
        return X_smote, y_smote  # too few points for LOF to be meaningful

    lof = LocalOutlierFactor(n_neighbors=lof_neighbors)
    is_inlier = lof.fit_predict(X_minor) == 1  # -1 = outlier, 1 = inlier

    X_majority = X_smote[~minor_mask]
    y_majority = y_smote[~minor_mask]
    X_final = np.vstack([X_majority, X_minor[is_inlier]])
    y_final = np.hstack([y_majority, np.full(is_inlier.sum(), minority_label)])
    return X_final, y_final
