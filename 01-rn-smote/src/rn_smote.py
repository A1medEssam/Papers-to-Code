"""Implements the paper's Algorithm 3 / Fig. 5 pipeline:
  1. SMOTE-oversample the training data to full balance.
  2. Extract the (now-balanced) minority pool -- original + synthetic points.
  3. Run DBSCAN on that pool. eps is found via the k-distance knee method
     (Algorithm 3, FindEps); MinPts = K = ln(N) (Eq. 2). The paper reuses
     the SAME K as both the neighbor count for the k-distance graph and
     MinPts -- both are wired to the same value here.
  4. Drop DBSCAN's noise cluster; combine the surviving minority points
     with the original (imbalanced) training data.
  5. SMOTE again to rebalance.

This fixes two issues found in the original notebook (NTDS_Project_01.ipynb):
  - the minority class was extracted by a hardcoded label (`y_smote == 1`)
    rather than by whichever class SMOTE actually oversampled, which after
    this dataset's label encoding silently pointed at the majority class;
  - the k-distance graph was built with a fixed k=5 instead of the same
    K = ln(N) the paper uses for MinPts.
"""
from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def find_eps(X, k):
    """Paper Algorithm 3 (FindEps): k-distance graph + knee point."""
    if len(X) <= k:
        nn = NearestNeighbors(n_neighbors=len(X)).fit(X)
        dists, _ = nn.kneighbors(X)
        fallback = float(np.median(dists[:, -1]))
        return fallback if fallback > 0 else 1e-6

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = nn.kneighbors(X)
    k_dist = np.sort(dists[:, -1])

    knee = KneeLocator(
        range(len(k_dist)), k_dist, curve="convex", direction="increasing"
    )
    if knee.knee is None:
        return float(np.median(k_dist))
    return float(k_dist[knee.knee])


def _safe_k_neighbors(y, minority_label, requested=5):
    """SMOTE requires k_neighbors < (minority count in this training fold).
    A few datasets here have single-digit minority counts before
    oversampling (Stamps: 6, Lymphography: 6, WBC: 10), so a k-fold split
    can leave a fold's training minority count below the library default
    of 5. Scale k_neighbors down rather than let the fold crash."""
    n_minor = int(np.sum(y == minority_label))
    return max(1, min(requested, n_minor - 1))


def rn_smote(X_train, y_train, random_state=42):
    """Returns (X_final, y_final): the RN-SMOTE-resampled training set."""
    counts = Counter(y_train)
    minority_label = min(counts, key=counts.get)

    k_neighbors = _safe_k_neighbors(y_train, minority_label)
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    X_minor = X_smote[y_smote == minority_label]
    n = len(X_minor)
    K = max(2, int(np.log(n))) if n > 2 else 2  # paper Eq. 2: K = ln(N)

    eps = find_eps(X_minor, k=K)
    labels = DBSCAN(eps=eps, min_samples=K).fit_predict(X_minor)
    clean_minor = X_minor[labels != -1]

    if len(clean_minor) == 0:
        # DBSCAN flagged the entire minority pool as noise (can happen on
        # very small or very uniform minority classes). Fall back to the
        # plain SMOTE result rather than collapsing the minority class.
        return X_smote, y_smote

    X_combined = np.vstack([X_train, clean_minor])
    y_combined = np.hstack([y_train, np.full(len(clean_minor), minority_label)])

    return smote.fit_resample(X_combined, y_combined)
