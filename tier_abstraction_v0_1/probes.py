"""Linear probes, within-superclass confusion concentration (the primary abstraction
signature), and linear CKA.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import FINE_TO_COARSE


def linear_probe(X_train, y_train, X_test, y_test, max_iter=2000, C=1.0):
    """Standardize -> multinomial logistic regression. Returns (test_acc, test_predictions)."""
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    acc = float((pred == y_test).mean())
    return acc, pred


def within_super_confusion(fine_pred, fine_true):
    """Among misclassified fine examples, fraction whose PREDICTED fine class shares the
    superclass of the TRUE fine class. High value => errors concentrate within superclass
    (the signature of tier abstraction). Returns (within_super_frac, n_errors)."""
    fine_pred = np.asarray(fine_pred)
    fine_true = np.asarray(fine_true)
    wrong = fine_pred != fine_true
    n_err = int(wrong.sum())
    if n_err == 0:
        return float("nan"), 0
    pred_coarse = FINE_TO_COARSE[fine_pred[wrong]]
    true_coarse = FINE_TO_COARSE[fine_true[wrong]]
    within = float((pred_coarse == true_coarse).mean())
    return within, n_err


def linear_cka(X, Y):
    """Linear CKA between two representations (rows = samples). Returns scalar in [0,1]."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")
    denom = norm_x * norm_y
    return float(hsic / denom) if denom > 0 else float("nan")


def chance_within_super():
    """Expected within-superclass error fraction for a uniformly-random wrong prediction:
    for a true class in superclass s (size m_s), among the other 99 fine classes,
    (m_s - 1) share its superclass. Averaged over classes => a baseline for #5."""
    counts = np.bincount(FINE_TO_COARSE, minlength=20)
    fracs = []
    for fine in range(100):
        s = FINE_TO_COARSE[fine]
        same = counts[s] - 1           # other fine classes in same superclass
        fracs.append(same / 99.0)      # 99 possible wrong labels
    return float(np.mean(fracs))


if __name__ == "__main__":
    print("chance within-super error frac:", round(chance_within_super(), 4))
