import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    idx = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred): cm[idx[t], idx[p]] += 1
    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    tn, fp, fn, tp = cm.ravel()
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    tn, fp, fn, tp = cm.ravel()
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2): return 0.0
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * p * r / (p + r)) if (p + r) else 0.0
