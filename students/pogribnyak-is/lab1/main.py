from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from sklearn.metrics import accuracy_score as sklearn_accuracy
from sklearn.tree import DecisionTreeClassifier as SklearnTree

from data.datasets import DATASETS
from metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from model import DecisionTree, ReducedErrorPruner

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {"name": "breast_cancer", "test_size": 0.2, "val_size": 0.25, "seed": 42},
    "model": {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_impurity_decrease": 0.0,
    },
}


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = {**DEFAULT_CONFIG}
    if path is None:
        if Path("config.yaml").exists(): path = "config.yaml"
    if not path: return cfg
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        with open(path, encoding="utf-8") as f: file_cfg = yaml.safe_load(f) or {}
    else: raise ValueError(f"Неизвестный формат: {path}")
    for k, v in file_cfg.items():
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict): cfg[k] = {**cfg[k], **v}
        else: cfg[k] = v
    return cfg


def to_binary(y: np.ndarray) -> np.ndarray:
    u = np.unique(y)
    if len(u) != 2: return y
    m = {a: i for i, a in enumerate(sorted(u))}
    return np.array([m[v] for v in y])


def print_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


def main() -> None:
    cfg = load_config()
    ds = cfg["dataset"]
    tr = cfg["model"]

    name = ds["name"]
    if name not in DATASETS:
        print(f"Нет датасета '{name}'. Доступны: {list(DATASETS)}")
        return

    dataset = DATASETS[name](seed=ds["seed"])
    n_nan_raw = int(np.isnan(dataset.X).sum())
    print(f"\n=== Датасет {name}: {len(dataset.X)} x {dataset.X.shape[1]} ===")
    print(f"Классы: {np.bincount(to_binary(dataset.y))}, NaN в X (после preprocess): {n_nan_raw}")

    X_tr, y_tr, X_va, y_va, X_te, y_te = dataset.split_and_scale(ds["test_size"], ds["val_size"])
    print(
        f"NaN после масштабирования: train={np.isnan(X_tr).sum()}, "
        f"val={np.isnan(X_va).sum()}, test={np.isnan(X_te).sum()}"
    )
    y_tr, y_va, y_te = to_binary(y_tr), to_binary(y_va), to_binary(y_te)

    tree = DecisionTree(
        max_depth=tr["max_depth"],
        min_samples_split=tr["min_samples_split"],
        min_samples_leaf=tr["min_samples_leaf"],
        min_impurity_decrease=tr["min_impurity_decrease"],
    )
    tree.fit(X_tr, y_tr)
    print(f"\n=== Своё дерево: глубина {tree.get_depth()}, листьев {tree.get_n_leaves()} ===")

    pred_tr = tree.predict(X_tr)
    pred_te = tree.predict(X_te)
    print_metrics("Train", y_tr, pred_tr)
    print_metrics("Test", y_te, pred_te)

    reduced = ReducedErrorPruner(X_va, y_va).prune(tree)
    print(f"\n=== После прунинга: глубина {reduced.get_depth()}, листьев {reduced.get_n_leaves()} ===")
    print_metrics("Train", y_tr, reduced.predict(X_tr))
    print_metrics("Test", y_te, reduced.predict(X_te))


    sk_params = dict(
        max_depth=tr["max_depth"],
        min_samples_split=tr["min_samples_split"],
        min_samples_leaf=tr["min_samples_leaf"],
        random_state=ds["seed"],
    )
    sk = SklearnTree(**sk_params)
    sk.fit(X_tr, y_tr)
    print(f"\n=== Sklearn (без прунинга): глубина {sk.get_depth()}, листьев {sk.get_n_leaves()} ===")
    print_metrics("Sklearn train", y_tr, sk.predict(X_tr))
    print_metrics("Sklearn test", y_te, sk.predict(X_te))

    path = sk.cost_complexity_pruning_path(X_tr, y_tr)
    best_alpha, best_va = 0.0, -1.0
    for a in path.ccp_alphas:
        clf = SklearnTree(**sk_params, ccp_alpha=float(a))
        clf.fit(X_tr, y_tr)
        va_acc = sklearn_accuracy(y_va, clf.predict(X_va))
        if va_acc > best_va:
            best_va, best_alpha = va_acc, float(a)

    sk_p = SklearnTree(**sk_params, ccp_alpha=best_alpha)
    sk_p.fit(X_tr, y_tr)
    print(
        f"\n=== Sklearn + CCP (ccp_alpha={best_alpha:.6g}, val acc={best_va:.4f}): "
        f"глубина {sk_p.get_depth()}, листьев {sk_p.get_n_leaves()} ==="
    )
    print_metrics("Sklearn+CCP train", y_tr, sk_p.predict(X_tr))
    print_metrics("Sklearn+CCP test", y_te, sk_p.predict(X_te))

    pred_te_r = reduced.predict(X_te)
    pred_te_sk = sk.predict(X_te)
    pred_te_skp = sk_p.predict(X_te)
    rows = [
        ("Своё (test)", accuracy_score(y_te, pred_te), f1_score(y_te, pred_te)),
        ("Прунинг, своё (test)", accuracy_score(y_te, pred_te_r), f1_score(y_te, pred_te_r)),
        ("Sklearn (test)", sklearn_accuracy(y_te, pred_te_sk), f1_score(y_te, pred_te_sk)),
        ("Sklearn+CCP (test)", sklearn_accuracy(y_te, pred_te_skp), f1_score(y_te, pred_te_skp)),
    ]
    print("\n=== Сводка (test) ===")
    for label, acc, f1 in rows:
        print(f"  {label:<22} acc={acc:.4f}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
