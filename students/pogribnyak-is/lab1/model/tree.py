from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class TreeNode:
    feature: Optional[int] = None
    threshold: Optional[Any] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[Any] = None
    samples: int = 0
    impurity: float = 0.0
    depth: int = 0

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class Splitter:
    @staticmethod
    def gini_impurity(y: np.ndarray) -> float:
        if len(y) == 0: return 0.0
        p = np.bincount(y) / len(y)
        p = p[p > 0]
        return float(1.0 - np.sum(p**2))

    @staticmethod
    def weighted_gini(left_y: np.ndarray, right_y: np.ndarray) -> float:
        n_l, n_r = len(left_y), len(right_y)
        n = n_l + n_r
        if n == 0: return 0.0
        return (n_l / n) * Splitter.gini_impurity(left_y) + (n_r / n) * Splitter.gini_impurity(right_y)

    @staticmethod
    def _spread_missing(missing_idx: np.ndarray, left_m: np.ndarray, right_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_m, right_m = left_m.copy(), right_m.copy()
        n_l, n_r = int(left_m.sum()), int(right_m.sum())
        tot = n_l + n_r
        if missing_idx.size == 0: return left_m, right_m
        k = int(len(missing_idx) * (n_l / tot)) if tot > 0 else len(missing_idx) // 2
        left_m[missing_idx[:k]] = True
        right_m[missing_idx[k:]] = True
        return left_m, right_m

    @staticmethod
    def find_best_ordered_split(
        X: np.ndarray, y: np.ndarray, feature_idx: int, missing_mask: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], float, np.ndarray, np.ndarray]:
        fv = X[:, feature_idx]
        uniq = np.sort(np.unique(fv[~np.isnan(fv)]))
        if len(uniq) < 2:
            return None, float("inf"), np.array([]), np.array([])

        best_g, best_th, best_li, best_ri = float("inf"), None, None, None
        for th in (uniq[:-1] + uniq[1:]) / 2.0:
            if missing_mask is not None and missing_mask.any():
                lm = (fv <= th) & ~missing_mask
                rm = (fv > th) & ~missing_mask
                lm, rm = Splitter._spread_missing(np.where(missing_mask)[0], lm, rm)
            else:
                lm, rm = fv <= th, fv > th
            li, ri = np.where(lm)[0], np.where(rm)[0]
            if len(li) == 0 or len(ri) == 0:
                continue
            g = Splitter.weighted_gini(y[li], y[ri])
            if g < best_g:
                best_g, best_th, best_li, best_ri = g, float(th), li, ri

        if best_th is None:
            return None, float("inf"), np.array([]), np.array([])
        return best_th, best_g, best_li, best_ri

    @staticmethod
    def find_best_numeric_split(
        X: np.ndarray, y: np.ndarray, feature_idx: int, missing_mask: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], float, np.ndarray, np.ndarray]:
        fv = X[:, feature_idx]
        if np.sum(~np.isnan(fv)) < 2:
            return None, float("inf"), np.array([]), np.array([])
        th = 0.5
        if missing_mask is not None and missing_mask.any():
            lm = (fv <= th) & ~missing_mask
            rm = (fv > th) & ~missing_mask
            lm, rm = Splitter._spread_missing(np.where(missing_mask)[0], lm, rm)
        else:
            lm, rm = fv <= th, fv > th
        li, ri = np.where(lm)[0], np.where(rm)[0]
        if len(li) == 0 or len(ri) == 0:
            return None, float("inf"), np.array([]), np.array([])
        return th, Splitter.weighted_gini(y[li], y[ri]), li, ri


class DecisionTree:
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root: Optional[TreeNode] = None
        self.feature_names: Optional[List[str]] = None
        self.n_features_ = 0
        self.n_classes_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        X, y = np.asarray(X), np.asarray(y)
        if len(X) != len(y):
            raise ValueError("X и y разной длины")
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features_)]
        uniq = np.unique(y)
        m = {v: i for i, v in enumerate(uniq)}
        y_int = np.array([m[v] for v in y])
        self.root = self._build_tree(X, y_int, 0, set())

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, used: set) -> TreeNode:
        n_samples = len(X)
        n_cls = len(np.unique(y))
        imp = Splitter.gini_impurity(y)
        node = TreeNode(samples=n_samples, impurity=imp, depth=depth)

        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or n_cls == 1
            or imp == 0.0
        ):
            node.value = int(np.argmax(np.bincount(y)))
            return node

        best_f = best_t = best_li = best_ri = None
        best_g = float("inf")

        for j in range(X.shape[1]):
            if j in used:
                continue
            miss = np.isnan(X[:, j])
            uq = np.unique(X[~np.isnan(X[:, j]), j])
            if len(uq) <= 1:
                continue
            if len(uq) == 2 and all(v in (0, 1) for v in uq):
                t, g, li, ri = Splitter.find_best_numeric_split(X, y, j, miss)
            else:
                t, g, li, ri = Splitter.find_best_ordered_split(X, y, j, miss)
            if t is not None and g < best_g:
                best_g, best_f, best_t, best_li, best_ri = g, j, t, li, ri

        if (
            best_f is None
            or len(best_li) < self.min_samples_leaf
            or len(best_ri) < self.min_samples_leaf
            or (imp - best_g) < self.min_impurity_decrease
        ):
            node.value = int(np.argmax(np.bincount(y)))
            return node

        node.feature, node.threshold = best_f, best_t
        node.left = self._build_tree(X[best_li], y[best_li], depth + 1, used.copy())
        node.right = self._build_tree(X[best_ri], y[best_ri], depth + 1, used.copy())
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([_predict_from(self.root, row) for row in X], dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        P = np.zeros((len(X), self.n_classes_))
        for i, row in enumerate(X):
            node = self.root
            while not node.is_leaf():
                v = row[node.feature]
                if np.isnan(v):
                    P[i] = 1.0 / self.n_classes_
                    break
                if isinstance(node.threshold, (int, float)):
                    node = node.left if v <= node.threshold else node.right
                else:
                    node = node.left if v == node.threshold else node.right
            else:
                P[i, node.value] = 1.0
        return P

    def get_depth(self) -> int:
        return 0 if self.root is None else self._depth(self.root)

    def _depth(self, node: TreeNode) -> int:
        if node.is_leaf():
            return node.depth
        return max(self._depth(node.left), self._depth(node.right))

    def get_n_leaves(self) -> int:
        return 0 if self.root is None else self._leaves(self.root)

    def _leaves(self, node: TreeNode) -> int:
        if node.is_leaf():
            return 1
        return self._leaves(node.left) + self._leaves(node.right)


def _copy_node(node: TreeNode) -> TreeNode:
    n = TreeNode(
        feature=node.feature,
        threshold=node.threshold,
        value=node.value,
        samples=node.samples,
        impurity=node.impurity,
        depth=node.depth,
    )
    if not node.is_leaf():
        n.left = _copy_node(node.left)
        n.right = _copy_node(node.right)
    return n


def _copy_tree(tree: DecisionTree) -> DecisionTree:
    t = DecisionTree(
        max_depth=tree.max_depth,
        min_samples_split=tree.min_samples_split,
        min_samples_leaf=tree.min_samples_leaf,
        min_impurity_decrease=tree.min_impurity_decrease,
    )
    t.n_features_, t.n_classes_, t.feature_names = tree.n_features_, tree.n_classes_, tree.feature_names
    t.root = _copy_node(tree.root)
    return t


def _split_val_rows(X: np.ndarray, node: TreeNode) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        e = np.array([], dtype=int)
        return e, e
    col = X[:, node.feature]
    miss = np.isnan(col)
    if isinstance(node.threshold, (int, float)):
        lm = (col <= node.threshold) & ~miss
        rm = (col > node.threshold) & ~miss
    else:
        lm = (col == node.threshold) & ~miss
        rm = (col != node.threshold) & ~miss
    lm, rm = Splitter._spread_missing(np.where(miss)[0], lm, rm)
    return np.where(lm)[0], np.where(rm)[0]


def _predict_from(node: TreeNode, sample: np.ndarray) -> int:
    cur = node
    while not cur.is_leaf():
        v = sample[cur.feature]
        if np.isnan(v):
            cur = cur.left
            continue
        if isinstance(cur.threshold, (int, float)):
            cur = cur.left if v <= cur.threshold else cur.right
        else:
            cur = cur.left if v == cur.threshold else cur.right
    return int(cur.value) if cur.value is not None else 0


def _prune(node: TreeNode, X: np.ndarray, y: np.ndarray) -> TreeNode:
    if node.is_leaf():
        return node
    li, ri = _split_val_rows(X, node)
    node.left = _prune(node.left, X[li], y[li])
    node.right = _prune(node.right, X[ri], y[ri])
    if len(y) == 0:
        return node
    pred = np.array([_predict_from(node, x) for x in X])
    maj = int(np.argmax(np.bincount(y)))
    if np.mean(maj != y) <= np.mean(pred != y):
        node.feature = node.threshold = None
        node.left = node.right = None
        node.value = maj
    return node


class ReducedErrorPruner:
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray):
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)

    def prune(self, tree: DecisionTree) -> DecisionTree:
        t = _copy_tree(tree)
        t.root = _prune(t.root, self.X_val, self.y_val)
        return t
