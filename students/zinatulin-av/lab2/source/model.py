import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomForestClassifier:

    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees_ = []
        self.feat_indices_ = []
        self.oob_indices_ = []
        self.classes_ = None
        self.n_features_ = None
        self.oob_score_ = None
        self.feature_importances_ = None

    def _get_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)) + 1)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        n_feat = self._get_max_features(self.n_features_)
        rng = np.random.default_rng(self.random_state)

        self.trees_ = []
        self.feat_indices_ = []
        self.oob_indices_ = []

        for _ in range(self.n_estimators):
            seed = rng.integers(0, 2**31 - 1)
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = np.where(oob_mask)[0]

            feat_idx = rng.choice(self.n_features_, size=n_feat, replace=False)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(seed),
            )
            tree.fit(X[np.ix_(boot_idx, feat_idx)], y[boot_idx])

            self.trees_.append(tree)
            self.feat_indices_.append(feat_idx)
            self.oob_indices_.append(oob_idx)

        self.oob_score_ = self._compute_oob_score(X, y)
        return self

    def _compute_oob_score(self, X, y):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        oob_preds = np.zeros((n_samples, n_classes))
        oob_counts = np.zeros(n_samples)

        for tree, feat_idx, oob_idx in zip(self.trees_, self.feat_indices_, self.oob_indices_):
            if len(oob_idx) == 0:
                continue
            proba = tree.predict_proba(X[np.ix_(oob_idx, feat_idx)])
            for i, idx in enumerate(oob_idx):
                oob_preds[idx] += proba[i]
                oob_counts[idx] += 1

        valid = oob_counts > 0
        if not np.any(valid):
            return 0.0
        pred_labels = self.classes_[np.argmax(oob_preds[valid], axis=1)]
        return np.mean(pred_labels == y[valid])

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes))

        for tree, feat_idx in zip(self.trees_, self.feat_indices_):
            proba_sum += tree.predict_proba(X[:, feat_idx])

        return proba_sum / len(self.trees_)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_feature_importances_oob(self, X, y, n_repeats=1, random_state=None):
        rng = np.random.default_rng(random_state)
        base = self._compute_oob_score(X, y)
        importances = np.zeros(self.n_features_)

        for j in range(self.n_features_):
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                perm_score = self._compute_oob_score(X_perm, y)
                drops.append(base - perm_score)
            importances[j] = np.mean(drops)

        if importances.sum() > 0:
            importances = importances / importances.sum()

        self.feature_importances_ = importances
        return importances

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self
