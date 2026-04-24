from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame

from data.scaler import StandardScaler


class Dataset(ABC):
    def __init__(self, target_col: str = "target", seed: int = 42):
        self.seed = seed
        self.target_col = target_col
        self.df: DataFrame = DataFrame()
        self.X = self.y = None

        self.load_data()

    @abstractmethod
    def load(self) -> DataFrame:
        ...

    @abstractmethod
    def preprocess(self) -> None:
        ...

    def load_data(self):
        self.load()
        self.preprocess()
        self.X = self.df.drop(columns=[self.target_col]).to_numpy(dtype=np.float32)
        self.y = self.df[self.target_col].to_numpy()

    def split_indices(self, n: int, test_size=0.2, val_size=0.25):
        np.random.seed(self.seed)
        idx = np.random.permutation(n)
        nt = int(n * test_size)
        nv = int(n * val_size)
        return idx[nt + nv :], idx[nt : nt + nv], idx[:nt]

    def split_and_scale(self, test_size=0.2, val_size=0.25):
        tr, va, te = self.split_indices(len(self.X), test_size, val_size)
        self.X_train, self.y_train = self.X[tr], self.y[tr]
        self.X_val, self.y_val = self.X[va], self.y[va]
        self.X_test, self.y_test = self.X[te], self.y[te]
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_val = sc.transform(self.X_val)
        self.X_test = sc.transform(self.X_test)
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
