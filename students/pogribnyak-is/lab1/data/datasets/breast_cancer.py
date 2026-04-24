import numpy as np
from pandas import DataFrame
from sklearn.datasets import load_breast_cancer

from data.dataset import Dataset


class BreastCancerDataset(Dataset):
    def load(self) -> None:
        data = load_breast_cancer()
        self.df = DataFrame(data.data, columns=data.feature_names)
        self.df["target"] = data.target

    def preprocess(self) -> None:
        self.df["target"] = np.where(self.df["target"] == 0, -1, 1)
        X = self.df.drop(columns=["target"]).copy()
        rng = np.random.default_rng(self.seed)
        miss_rate = 0.06
        mask = rng.random(X.shape) < miss_rate
        X = X.mask(mask)
        self.df[X.columns] = X
        self.df = self.df.astype(np.float32)
