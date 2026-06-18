import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.max_data = None
        self.min_data = None

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.max_data = data.max(axis=0)
        self.min_data = data.min(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        return (data - self.min_data)/(self.max_data - self.min_data)


class StandardScaler:
    def __init__(self):
        self.mean_data = None
        self.std_data = None

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.mean_data = data.mean(axis=0)
        self.std_data = data.std(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        return (data - self.mean_data)/self.std_data
