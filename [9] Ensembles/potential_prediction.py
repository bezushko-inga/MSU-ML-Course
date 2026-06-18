import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        for idx in range(x.shape[0]):
            current = x[idx].copy()

            row_ids = []
            col_ids = []

            height, width = current.shape

            for r in range(height):
                if current[r, :].min() <= 2:
                    row_ids.append(r)

            for c in range(width):
                if current[:, c].min() <= 2:
                    col_ids.append(c)

            if row_ids:
                row_center = (row_ids[0] + row_ids[-1]) // 2
                row_shift = height // 2 - row_center
                current = np.roll(current, row_shift, axis=0)

            if col_ids:
                col_center = (col_ids[0] + col_ids[-1]) // 2
                col_shift = width // 2 - col_center
                current = np.roll(current, col_shift, axis=1)

            x[idx] = current
        return x.reshape(x.shape[0], -1)


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = Pipeline(
        [
            ("vectorizer", PotentialTransformer()),
            (
                "regr",
                ExtraTreesRegressor(
                    n_estimators=1000,
                    random_state=265,
                    max_depth=100,
                    max_features=100,
                    criterion="absolute_error",
                ),
            ),
        ]
    )
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
