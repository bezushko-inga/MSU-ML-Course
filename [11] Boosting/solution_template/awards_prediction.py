from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

from numpy import ndarray
from lightgbm import LGBMRegressor

"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    train_df = pd.read_json(train_file, lines=True)
    test_df = pd.read_json(test_file, lines=True)

    train_df = train_df.fillna("label_empty")
    test_df = test_df.fillna("label_empty")

    list_columns = ["genres", "directors", "filming_locations", "keywords"]

    for col in list_columns:
        for row in range(train_df.shape[0]):
            if train_df.at[row, col] != "label_empty":
                train_df.at[row, col] = ",".join(
                    item for item in train_df.at[row, col] if item
                )
        train_df[col] = train_df[col].astype("category")

        for row in range(test_df.shape[0]):
            if test_df.at[row, col] != "label_empty":
                test_df.at[row, col] = ",".join(
                    item for item in test_df.at[row, col] if item
                )
        test_df[col] = test_df[col].astype("category")

    for actor in range(3):
        gender_column = f"actor_{actor}_gender"
        train_df[gender_column] = train_df[gender_column].astype("category")
        test_df[gender_column] = test_df[gender_column].astype("category")

    y_train = train_df["awards"]
    del train_df["awards"]

    regressor = LGBMRegressor(n_estimators=140, depth=10)
    regressor.fit(train_df, y_train)

    return regressor.predict(test_df)
