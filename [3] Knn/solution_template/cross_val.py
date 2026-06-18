import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    m = np.arange(num_objects)
    k = num_objects//num_folds
    f = []
    for i in range(num_folds):
        if i != num_folds - 1:
            f.append(m[i*k:(i+1)*k])
        else:
            f.append(m[i*k:])
    vt = []
    for i in range(num_folds):
        vt.append((np.hstack([f[j] for j in range(num_folds) if j != i]), f[i]))
    return vt


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    res = {}
    for norm, name in parameters["normalizers"]:
        for n in parameters["n_neighbors"]:
            for m in parameters["metrics"]:
                for w in parameters["weights"]:
                    sc = []
                    for t, v in folds:
                        x_t = X[t]
                        y_t = y[t]
                        x_v = X[v]
                        y_v = y[v]
                        if name != "None":
                            norm.fit(x_t)
                            x_t = norm.transform(x_t)
                            x_v = norm.transform(x_v)
                        model = knn_class(n_neighbors=n, metric=m, weights=w)
                        model.fit(x_t, y_t)
                        y_p = model.predict(x_v)
                        s = score_function(y_v, y_p)
                        sc.append(s)
                    res[(name, n, m, w)] = np.mean(sc)
    return res
