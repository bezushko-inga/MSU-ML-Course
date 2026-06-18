import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super().__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.feature_names = list(X.columns)

        self.value_lists = [
            sorted(X[col].unique().tolist()) for col in self.feature_names
        ]

        self.value_to_index = [
            {val: idx for idx, val in enumerate(vals)}
            for vals in self.value_lists
        ]

        self.col_offsets = []
        offset = 0
        for vals in self.value_lists:
            self.col_offsets.append(offset)
            offset += len(vals)

        self.n_output_features_ = offset

        return self

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        n_samples = len(X)
        Z = np.zeros((n_samples, self.n_output_features_), dtype=self.dtype)

        for col_id, col in enumerate(self.feature_names):
            mp = self.value_to_index[col_id]
            base = self.col_offsets[col_id]
            col_values = X[col].values

            for row_id, val in enumerate(col_values):
                idx = mp.get(val)
                if idx is not None:
                    Z[row_id, base + idx] = 1.0

        return Z

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """

        self.cols = list(X.columns)
        self.successes = []
        self.counters = []

        for col in self.cols:
            column_vals = X[col].values
            uniq_vals = np.unique(column_vals)

            s_map = {}
            c_map = {}

            for u in uniq_vals:
                mask = (column_vals == u)
                s_map[u] = Y[mask].mean()
                c_map[u] = mask.mean()

            self.successes.append(s_map)
            self.counters.append(c_map)

        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        n_samples = len(X)
        n_features = len(self.cols)

        succ_mat = np.zeros((n_samples, n_features), dtype=self.dtype)
        cnt_mat = np.zeros((n_samples, n_features), dtype=self.dtype)

        for f_id, col in enumerate(self.cols):
            s_map = self.successes[f_id]
            c_map = self.counters[f_id]
            vals = X[col].values

            for row_id, v in enumerate(vals):
                succ_mat[row_id, f_id] = s_map.get(v, 0)
                cnt_mat[row_id, f_id] = c_map.get(v, 0)

        ratio = (succ_mat + a) / (cnt_mat + b)

        merged = []
        for f_id in range(n_features):
            merged.append(succ_mat[:, f_id])
            merged.append(cnt_mat[:, f_id])
            merged.append(ratio[:, f_id])

        return np.column_stack(merged).astype(self.dtype)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.cols = list(X.columns)
        n_samples = len(X)
        n_features = len(self.cols)

        y = np.asarray(Y, dtype=float)

        self.successes = np.zeros((n_samples, n_features), dtype=self.dtype)
        self.counters = np.zeros((n_samples, n_features), dtype=self.dtype)

        for valid_idx, train_idx in group_k_fold(n_samples, n_splits=self.n_folds, seed=seed):
            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]
            X_val = X.iloc[valid_idx]

            fold_s = []
            fold_c = []

            for col in self.cols:
                col_vals_tr = X_tr[col].values
                uniq_vals = np.unique(col_vals_tr)

                s_map = {}
                c_map = {}

                for u in uniq_vals:
                    mask = (col_vals_tr == u)
                    s_map[u] = y_tr[mask].mean()
                    c_map[u] = mask.mean()

                fold_s.append(s_map)
                fold_c.append(c_map)

            for f_id, col in enumerate(self.cols):
                col_vals_val = X_val[col].values
                s_map = fold_s[f_id]
                c_map = fold_c[f_id]

                for loc, v in enumerate(col_vals_val):
                    g = valid_idx[loc]
                    self.successes[g, f_id] = s_map.get(v, 0)
                    self.counters[g, f_id] = c_map.get(v, 0)

        return self

    def transform(self, X, a=1e-5, b=1e-5):
        """
        returns matrix [n_samples, 3*n_features]
        """
        s = self.successes
        c = self.counters
        r = (s + a) / (c + b)

        n_features = len(self.cols)

        merged = []
        for f in range(n_features):
            merged.append(s[:, f])
            merged.append(c[:, f])
            merged.append(r[:, f])

        return np.column_stack(merged).astype(self.dtype)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    """
    uniq = np.unique(x)
    w = []
    for v in uniq:
        mask = (x == v)
        w.append(y[mask].mean())

    return np.asarray(w, dtype=np.float64)