import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    """
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    """

    # Ваш код здесь:＼(º □ º l|l)/

    cluster_counts = np.unique(labels, return_counts=True)[1]
    if len(cluster_counts) == 1:
        return 0

    x = x[np.argsort(labels)]

    pairwise = sklearn.metrics.pairwise.pairwise_distances(
        x, metric="euclidean", n_jobs=-1
    )

    start = 0
    cluster_index = []
    cluster_sums = np.zeros((len(labels), len(cluster_counts)))

    for idx in range(len(cluster_counts)):
        size = cluster_counts[idx]
        finish = start + size
        block = pairwise[:, start:finish]
        cluster_sums[:, idx] = np.sum(block, axis=1)
        start += size
        cluster_index = cluster_index + [idx] * size

    rows = np.arange(len(cluster_sums))
    cols = cluster_index

    s = cluster_sums[rows, cols]
    s = np.divide(
        s,
        cluster_counts[cluster_index] - 1,
        where=cluster_counts[cluster_index] != 1,
        out=np.zeros_like(labels, dtype=float),
    )

    cluster_sums[rows, cols] = np.inf

    nearest_idx = np.argmin(cluster_sums, axis=1)
    rows = np.arange(cluster_sums.shape[0])
    d = cluster_sums[rows, nearest_idx] / cluster_counts[nearest_idx]

    max_ds = np.maximum(d, s)
    sil_score = np.divide(
        d - s, max_ds, where=max_ds != 0, out=np.zeros_like(labels, dtype=float)
    )
    sil_score[cluster_counts[cluster_index] == 1] = 0
    sil_score = sil_score.mean()

    return sil_score


def bcubed_score(true_labels, predicted_labels):
    """
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    """

    # Ваш код здесь:＼(º □ º l|l)/

    true_mask = true_labels[:, np.newaxis] == true_labels
    pred_mask = predicted_labels[:, np.newaxis] == predicted_labels
    common_mask = true_mask * pred_mask

    true_count = np.sum(true_mask, axis=1)
    pred_count = np.sum(pred_mask, axis=1)
    common_count = np.sum(common_mask, axis=1)

    precision = np.mean(common_count / pred_count)
    recall = np.mean(common_count / true_count)

    score = 2 * (precision * recall) / (precision + recall)

    return score
