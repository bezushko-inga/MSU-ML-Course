from re import X
import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x_elem, x_count = np.unique(x, return_counts=True)
    y_elem, y_count = np.unique(y, return_counts=True)
    return np.array_equal(x_elem, y_elem) and np.array_equal(x_count, y_count)


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    ans = -1
    if x.size >= 2:
        x1, x2 = x[:-1], x[1:]
        mask = (x1 % 3 == 0) | (x2 % 3 == 0)
        if mask.any():
            ans = int(np.max(x1[mask]*x2[mask]))
    return ans


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.tensordot(image, weights, axes=([2], [0]))


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    X = np.repeat(x[:, 0], x[:, 1])
    Y = np.repeat(y[:, 0], y[:, 1])
    if len(X) != len(Y):
        return -1
    return np.dot(X, Y)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    scalar = np.float64(X @ Y.T)
    normed_x, normed_y = np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1)
    normprod = normed_x.reshape(-1, 1)*normed_y.reshape(1, -1)
    cos_dist = np.divide(scalar, normprod, out=np.zeros_like(scalar), where=normprod != 0)
    cos_dist[normprod == 0] = 1
    return cos_dist
