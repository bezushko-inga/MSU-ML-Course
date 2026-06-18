from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return Counter(x) == Counter(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    prod = -1
    n = len(x)
    for i in range(n - 1):
        a, b = x[i], x[i + 1]
        if a % 3 == 0 or b % 3 == 0:
            prod = max(prod, a*b)
    return prod


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    h = len(image)
    w = len(image[0])
    n = len(image[0][0])
    res = [[0. for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            summa = 0
            for k in range(n):
                summa += image[i][j][k]*weights[k]
            res[i][j] = summa
    return res


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    X, Y = [], []
    for value, count in x:
        X.extend([value]*count)
    for value, count in y:
        Y.extend([value]*count)
    if len(X) != len(Y):
        return -1
    return sum(i*j for i, j in zip(X, Y))


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    n, m, d = len(X), len(Y), len(Y[0])
    cos_dist = []
    for i in range(n):
        row = []
        for j in range(m):
            normed_x, normed_y, s = 0, 0, 0
            for k in range(d):
                normed_x += X[i][k]**2
                normed_y += Y[j][k]**2
                s += X[i][k]*Y[j][k]
            if normed_x == 0 or normed_y == 0:
                row.append(1)
            else:
                row.append(s/(normed_x*normed_y)**0.5)
        cos_dist.append(row)
    return cos_dist
