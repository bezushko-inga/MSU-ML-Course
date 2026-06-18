from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    n = len(X)
    m = len(X[0])
    Y = []
    for i in range(0, n, 4):
        Y_i = []
        for j in range(120, 500, 5):
            Y_i.append(X[i][j])
        Y.append(Y_i)
    return Y


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    n = len(X)
    m = len(X[0])
    dsum = 0
    found = False
    for i in range(min(n, m)):
        if X[i][i] >= 0:
            found = True
            dsum += X[i][i]
    if not found:
        dsum = -1
    return dsum


def replace_values(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    n = len(X)
    m = len(X[0])
    Y = deepcopy(X)
    for col in range(m):
        M = 0
        for row in range(n):
            M += X[row][col]
        M /= n
        for row in range(n):
            el = X[row][col]
            if el > 1.5*M or el < 0.25*M:
                Y[row][col] = -1
    return Y