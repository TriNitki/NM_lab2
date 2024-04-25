from typing import Any
import numpy as np
from numpy import ndarray
import math

from utils import Input

def norm(obj):
    # Получение норма вектора или матрицы
    def norm2(A):
        f = 0

        for i in range(0, len(A)):
            for j in range(0, len(A[0])):
                f += abs(A[i][j])**2

        return math.sqrt(f)
    
    def norm1(B):
        return math.sqrt(sum([pow(_, 2) for _ in B]))
    
    if not isinstance(obj[0], list):
        result = norm1(obj)
    else:
        result = norm2(obj)
    
    if result == 0:
        result = 0.00000001
    
    return result


def ort(inp: Input):
    '''Ортагонализация'''
    def _scalar_mul(arr1, arr2):
        s = 0
        for x, y in zip(arr1, arr2):
            s += x * y
        return s
    
    U = inp.matrix
    
    for j in range(len(U[0])):
        for k in range(j):
            U[:, j] -= _scalar_mul(U[:, k], U[:, j]) * U[:, k]
        U[:, j] /= norm(U[:, j])
    
    return U


def inv(matrix):
    '''Получение обратной матрицы'''
    def minor(matrix):
        minor_matrix = []
        for i in range(len(matrix)):
            minor_row = []
            for j in range(len(matrix)):
                minor = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
                minor_row.append((-1) ** (i + j) * determinant(minor))
            minor_matrix.append(minor_row)
        return minor_matrix
    
    def determinant(matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        elif len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            det = 0
            for i in range(len(matrix)):
                minor = [row[:i] + row[i+1:] for row in matrix[1:]]
                det += (-1) ** i * matrix[0][i] * determinant(minor)
            return det
    
    def transpose(matrix):
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    matrix = matrix.tolist()
    
    det = determinant(matrix)
    adjoint = transpose(minor(matrix))
    inverse_matrix = [[element / det for element in row] for row in adjoint]
    return np.array(inverse_matrix)


def dot(M, v):
    def mydot(v1, v2):
        return sum([x*y for x,y in zip(v1, v2)])
    
    M, v = M.tolist(), v.tolist()
    
    res = [mydot(r,v) for r in M]
    
    return np.array(res)


def solve(orted: ndarray[Any], b):
    '''Решение по СЛАУ'''
    orted = orted.tolist()
    n = len(orted)

    Ab = [row[:] + [bi] for row, bi in zip(orted, b)]

    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(Ab[r][i]))
        Ab[i], Ab[max_row] = Ab[max_row], Ab[i]

        divisor = Ab[i][i]
        for j in range(i, n + 1):
            Ab[i][j] /= divisor

        for k in range(i + 1, n):
            multiplier = Ab[k][i]
            for j in range(i, n + 1):
                Ab[k][j] -= multiplier * Ab[i][j]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i][n]
        for j in range(i + 1, n):
            x[i] -= Ab[i][j] * x[j]

    return np.array(x)


def solve_iter(inp: Input):
    '''Решение при помощи метода простой итерации'''
    def matmult(A, B):
        m1, m2 = A.tolist(), B.tolist()
        
        result = [
            [sum(x * y for x, y in zip(m1_r, m2_c)) for m2_c in zip(*m2)] for m1_r in m1
        ]
        
        return np.array(result)

    order = len(inp.matrix)
    x = np.zeros(order)
    accr = np.ones(order) * 1e-5

    x_prev = np.random.rand(order)

    while (norm(x - x_prev) / norm(x) > norm(accr)):
        x_prev = x
        x = dot(inv(inp.matrix), inp.vector) + dot(np.eye(order) - matmult(inv(inp.matrix), inp.matrix), x)

    return x


def get_nev_vector(inp: Input, slau):
    '''Получение вектора невязки'''
    eps = 0
    res = []
    for i in range(len(inp.vector)):
        for j in range(len(slau)):
            eps += slau[j] * inp.matrix[i][j]
        eps -= inp.vector[i]
        res.append(eps)
        eps = 0
    
    return np.array(res)


def norm_nev_vector(nev):
    '''Нормализация вектора невязки'''
    return math.sqrt(sum([pow(_, 2) for _ in nev]))


def get_nev_matrix(inp: Input):
    '''Получение матрицы невязки'''
    blank = np.eye(len(inp.matrix)) 
    res = inp.matrix * (blank / inp.matrix) - blank
    return res


def norm_nev_matrix(mat_nev):
    '''Нормировка матрицы невязки'''
    return norm(mat_nev)