import numpy as np


def doolittle(_A: np.ndarray, _b: np.ndarray):
    A = _A.copy()
    b = _b.copy()
    n = len(A)

    # decomposition
    for i in range(0, n-1):
        for j in range(i+1, n):
            lamb = A[j, i]/A[i, i]
            # U
            A[j, i+1:n] = A[j, i+1:n] - lamb * A[i, i+1:n]
            # L
            A[j, i] = lamb

    # solve the problem
    # solve y
    for i in range(0, n):
        b[i] = b[i] - np.dot(b[0:i], A[i, 0:i])

    # because a[n:n] = [], the last will be calc separately
    b[n-1] = b[n-1]/A[n-1, n-1]

    # solve x
    for i in range(n-2, -1, -1):
        b[i] = (b[i] - np.dot(b[i+1:n], A[i, i+1:n])) / A[i, i]

    return b


A = np.array([[3.5, 2.77, 0.67, 1.8],
             [-1.8, 2.68, 3.44, -0.09],
             [0.27, 5.07, 6.9, 1.61],
             [1.71, 5.45, 2.68, 1.71]])
b = np.array([7.31, 4.23, 13.85, 11.55])

x = doolittle(A, b)
detA = np.linalg.det(A)
Ax = np.dot(A, x)
normA = np.sqrt(np.sum(A*A))

print('x is', x)
print('|A| is', detA, '||A|| is', normA)
print('The result of x is', x)
print('Ax is', Ax)
print('The error between Ax and b is', np.abs(Ax - b)/b)
