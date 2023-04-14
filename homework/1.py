import numpy as np 
from typing import Callable

# beginning of copy from quiz3

# modify this function to optimize calculations for a particular A
def sigma(Ai: np.ndarray, x: np.ndarray, i: int):
    sum = 0.0
    for j in range(len(Ai)):
        if j == i :
            continue
        sum += Ai[j]*x[j]
    return sum

def iterEqs(A: np.ndarray, x: np.ndarray, b: np.ndarray, omega: float):
    x = x.copy()
    n = len(x)

    for i in range(0, n):
        x[i] = omega*(b[i]-sigma(A[i], x, i))/A[i, i] + (1 - omega)*x[i]

    return x

def gaussSeidel(
        iterEqs: Callable,
        A: np.ndarray,
        x: np.ndarray,
        b: np.ndarray, 
        tol = 1e-9, 
        max = 500, 
        k = 10, 
        p = 1,
        relaxation = True
    ):
    omega = 1.0
    dx1 = 0.0
    dx2 = 0.0

    for i in range(1, max+1):
        xNew = iterEqs(A, x, b, omega)

        # update omega
        dx = np.sqrt(np.dot(xNew - x, xNew - x))
        if dx < tol: return x, i , omega
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            if relaxation:
                omega = 2.0/(1.0 + np.sqrt(1.0 - (dx2 / dx1)**(1/p)))

        x = xNew

    print('Gauss-Seidel failed to converge')
    return x, max, omega

# end of copy from quiz3

A = np.array([[-3.0, 1.0, 0.0, 0.0, 0.0],
              [3.0, -6.0, 3.0, 0.0, 0.0],
              [0.0, 3.0, -6.0, 3.0, 0.0],
              [0.0, 0.0, 3.0, -6.0, 3.0],
              [0.0, 0.0, 0.0, 3.0, -5.0]])
x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
b = np.array([-80.0, 0.0, 0.0, 60.0, 0.0])

def round(arr):
    result = []
    for i in range(len(arr)):
        result.append('{:.4f}'.format(arr[i]))
    return result

x, max, omega = gaussSeidel(iterEqs, A, x, b, tol=1e-4, relaxation=True)
print('with relaxation')
print('x = {x}, iteration = {max}'.format(x=round(x), max=max))
print('Ax = {Ax}, b = {b}'.format(Ax = round(np.dot(A, x)), b = b))
print('difference = {difference}'.format(difference = np.dot(A, x)-b))
print()

x, max, omega = gaussSeidel(iterEqs, A, x, b, tol=1e-4, relaxation=False)
print('without relaxation')
print('x = {x}, iteration = {max}'.format(x=round(x), max=max))
print('Ax = {Ax}, b = {b}'.format(Ax = round(np.dot(A, x)), b = b))
print('difference = {difference}'.format(difference = np.dot(A, x)-b))
