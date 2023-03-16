from typing import Callable
import numpy as np
import math

def buildA(n: int):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i == j :
                A[i, j] = 4
            elif i+1 == j or i-1 == j :
                A[i, j] = -1

    A[0, n-1] = 1
    A[n-1, 0] = 1

    return A

def buildB(n: int):
    b = np.zeros(n)
    b[n-1] = 100

    return b

def buildInitX(n: int):
    return np.ones(n)

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
        p = 1
    ):
    omega = 1.0
    dx1 = 0.0
    dx2 = 0.0

    for i in range(1, max+1):
        xNew = iterEqs(A, x, b, omega)

        # update omega
        dx = math.sqrt(np.dot(xNew - x, xNew - x))
        if dx < tol: return x, i , omega
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - (dx2 / dx1)**(1/p)))

        x = xNew

    print('Gauss-Seidel failed to converge')
    return x, max, omega

n = int(input('n(20): ') or '20')
A = buildA(n)
b = buildB(n)
x = buildInitX(n)

x, i, omega = gaussSeidel(iterEqs, A, x, b)
diff = np.dot(A, x) - b
print('result:', x, i , omega)
print('difference:', diff)
print('average diff: ', math.sqrt(np.dot(diff, diff)))
