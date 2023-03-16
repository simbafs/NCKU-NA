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
def Av(A, v: np.ndarray):
    return np.dot(A, v)

def conjGrad(A, x, b, tol = 1e-9):
    n = len(b)
    r = b - Av(A, x)
    s = r.copy()
    i = 0

    for i in range(n):
        u = Av(A, s)
        alpha = np.dot(s, r)/np.dot(s, u)
        x = x + alpha*s
        r = b - Av(A, x)
        if(math.sqrt(np.dot(r, r))) < tol: break
        else: 
            beta = -np.dot(r, u)/np.dot(s, u)
            s = r + beta*s

    return x, i

n = int(input('n(20): ') or '20')
A = buildA(n)
b = buildB(n)
x = buildInitX(n)

x, i = conjGrad(A, x, b)
diff = np.dot(A, x) - b
print('result:', x, i)
print('difference:', diff)
print('average diff: ', math.sqrt(np.dot(diff, diff)))
