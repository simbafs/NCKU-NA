import numpy as np
import math

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
        b[i] = b[i] - np.dot(A[i, 0:i], b[0:i])

    # because a[n:n] = [], the last will be calc separately
    b[n-1] = b[n-1]/A[n-1, n-1]

    # solve x
    for i in range(n-2, -1, -1):
        b[i] = (b[i] - np.dot(A[i, i+1:n], b[i+1:n])) / A[i, i]

    return b

def polyFit(xData,yData,m):
    a = np.zeros((m+1,m+1))
    b = np.zeros(m+1)
    s = np.zeros(2*m+1)
    for i in range(len(xData)):
        temp = yData[i]
        for j in range(m+1):
            b[j] = b[j] + temp
            temp = temp*xData[i]
        temp = 1.0
        for j in range(2*m+1):
            s[j] = s[j] + temp
            temp = temp*xData[i]
    for i in range(m+1):
        for j in range(m+1):
            a[i,j] = s[i+j]
    return doolittle(a,b)

def stdDev(c,xData,yData):
    def evalPoly(c,x):
        m = len(c) - 1
        p = c[m]
        for j in range(m):
            p = p*x + c[m-j-1]
        return p
    n = len(xData) - 1
    m = len(c) - 1
    sigma = 0.0
    for i in range(n+1):
        p = evalPoly(c,xData[i])
        sigma = sigma + (yData[i] - p)**2
    sigma = math.sqrt(sigma/(n - m))
    return sigma

def printPolynomial(a, round = 6):
    print('f(x) = ', sep='', end='')
    for n in range(len(a)):
        print(np.round(a[n], round), 'x^', n, sep='', end='')
        if(n != len(a)-1): print(' + ', sep='', end='')
    print()

# polynomial(2, 4, 6, 8)(x) -> 2x^0 + 4x^1 + 6x^2 + 8x^3
def polynomial(a):
    def f(x):
        r = 0 
        for n in range(len(a)):
            r += a[n] * x**n
        return r
    return f

xData = np.array([0.0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150])
yData = np.array([1.0, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741])

cofficients = polyFit(xData, yData, 2)
printPolynomial(cofficients)
f = polynomial(cofficients)
print('f(10.5) = {f}'.format(f = f(10.5)))
