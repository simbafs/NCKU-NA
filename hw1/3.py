import numpy as np

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

def polyFit(xData: np.ndarray, yData: np.ndarray, m:int):
    A = np.zeros((m+1, m+1))
    b = np.zeros(m+1)
    s = np.zeros(2*m+1)
    for i in range(len(xData)):
        tmp = yData[i]
        for j in range(m+1):
            b[j] = b[j] + tmp
            tmp = tmp*xData[i]
        tmp = 1.0
        for j in range(2*m+1):
            s[j] = s[j] + tmp
            tmp = tmp*xData[i]
    for i in range(m+1):
        for j in range(m+1):
            A[i][j] = s[i+j]

    return doolittle(A, b)

xData = np.array([0.0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150])
yData = np.array([1.0, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741])

def dev(xData, yData, poly):
    d = 0
    for i in range(len(xData)):
        xx = poly(xData[i])
        x = yData[i]
        d += (x-xx)**2
    return np.sqrt(d)
    

for i in range(1, 7):
    cofficients = polyFit(xData, yData, i)
    poly = polynomial(cofficients)
    print('degree = {i}, dev = {dev}'.format(i=i, dev = round(dev(xData, yData, poly), 6)))
    printPolynomial(cofficients)
    print('f(10.5) = {f}'.format(f=round(poly(10.5), 6)))
    print()
