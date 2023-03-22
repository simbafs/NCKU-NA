import numpy as np
import math

# directly copy from quiz2
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

# polynomial(2, 4, 6, 8)(x) -> 2x^0 + 4x^1 + 6x^2 + 8x^3
def polynomial(*a):
    def f(x):
        r = 0 
        for n in range(len(a)):
            r += a[n] * x**n
        return r
    return f

def printPolynomial(*a):
    print('f(x) = ', sep='', end='')
    for n in range(len(a)):
        print(a[n], 'x^', n, sep='', end='')
        if(n != len(a)-1): print(' + ', sep='', end='')
    print()

xData = np.array([1.0, 2.5, 3.5, 4.0, 1.1, 1.8, 2.2, 3.7])
yData = np.array([6.008, 15.722, 27.130, 33.772, 5.257, 9.549, 11.098, 28.828])

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

def stdDev(a, xData, yData):
    f = polynomial(*a)
    n = len(xData)-1
    m = len(a) - 1
    sigma = 0.0
    for i in range(n+1):
        sigma = sigma + (yData[i] - f(xData[i]))**2
    return math.sqrt(sigma/(n-m))

def testN(n):
    a = polyFit(xData, yData, n)
    d = stdDev(a, xData, yData)
    f = polynomial(*a)

    print('n =', n)
    printPolynomial(*a)
    print('stdDev: ', d)
    print('\t╭───────┬───────┬───────────────────────╮')
    print('', 'x', 'y', 'f(x)\t\t', '',  sep='\t│')
    for i in range(len(xData)):
        print('', xData[i], yData[i], f(xData[i]), '',  sep='\t│')
    print('\t╰───────┴───────┴───────────────────────╯')
    return a, d

bestN = math.inf
bestStddev = math.inf
bestA = ()
for n in (1, 2):
    a, stddev = testN(n)
    if stddev < bestStddev:
        bestN = n
        bestStddev = stddev
        bestA = a

print('n =', bestN, 'is better with stdDev =', bestStddev, ', which is ', end='')
printPolynomial(*bestA)
