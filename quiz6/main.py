import numpy as np

# directly copy from quzi4
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
def polynomial(a):
    def f(x):
        r = 0
        for n in range(len(a)):
            r += a[n] * x**n
        return r
    return f

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

def printPolynomial(name, a):
    print(name, '(x) = ', sep='', end='')
    for n in range(len(a)):
        print(a[n], 'x^', n, sep='', end='')
        if(n != len(a)-1): print(' + ', sep='', end='')
    print()
# end of copy from quiz4

def d(a):
    b = np.zeros(len(a)-1)
    for i in range(len(b)):
        b[i]= a[i+1]*(i+1)
    return b


xData = np.array([-2.2, -0.3, 0.8, 1.9])
yData = np.array([15.180, 10.962, 1.920, -2.040])

f = [8.448, -8.56, -0.3, 1]

F = polyFit(xData, yData, 3)

print('f       polynominal interpolation   error')
# df
df = polynomial(d(f))
dF = polynomial(d(F))

print(df(0), df(0), abs(df(0)-dF(0))/df(0), sep='\t', )

# ddf
ddf = polynomial(d(d(f)))
ddF = polynomial(d(d(F)))

print(ddf(0), ddf(0), abs(ddf(0)-ddF(0))/ddf(0), sep='\t')

print()
printPolynomial('f', f)
printPolynomial('F', F)
print()
printPolynomial('df', d(f))
printPolynomial('dF', d(F))
print()
printPolynomial('ddf', d(d(f)))
printPolynomial('ddF', d(d(F)))
