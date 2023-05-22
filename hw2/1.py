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

def printPolynomial(a, name='f'):
    print(name, '(x) = ', sep='', end='')
    for n in range(len(a)):
        print('{:.6f}'.format(a[n]), 'x^', n, sep='', end='')
        if(n != len(a)-1): print(' + ', sep='', end='')
    print()
# end of copy from quiz4

def d(a):
    b = np.zeros(len(a)-1)
    for i in range(len(b)):
        b[i]= a[i+1]*(i+1)
    return b

t = np.array([9, 10, 11]).astype(np.float64)
alpha = np.array([54.80, 54.06, 53.34])/180*np.pi
beta = np.array([65.59, 64.59, 63.62])/180*np.pi
a = 500 

xData = a*np.tan(beta)/(np.tan(beta)-np.tan(alpha))
yData = a*np.tan(alpha)*np.tan(beta)/(np.tan(beta)-np.tan(alpha))

Fx = polyFit(t, xData, 3)
Fy = polyFit(t, yData, 3)

dFx = d(Fx)
dFy = d(Fy)

printPolynomial(Fx, 'F_x')
printPolynomial(Fy, 'F_y')

printPolynomial(dFx, 'dF_x/dt')
printPolynomial(dFy, 'dF_y/dt')

print('t\tx\t\ty')
for i in range(len(xData)):
    print('{},\t{:.6f},\t{:.6f}'.format(t[i], xData[i], yData[i]))

vx = polynomial(dFx)(10)
vy = polynomial(dFy)(10)

print('v(10) = {:.6f}, climb angle = {:.6f}°'.format(np.sqrt(vx**2+vy**2), np.arctan(vy/vx)/np.pi*180))
