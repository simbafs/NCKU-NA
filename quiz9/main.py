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
        b[i] = b[i] - np.dot(b[0:i], A[i, 0:i])

    # because a[n:n] = [], the last will be calc separately
    b[n-1] = b[n-1]/A[n-1, n-1]

    # solve x
    for i in range(n-2, -1, -1):
        b[i] = (b[i] - np.dot(b[i+1:n], A[i, i+1:n])) / A[i, i]

    return b

def newtonRaphson2(f,x,tol=1.0e-9):
    def jacobian(f,x):
        h = 1.0e-4
        n = len(x)
        jac = np.zeros((n,n))
        f0 = f(x)
        for i in range(n):
            temp = x[i]
            x[i] = temp + h
            f1 = f(x)
            x[i] = temp
            jac[:,i] = (f1 - f0)/h
        return jac,f0

    for i in range(30):
        jac,f0 = jacobian(f,x)
        if math.sqrt(np.dot(f0,f0)/len(x)) < tol: return x
        dx = doolittle(jac,-f0)
        x = x + dx
        if math.sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)),1.0):
            return x
    print('Too many iterations')

def residual(y): # Residuals of finite diff. Eqs. (8.11)
    hh = h*h
    r = np.zeros(m + 1)
    r[0] = y[0] - alpha
    for i in range(1,m):
        r[i] = y[i-1] - 2.0*y[i] + y[i+1] - hh*F(x[i],y[i],(y[i+1] - y[i-1])/(2.0*h))
    r[m] = 2*y[m-1] - 2*y[m] - hh*F(x[m], y[m], beta) + 2*h*beta
    return r

def startSoln(x): # Starting solution y(x)
    y = np.zeros(m + 1)
    for i in range(m + 1): y[i] = 0.5*x[i]
    return y

def F(x, y, yy):
    return - 4*x*y*yy - 2*y**2

def f(x):
    return 1/(2+x**2)

xStart = 0.0 # x at left end
xStop = 1.0 # x at right end
alpha = 0.5 
beta = -2/9
m = 20 # Number of mesh intervals
h = (xStop - xStart)/m
x = np.arange(xStart,xStop + h,h)
y = newtonRaphson2(residual,startSoln(x),1.0e-5)

maxDiff = 0 
maxDiffIndex = 0
print("x             y         diff")
for i in range(m+1):
    diff = f(x[i])-y[i]
    if diff > maxDiff: 
        maxDiff = diff
        maxDiffIndex = i
    print('{:.5f}{:14.5f}{:14.5e}'.format(x[i], y[i], diff))

print('max difference is {:.5e} at x = {}, error rate = {:.5f}%'.format(maxDiff, x[maxDiffIndex], maxDiff/y[maxDiffIndex]))
