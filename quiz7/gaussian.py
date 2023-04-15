import numpy as np 
import math

def f(x, dx=1e-6):
    if x == 2:
        x = x-dx # prevent from divided by 0
    if x == -2:
        x = x+dx
    return 2*np.sqrt((16-3*x**2)/(16-4*x**2))

def gaussNodes(m, tol=10e-9):

    def legendre(t, m):
        p0 = 1.0
        p1 = t
        p = 0
        for k in range(1, m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1
            p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p,dp

    A = np.zeros(m)
    x = np.zeros(m)
    nRoots = int((m + 1)/2) # Number of non-neg. roots
    for i in range(nRoots):
        t = math.cos(math.pi*(i + 0.75)/(m + 0.5))# Approx. root
        for j in range(30):
            p,dp = legendre(t,m) # Newton-Raphson
            dt = -p/dp; t = t + dt # method
            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break
    return x, A

def gaussQuad(f, a, b, m, tol=1e-9):
    c1 = (b + a)/2.0
    c2 = (b - a)/2.0
    x,A = gaussNodes(m, tol)
    sum = 0.0
    for i in range(len(x)):
        sum = sum + A[i]*f(c1 + c2*x[i])
    return c2*sum

tol = 1e-5
I = 0
a = -2
b = 2
k = 0

for k in range(1, 1000):
    Inew = gaussQuad(f, a, b, k, tol)
    print('k = {}, Inew = {}, difference = {}'.format(k, Inew, abs(Inew - I)))
    if abs(Inew - I) < tol: break
    I = Inew

print('iterat {k} times, integral result is {I}'.format(k = k, I = I))
