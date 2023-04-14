import numpy as np
from random import random
import cmath

def evalPoly(a):
    n = len(a) - 1
    def f(x):
        p = a[n]
        dp = 0.0 + 0.0j
        ddp = 0.0 + 0.0j
        for i in range(1,n+1):
            ddp = ddp*x + 2.0*dp
            dp = dp*x + p
            p = p*x + a[n-i]
        return p,dp,ddp
    return f

def polyRoots(a,tol=1.0e-12):
    def laguerre(a,tol):
        x = random() # Starting value (random number)
        n = len(a) - 1
        for i in range(30):
            p,dp,ddp = evalPoly(a)(x)
            if abs(p) < tol: return x
            g = dp/p
            h = g*g - ddp/p
            f = cmath.sqrt((n - 1)*(n*h - g*g))
            if abs(g + f) > abs(g - f): dx = n/(g + f)
            else: dx = n/(g - f)
            x = x - dx
            if abs(dx) < tol: return x
        print("Too many iterations")
    def deflPoly(a,root): # Deflates a polynomial
        n = len(a)-1
        b = [(0.0 + 0.0j)]*n
        b[n-1] = a[n]
        for i in range(n-2,-1,-1):
            b[i] = a[i+1] + root*b[i+1]
        return b
    n = len(a) - 1
    roots = np.zeros((n),dtype=complex)
    for i in range(n):
        x = laguerre(a,tol)
        if abs(x.imag) < tol: x = x.real
        roots[i] = x
        a = deflPoly(a,x)
    return roots

def printPolynomial(a, round = 6):
    print('f(x) = ', sep='', end='')
    for n in range(len(a)):
        print(np.round(a[n], round), 'x^', n, sep='', end='')
        if(n != len(a)-1): print(' + ', sep='', end='')
    print()
    
cm = 12.0
km = 1500.0 
cofficients = [km**2, km*cm, 2*km, 2*cm, 1]

printPolynomial(cofficients)
x = polyRoots(cofficients)
print('possible (omega_r +- omega_i):')
for i in range(len(x)):
    print(x[i])
