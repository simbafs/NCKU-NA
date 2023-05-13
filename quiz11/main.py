import numpy as np
import math
from numpy.random import rand

def householder(a):
    n = len(a)
    for k in range(n-2):
        u = a[k+1:n,k]
        uMag = math.sqrt(np.dot(u,u))
        if u[0] < 0.0: uMag = -uMag
        u[0] = u[0] + uMag
        h = np.dot(u,u)/2.0
        v = np.dot(a[k+1:n,k+1:n],u)/h
        g = np.dot(u,v)/(2.0*h)
        v = v - g*u
        a[k+1:n,k+1:n] = a[k+1:n,k+1:n] - np.outer(v,u)  -np.outer(u,v)
        a[k,k+1] = -uMag
    return np.diagonal(a),np.diagonal(a,1)

def computeP(a):
    n = len(a)
    p = np.identity(n)*1.
    return p

def ridder(f,a,b,tol=1.0e-9):
    xOld = 0
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    # if np.sign(f2)!= np.sign(f3): x1 = x3; f1 = f3
    for i in range(30):
        c = 0.5*(a + b); fc = f(c)
        s = math.sqrt(fc**2 - fa*fb)
        if s == 0.0: return None
        dx = (c - a)*fc/s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx; fx = f(x)
        if i > 0:
            if abs(x - xOld) < tol*max(abs(x),1.0): return x
        xOld = x
        if np.sign(fc) == np.sign(fx):
            if np.sign(fa)!= np.sign(fx): b = x; fb = fx
            else: a = x; fa = fx
        else:
            a = c; b = x; fa = fc; fb = fx
    return None
    print('Too many iterations')

def sturmSeq(d,c,lam):
    n = len(d) + 1
    p = np.ones(n)
    p[1] = d[0] - lam
    for i in range(2,n):
        p[i] = (d[i-1] - lam)*p[i-1] - (c[i-2]**2)*p[i-2]
    return p

def lamRange(d,c,N):
    lamMin,lamMax = gerschgorin(d,c)
    r = np.ones(N+1)
    r[0] = lamMin
    for k in range(N,0,-1):
        lam = (lamMax + lamMin)/2.0
        h = (lamMax - lamMin)/2.0
        for i in range(1000):
            p = sturmSeq(d,c,lam)
            numLam = numLambdas(p)
            h = h/2.0
            if numLam < k: lam = lam + h
            elif numLam > k: lam = lam - h
            else: break
        lamMax = lam
        r[k] = lam
    return r

def gerschgorin(d,c):
    n = len(d)
    lamMin = d[0] - abs(c[0])
    lamMax = d[0] + abs(c[0])
    for i in range(1,n-1):
        lam = d[i] - abs(c[i]) - abs(c[i-1])
        if lam < lamMin: lamMin = lam
        lam = d[i] + abs(c[i]) + abs(c[i-1])
        if lam > lamMax: lamMax = lam
    lam = d[n-1] - abs(c[n-2])
    if lam < lamMin: lamMin = lam
    lam = d[n-1] + abs(c[n-2])
    if lam > lamMax: lamMax = lam
    return lamMin,lamMax

def numLambdas(p):
    n = len(p)
    signOld = 1
    numLam = 0
    for i in range(1,n):
        if p[i] > 0.0: sign = 1
        elif p[i] < 0.0: sign = -1
        else: sign = -signOld
        if sign*signOld < 0: numLam = numLam + 1
        signOld = sign
    return numLam

def inversePower3(d,c,s,tol=1.0e-6):
    n = len(d)
    e = c.copy()
    dStar = d - s 
    LUdecomp3(c,dStar,e) 
    x = rand(n) 
    xMag = math.sqrt(np.dot(x,x)) 
    x =x/xMag
    xOld = x
    for i in range(300): 
        xOld = x.copy() 
        LUsolve3(c,dStar,e,x) 
        xMag = math.sqrt(np.dot(x,x)) 
        x = x/xMag
        if np.dot(xOld,x) < 0.0: 
            sign = -1.0
            x = -x
    else: sign = 1.0
    if math.sqrt(np.dot(xOld - x,xOld - x)) < tol:
        return s + sign/xMag,x
    print('Inverse power method did not converge')

def LUdecomp3(c,d,e):
    c.setflags(write = 1)
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b

def eigenvals3(d,c,N):
    def f(x): # f(x) = |[A] - x[I]|
        p = sturmSeq(d,c,x)
        return p[len(p)-1]
    lam = np.zeros(N)
    r = lamRange(d,c,N) # Bracket eigenvalues
    for i in range(N): # Solve by Ridder’s method
        lam[i] = ridder(f,r[i],r[i+1])
    return lam

N = 3 
a = np.array([[7, -4, 3, -2, 1, 0],
              [-4, 8, -4, 3, -2, 1],
              [3, -4, 9, -4, 3, -2],
              [-2, 3, -4, 10, -4, 3],
              [1, -2, 3, -4, 11, -4],
              [0, 1, -2, 3, -4, 12]]).astype(np.float64)
xx = np.zeros((len(a),N))
d,c = householder(a) 
p = computeP(a) 
lambdas = eigenvals3(d,c,N) 
for i in range(N):
    s = lambdas[i]*1.0000001 
    lam,x = inversePower3(d,c,s) 
    xx[:,i] = x 
xx = np.dot(p,xx) 

print("Eigenvalues:\n",lambdas)
print("\nEigenvectors:\n",xx)
