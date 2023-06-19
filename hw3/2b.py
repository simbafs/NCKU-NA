import numpy as np
import math

def buildH(n):
    H = np.zeros((n,n))
    
    H[0, 0:3] = [7, -4, 1]
    H[1, 0:4] = [-4, 6, -4, 1]
    H[n-2, n-4:n] = [1, -4, 5, -2]
    H[n-1, n-3:n] = [1, -2, 1]

    return H 

def jacobi(a,tol = 1.0e-8): # Jacobi method
    def threshold(a):
        sum = 0.0
        for i in range(n-1):
            for j in range (i+1,n):
                sum = sum + abs(a[i,j])
        return 0.5*sum/n/(n-1)

    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
        aDiff = a[l,l] - a[k,k]
        if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
        else:
            phi = aDiff/(2.0*a[k,l])
            t = 1.0/(abs(phi) + math.sqrt(phi**2 + 1.0))
            if phi < 0.0: t = -t
        c = 1.0/math.sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = a[k,l]
        a[k,l] = 0.0
        a[k,k] = a[k,k] - t*temp
        a[l,l] = a[l,l] + t*temp
        for i in range(k): # Case of i < k
            temp = a[i,k]
            a[i,k] = temp - s*(a[i,l] + tau*temp)
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(k+1,l): # Case of k < i < l
            temp = a[k,i]
            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(l+1,n): # Case of i > l
            temp = a[k,i]
            a[k,i] = temp - s*(a[l,i] + tau*temp)
            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
        for i in range(n): # Update transformation matrix
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])

    n = len(a)

    p = np.identity(n,float)
    for k in range(20):
        mu = threshold(a) # Compute new threshold
        for i in range(n-1): # Sweep through matrix
            for j in range(i+1,n):
                if abs(a[i,j]) >= mu:
                    rotate(a,p,i,j)
        if mu <= tol: return np.diagonal(a),p
    print('Jacobi method did not converge')

n = 10
H = buildH(n)
lam, x = jacobi(H)

def printVector(vec):
    s = '[{:.6f}'.format(vec[0])
    for i in range(len(vec)-1):
        s = '{}, {:.6f}'.format(s, vec[i])
    s = '{}]'.format(s)

    return s

for i in range(len(lam)):
    print('lam: {:.7e}, y: {}'.format(lam[i], printVector(x[i])))
