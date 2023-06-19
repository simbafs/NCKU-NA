import numpy as np 
import math

def downhill(F,xStart,side=0.1,tol=1.0e-6):
    iLo = 0
    n = len(xStart) # Number of variables
    x = np.zeros((n+1,n))
    f = np.zeros(n+1)
    # Generate starting simplex
    x[0] = xStart
    for i in range(1,n+1):
        x[i] = xStart
        x[i,i-1] = xStart[i-1] + side
    # Compute values of F at the vertices of the simplex
    for i in range(n+1): f[i] = F(x[i])
    # Main loop
    for k in range(500):
        # Find highest and lowest vertices
        iLo = np.argmin(f)
        iHi = np.argmax(f)
        # Compute the move vector d
        d = (-(n+1)*x[iHi] + np.sum(x,axis=0))/n
        # Check for convergence
        if math.sqrt(np.dot(d,d)/n) < tol: return x[iLo]
        # Try reflection
        xNew = x[iHi] + 2.0*d
        fNew = F(xNew)
        if fNew <= f[iLo]: # Accept reflection
            x[iHi] = xNew
            f[iHi] = fNew
            # Try expanding the reflection
            xNew = x[iHi] + d
            fNew = F(xNew)
            if fNew <= f[iLo]: # Accept expansion
                x[iHi] = xNew
                f[iHi] = fNew
        else:
        # Try reflection again
            if fNew <= f[iHi]: # Accept reflection
                x[iHi] = xNew
                f[iHi] = fNew
            else:
            # Try contraction
                xNew = x[iHi] + 0.5*d
                fNew = F(xNew)
                if fNew <= f[iHi]: # Accept contraction
                    x[iHi] = xNew
                    f[iHi] = fNew
                else:
                # Use shrinkage
                    for i in range(len(x)):
                        if i != iLo:
                            x[i] = (x[i] - x[iLo])*0.5
                            f[i] = F(x[i])
    print("Too many iterations in downhill")
    return x[iLo]


L1 = 1.2
L2 = 1.5
L3 = 1.0
B = 3.5
H = 0.0
W1 = 20000
W2 = 30000

def F(theta):
    lam1 = 100000.0
    lam2 = 100000.0
    c1 = min(0.0, L1*np.cos(theta[0]) + L2*np.cos(theta[1]) + L3*np.cos(theta[2]) - B)
    c2 = min(0.0, L1*np.sin(theta[0]) + L2*np.sin(theta[1]) + L3*np.sin(theta[2]) - H)
    return -W1*L1*np.sin(theta[0]) - W2*(L1*np.sin(theta[0]) + L2*np.sin(theta[1])) + lam1*c1**2 + lam2*c2**2

xStart = np.array([1.0, 1.0, 1.0])
x = downhill(F, xStart, 0.01)
print("(theta1, theta2, theta3) = ({:.6f}, {:.6f}, {:.6f})".format(x[0], x[1], x[2]))
