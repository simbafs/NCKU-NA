import numpy as np 
import matplotlib.pyplot as plt

def integrate(F,x,y,xStop,h,tol=1.0e-6):
    a1 = 0.2; a2 = 0.3; a3 = 0.8; a4 = 8/9; a5 = 1.0
    a6 = 1.0
    c0 = 35/384; c2 = 500/1113; c3 = 125/192
    c4 = -2187/6784; c5 = 11/84
    d0 = 5179/57600; d2 = 7571/16695; d3 = 393/640
    d4 = -92097/339200; d5 = 187/2100; d6 = 1/40
    b10 = 0.2
    b20 = 0.075; b21 = 0.225
    b30 = 44/45; b31 = -56/15; b32 = 32/9
    b40 = 19372/6561; b41 = -25360/2187; b42 = 64448/6561
    b43 = -212/729
    b50 = 9017/3168; b51 =-355/33; b52 = 46732/5247
    b53 = 49/176; b54 = -5103/18656
    b60 = 35/384; b62 = 500/1113; b63 = 125/192;
    b64 = -2187/6784; b65 = 11/84
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    stopper = 0 # Integration stopper(0 = off, 1 = on)
    k0 = h*F(x,y)
    for i in range(500):
        k1 = h*F(x + a1*h, y + b10*k0)
        k2 = h*F(x + a2*h, y + b20*k0 + b21*k1)
        k3 = h*F(x + a3*h, y + b30*k0 + b31*k1 + b32*k2)
        k4 = h*F(x + a4*h, y + b40*k0 + b41*k1 + b42*k2 + b43*k3)
        k5 = h*F(x + a5*h, y + b50*k0 + b51*k1 + b52*k2 + b53*k3 + b54*k4)
        k6 = h*F(x + a6*h, y + b60*k0 + b62*k2 + b63*k3 + b64*k4 + b65*k5)
        dy = c0*k0 + c2*k2 + c3*k3 + c4*k4 + c5*k5
        E = (c0 - d0)*k0 + (c2 - d2)*k2 + (c3 - d3)*k3 + (c4 - d4)*k4 + (c5 - d5)*k5 - d6*k6
        e = np.sqrt(np.sum(E**2)/len(y))
        hNext = 0.9*h*(tol/e)**0.2
        # Accept integration step if error e is within tolerance
        if e <= tol:
            y = y + dy
            x = x + h
            X.append(x)
            Y.append(y)
            if stopper == 1: break # Reached end of x-range
            if abs(hNext) > 10.0*abs(h): hNext = 10.0*h
            # Check if next step is the last one; if so, adjust h
            if (h > 0.0) == ((x + hNext) >= xStop):
                hNext = xStop - x
            # stopper = 1
            k0 = k6*hNext/h
        else:
            if abs(hNext) < 0.1*abs(h): hNext = 0.1*h
            k0 = k0*hNext/h
        h = hNext
    return np.array(X),np.array(Y)

c = 5 
k = 120 
L = 0.2 
m= 1.0

def F(t, x):
    f = np.zeros(2)
    f[0] = x[1]
    f[1] = c/m/x[0]**2 - k/m*(x[0] - L)
    return f

t, X = integrate(F, 0.0, np.array([L, 0.0]), 1.0, 0.0025)
print()

xOld = np.array(X[0])
tol = 1e-9
period = 0

for i in range(1, len(X)):
    if t[i] > 0.2 and np.all(xOld - X[i] < tol):
        print('period is {:.6f}s'.format(t[i], i))
        period = t[i]
        break

plt.axline((0, 0), (0, 1), color='green')
plt.axline((period, 0), (period, 1), color='green')

plt.plot(t, X[:,0])
# plt.plot(t, X[:,1])
plt.grid(True)
plt.xlabel('t')
plt.ylabel('x')
plt.show()

# print(t)
# print(X)
