import numpy as np 
import matplotlib.pyplot as plt

g = 9.80665
CdM = 0.2028/80

def F(x, y):
    F = np.zeros(2)
    F[0] = y[1]
    F[1] = g - CdM*y[1]**2 
    return F 

def integrate(F,x,y,Stop,h):
    def run_kut4(F,x,y,h):
        K0 = h*F(x,y)
        K1 = h*F(x + h/2.0, y + K0/2.0)
        K2 = h*F(x + h/2.0, y + K1/2.0)
        K3 = h*F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while not Stop(x, y):
        # h = min(h,xStop - x)
        y = y + run_kut4(F,x,y,h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)

x = 0
y = np.array([0.0, 0.0])

X, Y = integrate(F, x, y, Stop=lambda x, y: y[0] >= 5000, h=0.2)

print('y({x}) = {y}, y\'({x}) = {yy}'.format(x = X[-1], y = Y[-1][0], yy = Y[-1][1]))

plt.plot(X, Y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(('y', 'y\''))
plt.show()
