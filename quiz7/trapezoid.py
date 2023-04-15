import numpy as np 

def f(x, dx=1e-6):
    if x == 2:
        x = x-dx # prevent from divided by 0
    if x == -2:
        x = x+dx
    return 2*np.sqrt((16-3*x**2)/(16-4*x**2))

def trapezoid(f,a,b,Iold,k):
    if k == 1:Inew = (f(a) + f(b))*(b - a)/2.0
    else:
        n = 2**(k -2) # Number of new points
        h = (b - a)/n # Spacing of new points
        x = a + h/2.0
        sum = 0.0
        for i in range(n):
            sum = sum + f(x)
            x = x + h
        Inew = (Iold + h*sum)/2.0
    return Inew

tol = 1e-5
I = 0
a = -2
b = 2
k = 0

for k in range(1, 1000):
    Inew = trapezoid(f, a, b, I, k)
    print('k = {}, Inew = {}, difference = {}'.format(k, Inew, abs(Inew - I)))
    if abs(Inew - I) < tol: break
    I = Inew

print('iterat {k} times, integral result is {I}'.format(k = k, I = I))
