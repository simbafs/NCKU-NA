import numpy as np 

# beginning of copy from quiz3

def conjGrad(A, x, b, tol = 1e-9):
    n = len(b)
    r = b - np.dot(A, x)
    s = r.copy()
    i = 0

    for i in range(n):
        u = np.dot(A, s)
        alpha = np.dot(s, r)/np.dot(s, u)
        x = x + alpha*s
        r = b - np.dot(A, x)
        if(np.sqrt(np.dot(r, r))) < tol: break
        else: 
            beta = -np.dot(r, u)/np.dot(s, u)
            s = r + beta*s
        print(x)

    return x, i

# end of copy from quiz3

A = np.array([[-3.0, 1.0, 0.0, 0.0, 0.0],
              [3.0, -6.0, 3.0, 0.0, 0.0],
              [0.0, 3.0, -6.0, 3.0, 0.0],
              [0.0, 0.0, 3.0, -6.0, 3.0],
              [0.0, 0.0, 0.0, 3.0, -5.0]])
x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
b = np.array([-80.0, 0.0, 0.0, 60.0, 0.0])

def round(arr):
    result = []
    for i in range(len(arr)):
        result.append('{:.4f}'.format(arr[i]))
    return result

x, i = conjGrad(A, x, b)
print('x = {x}, i = {i}'.format(x = round(x), i = i))
print('Ax = {Ax}, b = {b}'.format(Ax = np.dot(A, x), b = b))
print('difference = {difference}'.format(difference = np.dot(A, x)-b))
