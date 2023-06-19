import numpy as np

def buildH(n):
    H = np.zeros((n,n))
    
    H[0, 0:3] = [7, -4, 1]
    H[1, 0:4] = [-4, 6, -4, 1]
    H[n-2, n-4:n] = [1, -4, 5, -2]
    H[n-1, n-3:n] = [1, -2, 1]

    return H 

print('H:\n', buildH(6))
print('P:\n', np.identity(6))
