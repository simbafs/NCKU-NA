import numpy as np 
import random

def laplace_pi(n):
    sum = 0
    sigma = 1/np.sqrt(2)
    for i in range(n):
        x = random.random()
        y = random.random()
        if x**2 + y**2 <= 1:
            sum += 1
    return sum/n*4

N = [1000, 10000, 100000]
for n in N:
    pi = [laplace_pi(n) for i in range(1000)]
    mean = np.mean(pi)
    std = np.std(pi)
    print("N = {},\tmean pi = {:.6f},\tstd pi = {:.6f},\terror = {:.6%}".format(n, mean, std, abs(mean - np.pi)/np.pi))
