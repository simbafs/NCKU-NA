import math
from typing import Callable, Tuple


def atan(x: float, stop: Callable[[float, int], bool]) -> Tuple[float, int]:
    result: float = 0.0
    n = 0

    while not stop(result, n):
        result += (-1)**n * x**(2*n+1) / (2*n+1)
        n += 1

    return result, n


pi1 = atan(1, lambda result, n: abs(result*4 - math.pi)/math.pi < 1e-4 or n > 1e5) * 4
pi2 = atan(1/math.sqrt(3), lambda result, n: abs(result*6 - math.pi)/math.pi < 1e-4 or n > 1e5) * 6

print('pi =', pi1[0], ', n =', pi1[1], ', error =', abs(pi1[0] - math.pi)/math.pi)
print('pi =', pi2[0], ', n =', pi2[1], ', error =', abs(pi2[0] - math.pi)/math.pi)
