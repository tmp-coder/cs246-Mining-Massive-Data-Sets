import numpy as np

def L2(a:np.array,b:np.array):
    d = a -b
    return d.dot(d)

def mahaten(a:np.array,b:np.array):
    d = a -b
    return np.sum(np.abs(d))
    