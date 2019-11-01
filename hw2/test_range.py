import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time

DATA_IN = 'data.txt'

data = pd.read_csv(DATA_IN,' ',header=None).values

def time_measure(fun):
    start = time.time()
    ans = fun()
    return time.time() - start,ans

def based_index():
    s= np.zeros(data.shape[1])
    for i in range(len(data)):
        s += data[i]
    return s

def based_iterater():
    s = np.zeros(data.shape[1])
    for e in data:
        s += e
    return s


if __name__ == "__main__":
    
    t1,s1 = time_measure(based_index)
    t2,s2 = time_measure(based_iterater)
    assert np.sum(np.abs(s1 - s2)) < 1e-6

    print('index based : {}, itrater based: {}'.format(t1,t2))