import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kmeans import kmeans
from distance import L2,mahaten


DATA_IN = 'data.txt'

INIT_1 = 'c1.txt'
INIT_2 = 'c2.txt'

data = pd.read_csv(DATA_IN,' ',header=None).values
c1 = pd.read_csv(INIT_1,' ',header=None).values
c2 = pd.read_csv(INIT_2,' ',header=None).values

# parameters

MAX_ITER = 100

dis_fun = mahaten


cost1,center1 = kmeans(data,20,c1,dis_fun)
cost2,center2 = kmeans(data,20,c2,dis_fun)

plt.plot(cost1,label='c1')
plt.plot(cost2,label='c2')
plt.legend()
plt.show()
