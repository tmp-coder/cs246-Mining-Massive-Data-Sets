"""
solution for p2

naive PageRank in spark

hoposithes:
    1. cannot store all edges in memory
    2. can store node in memory

data: 
    1. duplicated edge
    2. no dead-end
print: top5 pageRank,lowest 5 pagerank
"""

from pyspark import SparkConf,SparkContext
from data_clean import DEG_IN,DEG_OUT

import numpy as np

conf = SparkConf()
sc = SparkContext(conf=conf)

# params 

MAX_ITERS = 40

N_NODES = 1000

no_degs = np.zeros(N_NODES)

no_degs=sc.pickleFile(DEG_OUT).map(lambda x : (x[0],x[1][0])).collectAsMap()

assert len(no_degs) == N_NODES


cur = 1
r_dict= [dict([(i,1/N_NODES) for i in range(N_NODES)]),dict()]

for i in range(MAX_ITERS):
    r_dict[cur]=sc.pickleFile(DEG_IN).mapValues(
        lambda in_nodes: sum(list(map(lambda e : r_dict[cur^1][e]/no_degs[e],in_nodes)))
    ).collectAsMap()
    cur ^=1

ans = list(sorted(list(r_dict[cur].items()),key=lambda x : -x[1]))

sum_rk = sum(map(lambda e : e[1],ans),0)


print('sum page rank : {}'.format(sum_rk))

for e in ans:
    print("{},{}".format(e[0],e[1]))