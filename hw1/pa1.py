import os
import sys
import shutil
from itertools import chain
from collections import Counter


import pyspark as psk


FILE_IN = os.path.join('data','soc-LiveJournal1Adj.txt')
FILE_OUT = os.path.join('out','pa1')

if os.path.isdir(FILE_OUT):
    shutil.rmtree(FILE_OUT)

conf = psk.SparkConf()
sc = psk.SparkContext(conf=conf)

def constructGraph(line):
    u,fs = line.split('\t')
    return (int(u),set(map(int,filter(lambda x: len(x)>0,fs.split(',')))))


socLines = sc.textFile(FILE_IN)
soc = socLines.map(constructGraph)
socG = soc.collectAsMap()


def getRecoms(e):
    u,vs = e
    deg1 = filter(lambda v: u in socG.get(v,{}),vs)
    def getFriendsFrom(x):
        return filter(
            lambda e: e != u and(e not in socG[u]) and (x in socG[e]), # not a dirct friend of u, and matual connect to x
            socG[x]
        )
    deg2 = chain.from_iterable(
        map(
            getFriendsFrom,
            deg1
        )
    )
    x = Counter(deg2)
    return (u,list(map(
        lambda e: e[0],
        x.most_common()[:10]
    )))


soc.map(getRecoms).map(
    lambda e: "{}\t{}".format(e[0],",".join(map(str,e[1])))
).coalesce(
    1,True
).saveAsTextFile(FILE_OUT)

