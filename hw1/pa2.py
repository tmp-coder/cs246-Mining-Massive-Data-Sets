"""
the simple implementation of Aprior Algorithm
author @zouzhitao
"""

from pyspark import SparkConf,SparkContext,RDD

import os, shutil
from queue import PriorityQueue
from Apriori import genByMenbership,gen_candidate_itemset
import utils
SUPPORT_TH = 100

FILE_IN = os.path.join('../data','brwosing.txt')
FILE_OUT = os.path.join('../out','AssociationRule')

if os.path.isdir(FILE_OUT):
    shutil.rmtree(FILE_OUT)

conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile(FILE_IN)
dataStream = lines.map(
    lambda line: list(set(filter(lambda e: len(e)>0,line.split(' ')))) # distinct
)
L1 = dataStream.flatMap(
    lambda x: [
        (e,1)
        for e in x
    ]
).reduceByKey(
    lambda n1,n2: n1 + n2
).filter(
    lambda x: x[1] >= SUPPORT_TH
).collectAsMap()

assert '' not in L1.keys()


def filter_candidate_set_by_sub_set(ck:set,K:int,L1Set:set):
    return dataStream.map(
        lambda lst : sorted(list(filter(lambda e: e in L1Set, lst))) # this trick can accelarate speed
    ).flatMap(
        lambda lst: [(e ,1) for e in utils.ksubset(K,lst) if e in ck]
    ).reduceByKey(
        lambda n1,n2: n1 + n2
    ).filter(
        lambda x: x[1] >= SUPPORT_TH
    ).collectAsMap()

def filter_candidate_set_by_membership(ck:set,K:int,L1Set:set):
    return dataStream.map(
        lambda lst :set(filter(lambda e: e in L1Set, lst))
    ).flatMap(
        lambda se : [
            (e,1)
            for e in ck
                if utils.membership_test(se,e)
        ]
    ).reduceByKey(
        lambda n1,n2 : n1 + n2
    ).filter(
        lambda x : x[1] >=SUPPORT_TH
    ).collectAsMap()

c2 = gen_candidate_itemset(L1.keys(),1)


L1key = set(L1.keys())

L2 = filter_candidate_set_by_sub_set(c2,2,L1key)

c3 = gen_candidate_itemset(L2.keys(),2)

L3 = filter_candidate_set_by_sub_set(c3,3,L1key)

support_dict = {**L1,**L2,**L3}

def confidence(L : dict, topK:int):

    Q = PriorityQueue()
    for k,v in L.items():
        candidates = utils.generatek_subset(k)
        for e in candidates:
            confid = v / support_dict[e[0]]
            if Q.qsize() <topK:
                Q.put((confid,e))
            else:
                ca = Q.get()
                if ca[0] < confid:
                    ca = (confid,e)
                Q.put(ca)
    return sorted(list(map(lambda e : (e[1],e[0]),Q.queue)),key=lambda x:-x[1])

conf2 = confidence(L2,15)
conf3 = confidence(L3,15)
print("top 15 rule of 2-freq itemset:")
for e in conf2:
    print("{} -> {} : {}".format(e[0][0],e[0][1],e[1]))

print("top 15 rule of 2-freq itemset:")
for e in conf3:
    print("{} -> {} : {}".format(e[0][0],e[0][1],e[1]))


sc.stop()
