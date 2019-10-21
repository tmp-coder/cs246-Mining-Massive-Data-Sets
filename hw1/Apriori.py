"""
simple implementation of Apriori algorithm
author: @zouzhitao
refrence:
    1. J. Han, M. Kamber, J. Pei. Data Mining : Concepts and Techniques. 3rd Edition. ch6
        P253. 
"""

from pyspark import RDD

from itertools import chain

import utils

def gen_candidate_itemset(lk : set,k:int):
    candidate_lst = []
    key_lst = list(lk)
    def item_join(e1,e2):
        if e1[:-1] == e2[:-1] and e1[-1] != e2[-1]:
            cd =  tuple(sorted(e1[:-1] + (e1[-1],e2[-1])))
            if not has_not_freq_subset(lk,cd,k):
                return cd
        return None

    if k == 1:
        candidate_lst = [
            (key_lst[i],key_lst[j]) if key_lst[i] < key_lst[j] else (key_lst[j],key_lst[i])
            for i in range(len(key_lst))
                for j in range(i+1,len(key_lst))
        ]
    else:
        candidate_lst = [
            item_join(key_lst[i],key_lst[j])
            for i in range(len(key_lst))
                for j in range(i+1,len(key_lst))
        ]
        candidate_lst = filter(lambda x: x is not None, candidate_lst)

    return set(candidate_lst)
                    


def has_not_freq_subset(lk : set,candidate:tuple,k:int):
    """
    test if all of the k subset of candidate is in LK 
    Args:
        candidate : a sorted tuple of k items by comperitor of item
    Return:
        True if has a k subset of candidate not in LK, ortherise False
    """

    assert len(candidate) == k + 1 and (type(candidate) is tuple), "parameter error"

    for i in range(k + 1): # O(k ^2)
        subset = candidate[:i] + candidate[i+1:]
        if subset not in lk:
            return True

    return False

def genByMenbership(ck:set,lst:list):
    def membership_test(x : set,c:tuple):
        for e in c:
            if e not in x:
                return False
        return True
    x = set(lst)
    return [
        e
        for e in ck
            if membership_test(x,e)
    ]

    

