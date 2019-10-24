import numpy as np


from distance import L2,mahaten


def kmeans(data:np.array,MAX_ITER:int, init_point:np.array,dis_fun=L2):
    
    cost = np.zeros(MAX_ITER)
    for i in range(MAX_ITER):
        cluster = np.zeros_like(init_point)
        cnt = np.zeros(init_point.shape[0])

        for j in range(data.shape[0]):
            dist = np.array([dis_fun(data[j],e) for e in init_point])
            center = np.argmin(dist)
            cost[i] += dist[center]
            cluster[center]+= data[j]
            cnt[center] += 1
        
        init_point = cluster / cnt.reshape(cnt.shape[0],-1)

    return cost,init_point

    