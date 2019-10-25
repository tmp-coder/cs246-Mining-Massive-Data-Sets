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

def kmeanspp(data:np.array,k:int,MAX_ITER:int,dis_fun=L2):
    """
    kmeans++: https://en.wikipedia.org/wiki/K-means%2B%2B
    """
    
    init_point = np.zeros((k,data.shape[1]))
    init_point[0] = data[np.random.choice(data.shape[0])]
    for i in range(1,k):
        p = [np.min([dis_fun(point,cluster)for cluster in init_point]) for point in data]
        assert len(p) == data.shape[0]
        p/=sum(p)
        next_cluster = np.random.choice(len(p),1,p=p)
        init_point[i] = data[next_cluster]

        
    assert init_point.shape[0] == k and init_point.shape[1] == data.shape[1],"{}".format(init_point.shape) 
    return kmeans(data,MAX_ITER,init_point,dis_fun)    
    