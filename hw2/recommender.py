"""
in memory model
author: zouzhitao
"""

import numpy as np


class Recommender(object):
    def __init__(self,shape:tuple,latent_factor:int,learning_rate=0.01,reg=0.01):
        """
        Args:
            shape : tuple
            (m,n): m for items, n for users
            latent_factor: number of latent factor
        """
        MAX_INIT = np.sqrt(5/ latent_factor)
        self.Q = np.random.random((shape[0],latent_factor)) * MAX_INIT
        self.P = np.random.random((shape[1],latent_factor)) * MAX_INIT
        self.learning_rate = learning_rate
        self.reg = reg

    def error_dev_item(self, x:int,i:int,rate:float):
        """
        r_xi : user x, item i
        """
        return (rate - self.Q[i].dot(self.P[x]))

    def dev_item(self, x:int,i:int,rate:float):
        
        edi = self.error_dev_item(x,i,rate)
        delte_q = edi * self.P[x] -self.reg * self.Q[i]
        delte_p = edi * self.Q[i] - self.reg * self.P[x]

        return (delte_q,delte_p)
    
    def update(self, data,CAOMPUTE_LOSS=True):
        
        loss = -1
        loss =0
        i =0
        for uid,mid,rate in data:
            i += 1
            dq,dp = self.dev_item(uid,mid,rate)
            for e in dp:
                assert e is not np.nan,'uid :{},mid:{},rate:{}'.format(uid,mid,rate)
            for e in dq:
                assert e is not np.nan,'uid :{},mid:{},rate:{}'.format(uid,mid,rate)
                    
            self.Q[mid] += self.learning_rate * dq
            self.P[uid] += self.learning_rate * dp
            for e in self.Q[mid]:
                assert e is not np.nan,'uid :{},mid:{},rate:{}'.format(uid,mid,rate)
            for e in self.P[uid]:
                assert e is not np.nan,'uid :{},mid:{},rate:{}'.format(uid,mid,rate)

            if CAOMPUTE_LOSS :
                # loss = 0
                # print('----------- item loss ---------')
                # for uid,mid,rate in data:
                item_mse = (self.Q[mid].dot(self.P[uid]) - rate) ** 2
                # if i % 100 == 0:
                #     print('i : {}, uid : {}, mid: {} , item_mse: {}'.format(i,uid,mid,item_mse))
                loss += item_mse
        loss += self.reg * np.sum(np.array(e.dot(e)) for e in self.P)
        loss += self.reg * np.sum(np.array(e.dot(e)) for e in self.Q)
        print('loss : {}'.format(loss))

        return loss
    
    def train(self,data,MAX_INIT=40, save_loss=True):
        loss = np.zeros(MAX_INIT)

        for i in range(MAX_INIT):
            loss[i] = self.update(data)
        
        return loss
    
    def recover(self, ):
        return self.Q.dot(self.P.T)