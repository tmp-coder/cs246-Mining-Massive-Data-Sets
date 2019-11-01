import re
import pandas as pd
import numpy as np

from recommender import Recommender

DATA_TRAIN_PATH = 'ratings.train.txt'
VALID_PATH = 'ratings.val.txt'
train = pd.read_csv(DATA_TRAIN_PATH,'\t',header=None)
test = pd.read_csv(VALID_PATH,'\t',header=None)

assert train.shape[0] == 90000

MAX_UID,MAX_MID,_= train.max()
MIN_UID,MIN_MID,_ = train.min()

train = train.values
test = test.values
# preprocess
if MIN_UID == 1:
    train[:,0] -= 1
    test[:,0] -= 1
    MAX_UID -=1

if MIN_MID == 1:
    train[:,1] -=1
    test[:,1] -=1
    MIN_UID -=1

# parameters

k = 20
learning_rate = 0.01
reg_para = 0.1
MAX_INIT = 100
# training

rcd = Recommender((MAX_MID+1,MAX_UID+1),k,learning_rate,reg_para)

loss = rcd.train(train,MAX_INIT)

# prediction

pred = rcd.recover()

error = 0

for uid, mid,rate in test:
    error += (rate - pred[mid][uid]) ** 2

error /=  test.shape[0]

print('rMSE : {}'.format(np.sqrt(error)))

import matplotlib.pyplot as plt

plt.plot(loss)
plt.show()

"""
final answer: 0.91
"""