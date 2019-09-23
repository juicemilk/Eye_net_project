'''
Created on 2019年4月12日

@author: juicemilk
'''
import numpy as np
import random

"""
function declaration:
data_shuffle：shuffle the training data
“打乱数据顺序”
"""

"""
parameter declaration:
train_data: the order training data
“原始训练数据”
use_type: the type of training,including Height and Width
“使用训练数据的网络类型”
"""

def data_shuffle(train_data,use_type):
    train_data_len=list(range(len(train_data.get(use_type))))
    
    random.shuffle(train_data_len)
    
    keylist=list(train_data.keys())
    
    train_shuffer_data={}
    
    key_num=np.shape(keylist)[0]
    for i in range(key_num):
        train_shuffer_data[keylist[i]]=[]

    for i in train_data_len:
        for j in range(len(keylist)):
            train_shuffer_data.get(keylist[j]).append(train_data.get(keylist[j])[i]) 
    
    return train_shuffer_data