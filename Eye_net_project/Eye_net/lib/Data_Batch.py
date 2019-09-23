'''
Created on 2019年4月12日

@author: juicemilk
'''
import numpy as np

"""
function declaration:

data_batch: divide the shuffle data to batch data
"将数据按照指定数目划分批次"
"""

"""
parameter declaration:

train_shuffle_data: the input data
“需要划分的数据”
batch_size: the size of batch data
“每个批次数据的数目大小”
k: the order number of batch data
“正在划分批次数据的序号”
"""
def data_batch(train_shuffle_data,batch_size,k):
  
    train_batch_data={}
    keylist=list(train_shuffle_data.keys())
    
    key_num=np.shape(keylist)[0]
    
    for i in range(key_num):
        train_batch_data[keylist[i]]=train_shuffle_data.get(keylist[i])[k*batch_size:(k+1)*batch_size]
    
    return train_batch_data