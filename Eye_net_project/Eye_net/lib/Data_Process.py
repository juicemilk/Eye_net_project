'''
Created on 2019年6月5日

@author: juicemilk
'''
import numpy as np
import os

"""
function declaration:

data_process: Normalized data
“归一化数据”
"""
"""
parameter declaration:

train_data_file: the path of train data file
“训练数据的文件路径”
test_data_file: the path of test data file
“测试数据的文件路径”
max_dict_file: the path of max point file
“数据中每个维度的最大值”
min_dict_file: the path of min point file
“数据中每个维度的最小值”
"""

def data_process(train_data_file,test_data_file,max_dict_file,min_dict_file):
    print("正在归一化数据......")
    try:
        train_data=np.load(train_data_file,allow_pickle=True).tolist() 
        test_data=np.load(test_data_file,allow_pickle=True).tolist() 
        max_data=np.load(max_dict_file,allow_pickle=True).tolist() 
        min_data=np.load(min_dict_file,allow_pickle=True).tolist() 
    except Exception:
        print('错误！请检查测试集/训练集数据文件是否存在，若不存在，请先运行数据提取程序。')
        os._exit(0)
    keylist=list(max_data)
    
    train_data_p={}
    test_data_p={}

    key_num=np.shape(keylist)[0]
    for i in range(key_num):
        train_data_p[keylist[i]]=[]
        test_data_p[keylist[i]]=[]
        
    for i in range(key_num):
            max_val=np.array(max_data.get(keylist[i]))
            min_val=np.array(min_data.get(keylist[i]))
            train_data_p[keylist[i]]=((train_data.get(keylist[i])-min_val)/(max_val-min_val)).tolist()
            test_data_p[keylist[i]]=((test_data.get(keylist[i])-min_val)/(max_val-min_val)).tolist()

    print("归一化完成")
    return train_data_p,test_data_p,max_data,min_data