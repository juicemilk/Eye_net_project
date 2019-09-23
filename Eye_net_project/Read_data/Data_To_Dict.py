'''
Created on 2019年6月5日

@author: juicemilk
'''
import numpy as np
import math
import os

"""
function declaration:

data_to_dict: Combining data from different dimensions into dictionary form,and save data
“将不同维度的数据组合成字典形式，保存数据，并根据训练数据占比将数据划分为训练集和测试集”
"""

"""
parameter declaration:

Tx_file: the path of tx file
“tx文件的路径”
S_file: the path of tx file
“s文件的路径”
Rx_PF_file: the path of tx file
“rx_pf文件的路径”
Rx_ZERO_file: the path of tx file
“rx_zero文件的路径”
Label_file: the path of tx file
“label文件的路径”
Data_file: the save path of dictionary data(.txt)
“组合后的字典数据文件的保存路径（txt类型，方便查看）”
Data_dict_file: the save path of dictionary data(.npy)
“组合后的字典数据文件的保存路径”
train_rate: the rate of train data
“训练数据集占比”
train_file: the save path of train data(.txt)
“划分的训练数据集文件的保存路径（txt类型，方便查看）”
train_data_file: the save path of train data(.npy)
“划分的训练数据集文件的保存路径”
test_file: the save path of test data(.txt)
“划分的测试数据集文件的保存路径（txt类型，方便查看）”
test_data_file: the save path of test data(.npy)
“划分的测试数据集文件的保存路径”
max_file: the save path of max point(.txt)
“数据中每个维度最大值数据文件的保存路径（txt类型，方便查看）”
max_dict_file: the save path of max point(.npy)
“数据中每个维度最大值数据文件的保存路径”
min_file: the save path of min point(.txt)
“数据中每个维度最小值数据文件的保存路径（txt类型，方便查看）”
min_dict_file: the save path of min point(.npy)
“数据中每个维度最小值数据文件的保存路径”

"""

def data_to_dict(Tx_file,S_file,Rx_PF_file,Rx_ZERO_file,Label_file,Data_file,Data_dict_file,train_rate,train_file,train_data_file,test_file,test_data_file,max_file,max_dict_file,min_file,min_dict_file):
    print("正在将数据转为字典格式......")
    try:
        if not os.path.isdir(train_file.rsplit('/', 1)[0]):
            os.makedirs(train_file.rsplit('/', 1)[0])
        if not os.path.isdir(test_file.rsplit('/', 1)[0]):
            os.makedirs(test_file.rsplit('/', 1)[0])
    except Exception:
        print('保存数据的目录不存在，且无法创建目录。')
        os._exit(0)
    file=open(Data_file, 'w') 
    Tx=np.loadtxt(Tx_file,dtype='float',delimiter=',')  
    Rx_ZERO=np.loadtxt(Rx_ZERO_file,dtype='float',delimiter=',')
    Rx_ZERO=Rx_ZERO.reshape(len(Rx_ZERO),1)  
    Rx_PF=np.loadtxt(Rx_PF_file,dtype='float',delimiter=',')
    Rx_PF=Rx_PF.reshape(len(Rx_PF),1) 
    S=np.loadtxt(S_file,dtype='float',delimiter=',')
    Label=np.loadtxt(Label_file,dtype='float',delimiter=',')  
    
    sample_num=len(Label)
    
    data_all_dict={}
    keyword=['Tx','Rx','S','Height','Width']
    for i in range(len(keyword)):
        data_all_dict[keyword[i]]=[]
       
    for i in range(sample_num):
        Tx_index=Label[i][0]
        Rx_ZERO_index=Label[i][1]
        Rx_PF_index=Label[i][2]
        S_index=Label[i][3]
            
        data_all_dict.get('Tx').append(Tx[int(Tx_index)].tolist())
        data_all_dict.get('Rx').append(Rx_ZERO[int(Rx_ZERO_index)].tolist()+Rx_PF[int(Rx_PF_index)].tolist())
        data_all_dict.get('S').append(S[int(S_index-1)].tolist())
        data_all_dict.get('Height').append(Label[i,4:5].tolist())
        data_all_dict.get('Width').append(Label[i,5:6].tolist())
    
    for k in data_all_dict.keys():
        file.write(str(k)+'                ')
    file.write('\n')
    for i in range(sample_num):
        for v in data_all_dict.values():
            file.write(str(v[i])+'                ')
        file.write('\n')       
    file.close()
    print("正在保存数据字典文件......")
    np.save(Data_dict_file,data_all_dict)   
    print("保存完成") 
    
    print('正在划分数据集.......')
    file1=open(test_file, 'w') 
    file2=open(train_file, 'w') 
    file3=open(max_file, 'w') 
    file4=open(min_file, 'w')
    data=data_all_dict
    keylist=list(data.keys())
    sample_num=np.shape(data.get(keylist[0]))[0]
    random_num=np.random.choice(sample_num, math.floor(sample_num*train_rate),False).tolist()
    train_data={}
    test_data={}
    max_dict={}
    min_dict={}
    key_num=np.shape(keylist)[0]
    for i in range(key_num):
        train_data[keylist[i]]=[]
        test_data[keylist[i]]=[]
        max_dict[keylist[i]]=[]
        min_dict[keylist[i]]=[]
        
    for i in range(np.shape(keylist)[0]):
        if (keylist[i]=='Rx')|(keylist[i]=='Tx'):
            max_val=np.max(data.get(keylist[i]),0)
            min_val=np.min(data.get(keylist[i]),0)
            max_dict[keylist[i]]=max_val.tolist()
            min_dict[keylist[i]]=min_val.tolist()
        else:
            max_val=np.max(data.get(keylist[i]))
            min_val=np.min(data.get(keylist[i]))
            max_dict[keylist[i]].append(max_val)
            min_dict[keylist[i]].append(min_val)
        
    for i in range(sample_num):
        if i in random_num:
            for j in range(np.shape(keylist)[0]):
                train_data.get(keylist[j]).append(data.get(keylist[j])[i])
        else:
            for j in range(np.shape(keylist)[0]):
                test_data.get(keylist[j]).append(data.get(keylist[j])[i])
    print('划分完成，正在保存数据.......')
    for k in test_data.keys():
        file1.write(str(k)+'                ')
    file1.write('\n')
    for i in range(sample_num-len(random_num)):
        for v in test_data.values():
            file1.write(str(v[i])+'                ')
        file1.write('\n')
        
    file1.close()
    
    for k in train_data.keys():
        file2.write(str(k)+'                ')
    file2.write('\n')
    for i in range(len(random_num)):
        for v in train_data.values():
            file2.write(str(v[i])+'                ')
        file2.write('\n')
        
    file2.close()
    
    for k in max_dict.keys():
        file3.write(str(k)+'                ')
    file3.write('\n')
    for v in max_dict.values():
        file3.write(str(v)+'                ')
    file3.write('\n')
        
    file3.close()
    
    for k in min_dict.keys():
        file4.write(str(k)+'                ')
    file4.write('\n')
    for v in min_dict.values():
        file4.write(str(v)+'                ')
    file4.write('\n')
        
    file4.close()
    np.save(train_data_file, train_data)
    np.save(test_data_file, test_data)
    np.save(max_dict_file,max_dict)
    np.save(min_dict_file,min_dict)
    print("保存完成")