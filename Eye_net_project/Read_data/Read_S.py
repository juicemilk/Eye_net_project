'''
Created on 2019年6月4日

@author: juicemilk
'''
import numpy as np
import os
"""
function declaration:

read_s: read s data from original file
“从原始文件中读取s维度数据”
"""
"""
parameter declaration:

input_file_name: the path of original file
“原始文件的路径”
output_file_name: the save path of s data
“读取后数据的保存路径”
"""

def read_s(input_file_name,output_file_name):
    print("正在读取S数据......")
    try:
        step_data_p=np.loadtxt(input_file_name,dtype='float')
    except Exception:
        print('错误！无法读取原始S文件数据，请检查原始文件是否存在或检查输入文件路径是否正确')
        os._exit(0)
    step_data_p=step_data_p[1:len(step_data_p)]
    step_data=[]
    for i in range(np.shape(step_data_p)[1]):
        step_data.append(step_data_p[:,i])
    step_data=np.array(step_data)
    print("正在保存临时S数据文件......")
    np.savetxt(output_file_name,step_data,delimiter=',')
    print("保存完成")

