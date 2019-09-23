'''
Created on 2019年6月4日

@author: juicemilk
'''
import xlrd
import numpy as np
import os
"""
function declaration:

read_label: read label data from original file
“从原始文件中读取数据标签（眼宽和眼高）”
"""
"""
parameter declaration:

input_file_name: the path of original file
“原始文件的路径”
output_file_name: the save path of label data
“读取后数据的保存路径”
"""

def read_label(input_file_name,output_file_name):
    print("正在读取Label数据......")
    try:
        f=xlrd.open_workbook(input_file_name)
    except Exception:
        print('错误！无法读取原始RLabel文件数据，请检查原始文件是否存在或检查输入文件路径是否正确')
        os._exit(0)
    label_data=[]
    sheet1=f.sheet_by_index(0)
    for i in range(sheet1.nrows):
        if isinstance(sheet1.cell(i,0).value, str):
            continue
        label_data.append(sheet1.row_values(i)[1:7])
    label_data=np.array(label_data)
    print("正在保存临时Label数据文件......")
    np.savetxt(output_file_name,label_data,delimiter=',')
    print("保存完成")
    
