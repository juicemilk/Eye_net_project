'''
Created on 2019年6月4日

@author: juicemilk
'''
import xlrd
import numpy as np
import os
"""
function declaration:

read_tx: read tx data from original file
“从原始文件中读取tx维度数据”
"""
"""
parameter declaration:

input_file_name: the path of original file
“原始文件的路径”
output_file_name: the save path of tx data
“读取后数据的保存路径”
"""
def read_tx(input_file_name,output_file_name):
    print("正在读取TX数据......")
    try:
        f=xlrd.open_workbook(input_file_name)
    except Exception:
        print('错误！无法读取原始Tx文件数据，请检查原始文件是否存在或检查输入文件路径是否正确')
        os._exit(0)
    try:
        if not os.path.isdir(output_file_name.rsplit('/', 1)[0]):
            os.makedirs(output_file_name.rsplit('/', 1)[0])
    except Exception:
        print('保存数据的目录不存在，且无法创建目录。')
        os._exit(0)
    Tx_data=[]
    sheet1=f.sheet_by_index(0)
    for i in range(sheet1.nrows):
        if isinstance(sheet1.cell(i,0).value, str):
            continue
        Tx_data.append(sheet1.row_values(i)[1:4])
    Tx_data=np.array(Tx_data)
    print("正在保存临时TX数据文件......")
    np.savetxt(output_file_name,Tx_data,delimiter=',')
    print("保存完成")