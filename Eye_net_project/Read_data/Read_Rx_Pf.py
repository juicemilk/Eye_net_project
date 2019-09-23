'''
Created on 2019年6月4日

@author: juicemilk
'''
import xlrd
import numpy as np
import os
"""
function declaration:

read_rx_pf: read rx_pf data from original file
“从原始文件中读取rx-pf维度数据”
"""
"""
parameter declaration:

input_file_name: the path of original file
“原始文件的路径”
output_file_name: the save path of rx_pf data
“读取后数据的保存路径”
"""

def read_rx_pf(input_file_name,output_file_name):
    print("正在读取RX_PF数据......")
    try:
        f=xlrd.open_workbook(input_file_name)
    except Exception:
        print('错误！无法读取原始Rx_Pf文件数据，请检查原始文件是否存在或检查输入文件路径是否正确')
        os._exit(0)
    Rx_PF_data=[]
    sheet1=f.sheet_by_index(0)
    for i in range(sheet1.nrows):
        if isinstance(sheet1.cell(i,0).value, str):
            continue
        Rx_PF_data.append(sheet1.row_values(i)[1:2])
    Rx_PF_data=np.array(Rx_PF_data)
    print("正在保存临时RX_PF数据文件......")
    np.savetxt(output_file_name,Rx_PF_data,delimiter=',')
    print("保存完成")