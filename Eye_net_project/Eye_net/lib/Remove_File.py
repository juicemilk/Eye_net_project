'''
Created on 2019年9月21日

@author: juicemilk
'''

import os
"""
function declaration:

remove_file: Delete all files in the specified folder
“删除指定文件夹内的所有文件”
"""
"""
parameter declaration:
path: the root path of specified folder
“指定文件夹的根路径”
"""

def remove_file(path):
    for root,dirs,files in os.walk(path,topdown=False):
        for name in files:
            os.remove(os.path.join(root,name))