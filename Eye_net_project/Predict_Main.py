'''
Created on 2019年9月19日

@author: juicemilk
'''
import tensorflow as tf
from Eye_net_project.Eye_net.lib.Data_Process import data_process
from Eye_net_project.Eye_net.lib.Predict import predict

"""
function declaration:

predict_main: the function of predict
“网络预测函数”
"""
def predict_main():
    train_data_file='./data/final_data/train_data/train_data.npy'   #训练数据集的路径
    test_data_file='./data/final_data/test_data/test_data.npy'      #测试数据集的路径
    max_dict_file='./data/final_data/max.npy'                       #最大数据点集的路径
    min_dict_file='./data/final_data/min.npy'                       #最小数据点集的路径
    
    use_type='Height'                                 # 测试类型
    models_save_path='./models/'+use_type+'_net'     # 模型路径
    use_pie=False                                    # 是否使用Pie图表统计测试数据结果分布，需要安装pyecharts模块才能使用。默认不使用
                                                        
    
    _,test_data,data_max,data_min=data_process(train_data_file,
                                                test_data_file,
                                                max_dict_file,
                                                min_dict_file
                                                        ) 
    predict(test_data, data_max, data_min, use_type, models_save_path,use_pie)
    
    
    
def main(argv=None):
    predict_main()
if __name__=='__main__':
    tf.app.run()
