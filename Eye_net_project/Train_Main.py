'''
Created on 2019年9月19日

@author: juicemilk
'''
import tensorflow as tf
from Eye_net_project.Eye_net.Height_net.Height_Train import height_train
from Eye_net_project.Eye_net.Width_net.Width_Train import width_train
from Eye_net_project.Eye_net.lib.Data_Process import data_process

"""
function declaration:

train_main: the function of train
“训练网络函数”
"""
def train_main():
    
    train_data_file='./Data/final_data/train_data/train_data.npy'   #训练数据集的路径
    test_data_file='./Data/final_data/test_data/test_data.npy'      #测试数据集的路径
    max_dict_file='./Data/final_data/max.npy'                       #最大数据点集的路径
    min_dict_file='./Data/final_data/min.npy'                       #最小数据点集的路径
    
    """
    Normalized data, return train data and test data
    “对数据进行归一化处理，并返回归一化后的训练集和测试集数据”
    """
    train_data,test_data,data_max,data_min=data_process(train_data_file,
                                                        test_data_file,
                                                        max_dict_file,
                                                        min_dict_file
                                                        )
        
    """
    choose train parameter
    """
    use_type='Height'                             # the type of train net 选择训练网络的类型
    activation_mode = 'relu'                      # activation mode: 'relu','sigmoid','tanh' 选择激活函数
    regularization_mode = 'l2'                    # regularization mode: 'l1','l2' 选择正则化类型
    cost_mode = '2'                               # cost loss mode: '1' is quadratic term,'2' is abs 选择代价函数类型，作为优化目标函数
    loss_mode = 'mre'                             # loss mode: 'rmse','mse','mae','mre' 选择损失函数类型，作为评估网络性能指标函数
    is_batch_normalization = True                 # True:use batch normalization,False:don't use batch normalization 是否使用批量归一化
    regularization_rate = 0.0001                  # regularization rate 正则化率
    learning_rate = 0.001                         # learning rate 学习率
    keep_prob = 1                                 # use dropout,1 is don't use dropout 是否使用dropout
    momentum = 0.2                                # the momentum of 'Gradient descent with Momentum' 动量大小
    epochs = 200                                  # training epochs 训练轮数
    batch_size = 100                              # training batch size 训练批次大小
    models_save_path='./Models/'+use_type+'_net'  # the path of saved model 保存模型的路径
    model_name='my_model'                         # the name of saved model 保存模型的名字
    save_model_num=3                              # the number of saved newest model 保存最新模型的数目
    use_tensorboard=False                    # whether to use tensorboard to record the training process,default False 是否使用tensorboard记录训练过程，默认不使用
    
    
    if use_type=='Height':
        height_train( 
                activation_mode,
                regularization_mode,
                cost_mode,
                loss_mode,
                is_batch_normalization,
                regularization_rate,
                learning_rate,
                keep_prob,
                momentum,
                epochs,
                batch_size,
                train_data,
                test_data,
                data_max,
                data_min,
                use_type,
                models_save_path,
                model_name,
                save_model_num,
                use_tensorboard
                   )   
    elif use_type=='Width':
        width_train( 
                activation_mode,
                regularization_mode,
                cost_mode,
                loss_mode,
                is_batch_normalization,
                regularization_rate,
                learning_rate,
                keep_prob,
                momentum,
                epochs,
                batch_size,
                train_data,
                test_data,
                data_max,
                data_min,
                use_type,
                models_save_path,
                model_name,
                save_model_num,
                use_tensorboard
                )   
    else:
        print('ERROR ! please input correct train type')       
        

def main(argv=None):
    train_main()
if __name__=='__main__':
    tf.app.run()
