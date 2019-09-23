'''
Created on 2019年6月5日

@author: juicemilk
'''
"""
function declaration:

height_train:the network topology of training net

"眼高网络的结构"
"""
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from Eye_net_project.Eye_net.lib.Data_Shuffle import data_shuffle
from Eye_net_project.Eye_net.lib.Data_Batch import data_batch
from Eye_net_project.Eye_net.lib.Def_My_Layer import my_dense_layer,my_logits_layer
from Eye_net_project.Eye_net.lib.Remove_File import remove_file

"""
The number of neuron in every layer

"网络中每一层的神经元个数"
"""
TX_INPUT_NODE = 3
S_INPUT_NODE = 12
RX_INPUT_NODE =2
OUTPUT_NODE = 1

TX_LAYER1_NODE = 10
TX_LAYER2_NODE = 50
TX_LAYER3_NODE = 30
S_LAYER1_NODE=400
S_LAYER2_NODE=30
RX_LAYER1_NODE = 10
RX_LAYER2_NODE = 50
RX_LAYER3_NODE = 30
LAYER4_NODE = 800
LAYER5_NODE = 400
LAYER6_NODE = 200
LAYER7_NODE = 100
LAYER8_NODE = 20
LAYER9_NODE = 5
LAYER10_NODE = 1

def height_train(Activation_mode,Regularization_mode,Cost_mode,Loss_mode,Is_batch_normalization,Regularization_rate,Learning_rate,Keep_prob,MOMENTUM,Epochs,Batch_size,Train_data,Test_data,Data_max,Data_min,Use_type,Models_save_path,Model_name,Save_model_num,Use_tensorboard):

    
    tx = tf.placeholder(tf.float64,[None,TX_INPUT_NODE],name = 'tx-input') 
    s = tf.placeholder(tf.float64,[None,S_INPUT_NODE],name = 's-input')  
    rx = tf.placeholder(tf.float64,[None,RX_INPUT_NODE],name = 'rx-input') 
    y = tf.placeholder(tf.float64,[None,OUTPUT_NODE],name = 'y-input')
    training = tf.placeholder_with_default(False, shape=(), name='training')
    keep_prob = tf.placeholder(tf.float64,name='keep_prob') 
    
    with tf.name_scope('dnn_Tx'):
        TX_LAYER1_OUTPUT = my_dense_layer(tx, TX_LAYER1_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'tx_layer1', 'tx_layer1_output')
        
        TX_LAYER2_OUTPUT = my_dense_layer(TX_LAYER1_OUTPUT, TX_LAYER2_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'tx_layer2', 'tx_layer2_output')
        
        TX_LAYER3_OUTPUT = my_dense_layer(TX_LAYER2_OUTPUT, TX_LAYER3_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'tx_layer3', 'tx_layer3_output')
    
    with tf.name_scope('dnn_S'):
        S_LAYER1_OUTPUT = my_dense_layer(s, S_LAYER1_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 's_layer1', 's_layer1_output')
        
        S_LAYER2_OUTPUT = my_dense_layer(S_LAYER1_OUTPUT, S_LAYER2_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 's_layer2', 's_layer2_output')

               
    with tf.name_scope('dnn_Rx'):
        RX_LAYER1_OUTPUT = my_dense_layer(rx, RX_LAYER1_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'rx_layer1', 'rx_layer1_output')

        RX_LAYER2_OUTPUT = my_dense_layer(RX_LAYER1_OUTPUT, RX_LAYER2_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'rx_layer2', 'rx_layer2_output')

        RX_LAYER3_OUTPUT = my_dense_layer(RX_LAYER2_OUTPUT, RX_LAYER3_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'rx_layer3', 'rx_layer3_output')
        
    with tf.name_scope('dnn_all'):
        ALL_INPUT = tf.concat([TX_LAYER3_OUTPUT,S_LAYER2_OUTPUT,RX_LAYER3_OUTPUT],1,name = 'all_input')    
           
        LAYER4_OUTPUT = my_dense_layer(ALL_INPUT, LAYER4_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'layer4', 'layer4_output')
        
        LAYER5_OUTPUT = my_dense_layer(LAYER4_OUTPUT, LAYER5_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'layer5', 'layer5_output')
        
        LAYER6_OUTPUT = my_dense_layer(LAYER5_OUTPUT, LAYER6_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'layer6', 'layer6_output')
        
        LAYER7_OUTPUT = my_dense_layer(LAYER6_OUTPUT, LAYER7_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'layer7', 'layer7_output')
        
        LAYER8_OUTPUT = my_dense_layer(LAYER7_OUTPUT, LAYER8_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'layer8', 'layer8_output')
        
        LAYER9_OUTPUT = my_dense_layer(LAYER8_OUTPUT, LAYER9_NODE, training, keep_prob, Activation_mode, Regularization_mode, Is_batch_normalization, Regularization_rate, 'layer9', 'layer9_output')
        
        logits = my_logits_layer(LAYER9_OUTPUT, LAYER10_NODE, training, keep_prob, Regularization_mode, Is_batch_normalization, Regularization_rate, 'logits_layer', 'logits')
        
    with tf.name_scope('loss'):
        if Cost_mode=='1':
            logits_loss = tf.div(tf.reduce_mean(tf.square(logits-y)),2.0,name = 'logits_loss')
        elif Cost_mode=='2':
            logits_loss = tf.reduce_mean(tf.abs(logits-y),name = 'logits_loss')
        else:
            print('ERROR,please input correct cost mode')
        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name = 'regularization_loss')
        loss = tf.add(logits_loss,regularization_loss,name = 'loss')  
        
    with tf.name_scope('eval'):
        label_max=np.array(Data_max.get(Use_type))
        label_min=np.array(Data_min.get(Use_type))
        logits_real=logits*(label_max-label_min)+label_min
        y_real = y*(label_max-label_min)+label_min
        if Loss_mode=='rmse' :           
            error_rate=tf.div(tf.sqrt(tf.reduce_mean(tf.square(logits_real-y_real))),tf.reduce_mean(y_real),name='rmse_loss')
        elif Loss_mode=='mse':
            error_rate=tf.div(tf.reduce_mean(tf.square(logits_real-y_real)),tf.square(tf.reduce_mean(y_real)),name='mse_loss')
        elif Loss_mode=='mae':
            error_rate=tf.div(tf.reduce_mean(tf.abs(logits_real-y_real)),tf.reduce_mean(y_real),name='mae_loss')
        elif Loss_mode=='mre':
            error_rate=tf.reduce_mean(tf.div(tf.abs(logits_real-y_real),y_real),name='precent_loss')
        else:
            print('ERROR,please input correct loss mode')
            
    with tf.name_scope('train'):
        global_step = tf.Variable(0,trainable = False)
        optimizer = tf.train.MomentumOptimizer(Learning_rate,MOMENTUM,name = 'optimizer')
        if(Is_batch_normalization==False):
            train_op = optimizer.minimize(loss,global_step=global_step)
            saver = tf.train.Saver(max_to_keep=Save_model_num)
        else:
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops): 
                train_op = optimizer.minimize(loss,global_step=global_step)
            saver = tf.train.Saver(max_to_keep=Save_model_num,var_list=tf.global_variables())
        
    init = tf.global_variables_initializer()
    try:
        if not os.path.isdir(Models_save_path):
            os.makedirs(Models_save_path)
        if not os.path.isdir('./Logs/'+Use_type+'_net/tensorboard/train'):
            os.makedirs('./Logs/'+Use_type+'_net/tensorboard/train')
        if not os.path.isdir('./logs/'+Use_type+'_net/tensorboard/test'):
            os.makedirs('./Logs/'+Use_type+'_net/tensorboard/test')
        if not os.path.isdir('./Logs/'+Use_type+'_net/train_result'):
            os.makedirs('./Logs/'+Use_type+'_net/train_result')
    except Exception:
        print('无法创建模型存储目录或训练过程报告目录')
        os._exit(0)
    remove_file(Models_save_path)
    remove_file('./Logs/'+Use_type+'_net/tensorboard/train')
    remove_file('./Logs/'+Use_type+'_net/tensorboard/test')   
    if Use_tensorboard ==True:  
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('error rate',error_rate)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)            
        merged_summary_op=tf.summary.merge_all()
    
    loss_list={'train_loss':[],'test_loss':[]}
    error_rate_list={'train_error_rate':[],'test_error_rate':[]}
    
    loss_file=open('./Logs/'+Use_type+'_net/train_result/loss_list.txt', 'w') 
    error_rate_file=open('./Logs/'+Use_type+'_net/train_result/error_rate_list.txt', 'w') 
    
    with tf.Session() as sess:
        print('开始训练......')
        if Use_tensorboard==True:
            writer1=tf.summary.FileWriter('./Logs/'+Use_type+'_net/tensorboard/train',sess.graph)
            writer2=tf.summary.FileWriter('./Logs/'+Use_type+'_net/tensorboard/test',sess.graph)
        init.run()
        Epochs_step=(int)(len(Train_data.get(Use_type))/Batch_size)
        test_feed={keep_prob:1,training:False,tx:Test_data.get('Tx'),rx:Test_data.get('Rx'),s:Test_data.get('S'),y:Test_data.get(Use_type)}
        train_feed={keep_prob:1,training:False,tx:Train_data.get('Tx'),rx:Train_data.get('Rx'),s:Train_data.get('S'),y:Train_data.get(Use_type)}
		
#         x=[]
#         fig,ax=plt.subplots(1,2)
#         ax[0].set(title='loss')
#         ax[0].set_xlim(1,Epochs)
#         ax[0].set_ylim(0,0.5)
#         line1,=ax[0].plot(x,loss_list.get('train_loss'),c='red')
#         line2,=ax[0].plot(x,loss_list.get('test_loss'),c='blue')
#         ax[0].legend(['train_loss','test_loss'])
#         ax[1].set(title='error_rate')
#         ax[1].set_xlim(1,Epochs)
#         ax[1].set_ylim(0,1)
#         line3,=ax[1].plot(x,error_rate_list.get('train_error_rate'),c='red')
#         line4,=ax[1].plot(x,error_rate_list.get('test_error_rate'),c='blue')
#         ax[1].legend(['train_error_rate','test_error_rate'])

        for i in range(Epochs):  
            train_shuffer_data=data_shuffle(Train_data,Use_type)
            for j in range(Epochs_step) :
                train_batch_data=data_batch(train_shuffer_data, Batch_size, j)
                train_feed_batch={keep_prob:Keep_prob,training:True,tx:train_batch_data.get('Tx'),rx:train_batch_data.get('Rx'),s:train_batch_data.get('S'),y:train_batch_data.get(Use_type)}
                sess.run(train_op,feed_dict=train_feed_batch)
            if Use_tensorboard==True:
                train_loss,train_eval,summary= sess.run([loss,error_rate,merged_summary_op],feed_dict = train_feed)
                print('after %d Epochs training,the train loss is %g,the train error rate is %g'%(i,train_loss,train_eval))
                writer1.add_summary(summary,i)
                writer1.flush()
                test_loss,test_eval,summary= sess.run([loss,error_rate,merged_summary_op],feed_dict = test_feed)
                writer2.add_summary(summary,i)
                writer2.flush()
                print('after %d Epochs training,the test loss is %g,the test error rate is %g'%(i,test_loss,test_eval))
            else:
                train_loss,train_eval= sess.run([loss,error_rate],feed_dict = train_feed)
                print('after %d Epochs training,the train loss is %g,the train error rate is %g'%(i,train_loss,train_eval))
                test_loss,test_eval= sess.run([loss,error_rate],feed_dict = test_feed)
                print('after %d Epochs training,the test loss is %g,the test error rate is %g'%(i,test_loss,test_eval))       
            saver.save(sess, os.path.join(Models_save_path,Model_name),global_step = i)
            loss_list.get('train_loss').append(train_loss)
            loss_list.get('test_loss').append(test_loss)
            error_rate_list.get('train_error_rate').append(train_eval)
            error_rate_list.get('test_error_rate').append(test_eval)
			
#             x.append(i)
#             line1.set_xdata(x)
#             line1.set_ydata(loss_list.get('train_loss'))
#             line2.set_xdata(x)
#             line2.set_ydata(loss_list.get('test_loss'))
#             line3.set_xdata(x)
#             line3.set_ydata(error_rate_list.get('train_error_rate'))
#             line4.set_xdata(x)
#             line4.set_ydata(error_rate_list.get('test_error_rate'))
#             plt.pause(0.1)
            
        print('训练完成')
        print('正在保存训练过程数据.........')
        for k in loss_list.keys():
            loss_file.write(str(k)+'                ')
        loss_file.write('\n')
        for i in range(Epochs):
            for v in loss_list.values():
                loss_file.write(str(v[i])+'                ')
            loss_file.write('\n')
        loss_file.close()
                   
        for k in error_rate_list.keys():
            error_rate_file.write(str(k)+'                ')
        error_rate_file.write('\n')
        for i in range(Epochs):
            for v in error_rate_list.values():
                error_rate_file.write(str(v[i])+'                ')
            error_rate_file.write('\n')
        error_rate_file.close()
        print('保存完成')
        plt.figure(1)
        plt.plot(loss_list.get('train_loss'),c='red')
        plt.plot(loss_list.get('test_loss'),c='blue')
        plt.legend(['train_loss','test_loss'])
        plt.title('loss')
        plt.figure(2)
        plt.plot(error_rate_list.get('train_error_rate'),c='red')
        plt.plot(error_rate_list.get('test_error_rate'),c='blue')
        plt.legend(['train_error_rate','test_error_rate'])
        plt.title('error_rate')
        plt.show()