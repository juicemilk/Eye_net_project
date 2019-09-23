'''
Created on 2019年6月5日

@author: juicemilk
'''
import tensorflow as tf

"""
function declaration:

my_dense_layer:define the hide net layer
“定义隐藏层”
my_logits_layer:define the output net layer
“定义输出层”
"""

"""
parameter declaration:

tensor_name: the name of input tensor
“输入张量的名字”
hide_node: the number of neuron in the layer
“每一层神经元的数目”
training: the status of net,training or not training,it's a tensor
“网络是否正在训练”
keep_prob: the keep_prob of dropout,if you input 1 ,that mean you don't use dropout ,it's a tensor
“是否使用dropout”
activation_mode: choose the activation function 'relu','sigmoid','tanh'
“选择激活函数，包括relu，sigmoid，tanh”
regularization_mode: choose the mode of regularization,'l1','l2'
“选择正则化类型，包括l1范数和l2范数”
is_batch_normalization: whether to use batch normalization,True or False
“是否使用批量归一化”
regularization_rate: the rate of regularization 
“正则化率”
layer_name: the name of this layer
“网络名字”
output_name the name of this layer's output
“网络输出张量的名字”
"""
def my_dense_layer(tensor_name,hide_node,training,keep_prob,activation_mode,regularization_mode,is_batch_normalization,regularization_rate,layer_name,output_name):
    if(regularization_mode=='l1'):
        regularization_function=tf.contrib.layers.l1_regularizer(regularization_rate)
    elif(regularization_mode=='l2'):
        regularization_function=tf.contrib.layers.l2_regularizer(regularization_rate)
    else:
        print("ERROR ! please input correct regularization mode")
    dense_layer = tf.layers.dense(tensor_name,hide_node,kernel_regularizer=regularization_function,trainable=True,name=layer_name)
    
    if(is_batch_normalization=='True'):
        dense_layer =tf.layers.batch_normalization(dense_layer, training=training)
    else:
        pass
    
    dense_layer = tf.nn.dropout(dense_layer,keep_prob=keep_prob)  
    
    if(activation_mode=='relu'):
        dense_layer=tf.nn.relu(dense_layer,name=output_name)
    elif(activation_mode=='sigmoid'):
        dense_layer=tf.nn.sigmoid(dense_layer,name=output_name)
    elif(activation_mode=='tanh'):
        dense_layer=tf.nn.tanh(dense_layer,name=output_name)
    else:
        print("ERROR ! please input correct activation mode!")
    
    return dense_layer

def my_logits_layer(tensor_name,hide_node,training,keep_prob,regularization_mode,is_batch_normalization,regularization_rate,layer_name,output_name):
    if(regularization_mode=='l1'):
        regularization_function=tf.contrib.layers.l1_regularizer(regularization_rate)
    elif(regularization_mode=='l2'):
        regularization_function=tf.contrib.layers.l2_regularizer(regularization_rate)
    else:
        print("ERROR ! please input correct regularization mode")
    logits_layer = tf.layers.dense(tensor_name,hide_node,kernel_regularizer=regularization_function,trainable=True,name=layer_name)
    
    if(is_batch_normalization=='True'):
        logits_layer =tf.layers.batch_normalization(logits_layer, training=training)
    else:
        pass
    
    logits_layer = tf.nn.dropout(logits_layer,keep_prob=keep_prob)  
    logits_layer=tf.add(logits_layer,0,name=output_name)
    return logits_layer
    