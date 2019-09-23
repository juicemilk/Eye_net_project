'''
Created on 2019年3月30日

@author: juicemilk
'''
import tensorflow as tf
import numpy as np
import os
from pyecharts import Pie
from Eye_net_project.Eye_net.lib.Remove_File import remove_file

"""
function declaration:

predict: predict test data by trained models
“使用训练好的模型对测试数据进行测试”
"""

"""
parameter declaration:
Test_data: the test data
“测试数据路径”
Data_max: the max data point for normalization
“数据每个维度的最大值”
Data_min: the min data point for normalization
“数据每个维度的最小值”
Use_type: the type of predict
“测试网络类型”
Models_save_path: the path of trained models
“模型路径”
"""
def predict(Test_data,Data_max,Data_min,Use_type,Models_save_path,Ues_pie):
    print('正在测试测试集数据.......')
    try:
        if not os.path.isdir('./Logs/'+Use_type+'_net/predict_result'):
            os.makedirs('./Logs/'+Use_type+'_net/predict_result')
    except Exception:
        print('无法创建测试结果报告目录')
        os._exit(0)
    remove_file('./Logs/'+Use_type+'_net/predict_result')
    predict_result=open('./Logs/'+Use_type+'_net/predict_result/predict_result.txt', 'w') 
    for k in ['index','real_label','output_label','error_rate']:
        predict_result.write(k+'                ')
    predict_result.write('\n')
    label_max=np.array(Data_max.get(Use_type))
    label_min=np.array(Data_min.get(Use_type))
    error_percent=[]
    tf.reset_default_graph()  
    with tf.Session() as sess:  
        ckpt=tf.train.get_checkpoint_state(Models_save_path)
        try:
            saver=tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')   
            saver.restore(sess,ckpt.model_checkpoint_path)
        except Exception:
            print('ERROR ! no models exist,please check the path of saved models')
            os._exit(0)
        graph=tf.get_default_graph()
        tx = graph.get_tensor_by_name('tx-input:0') 
        
        s = graph.get_tensor_by_name('s-input:0') 
        
        rx = graph.get_tensor_by_name('rx-input:0')
        
        y = graph.get_tensor_by_name('y-input:0')
        
        training = graph.get_tensor_by_name('training:0')
        
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
    
        logits=graph.get_tensor_by_name('dnn_all/logits:0')   
        
        loss_mes=graph.get_tensor_by_name('loss/logits_loss:0')
        loss_reg=graph.get_tensor_by_name('loss/regularization_loss:0')
        loss = graph.get_tensor_by_name('loss/loss:0')
        test_feed={keep_prob:1,training:False,tx:Test_data.get('Tx'),rx:Test_data.get('Rx'),s:Test_data.get('S'),y:Test_data.get(Use_type)}
        loss_mesval,loss_regval,lossval,logitsval,label=sess.run([loss_mes,loss_reg,loss,logits,y],feed_dict=test_feed)
        logits_real=logitsval*(label_max-label_min)+label_min
        y_real = label*(label_max-label_min)+label_min
        length=len(logits_real[:,0])
        for i in range(length):
            temp=abs(logits_real[:,0][i]-y_real[:,0][i])/y_real[:,0][i]*100
            predict_result.write(str(i)+'                        '+str(y_real[:,0][i])+'                       '+str(logits_real[:,0][i])+'                   '+str(temp)+'%')
            predict_result.write('\n')
            error_percent.append(temp)
    error_percent=np.array(error_percent)
    error_percent=np.sort(error_percent)
    statistics_error=['相对误差均值：'+str(np.mean(error_percent))+'%',
                      '相对误差标准差：'+str(np.std(error_percent))+'%',
                      '误差中位数：'+str(np.median(error_percent))+'%',
                      '最大误差：'+str(np.max(error_percent))+'%',
                      '最小误差：'+str(np.min(error_percent))+'%']
   
    for i in range(5):
        print(statistics_error[i])
        predict_result.write(statistics_error[i]+'\n')
    predict_result.close()
    if Ues_pie==True:
        count_list={'0-5':0,'5-10':0,'10-15':0,'15-20':0,'20-25':0,'25-30':0,'30-35':0,'35-40':0,'40-45':0,
                    '45-50':0,'50-55':0,'55-60':0,'60-65':0,'65-70':0,'70-75':0,'75-80':0,'80-85':0,'85-90':0,
                    '90-95':0,'95-100':0,}
        for v in error_percent:
            if v<5:
                count_list['0-5']=count_list.get('0-5')+1
            if (v>=5)&(v<10):
                count_list['5-10']=count_list.get('5-10')+1
            if (v>=10)&(v<15):
                count_list['10-15']=count_list.get('10-15')+1
            if (v>=15)&(v<20):
                count_list['15-20']=count_list.get('15-20')+1
            if (v>=20)&(v<25):
                count_list['20-25']=count_list.get('20-25')+1
            if (v>=25)&(v<30):
                count_list['25-30']=count_list.get('25-30')+1
            if (v>=30)&(v<35):
                count_list['30-35']=count_list.get('30-35')+1
            if (v>=35)&(v<40):
                count_list['35-40']=count_list.get('35-40')+1
            if (v>=40)&(v<45):
                count_list['40-45']=count_list.get('40-45')+1
            if (v>=45)&(v<50):
                count_list['45-50']=count_list.get('45-50')+1
            if (v>=50)&(v<55):
                count_list['50-55']=count_list.get('50-55')+1
            if (v>=55)&(v<60):
                count_list['55-60']=count_list.get('55-60')+1
            if (v>=60)&(v<65):
                count_list['60-65']=count_list.get('60-65')+1
            if (v>=65)&(v<70):
                count_list['65-70']=count_list.get('65-70')+1
            if (v>=70)&(v<75):
                count_list['70-75']=count_list.get('70-75')+1
            if (v>=75)&(v<80):
                count_list['75-80']=count_list.get('75-80')+1
            if (v>=80)&(v<85):
                count_list['80-85']=count_list.get('80-85')+1
            if (v>=85)&(v<90):
                count_list['85-90']=count_list.get('85-90')+1
            if (v>=90)&(v<95):
                count_list['90-95']=count_list.get('90-95')+1
            if (v>=95)&(v<100):
                count_list['95-100']=count_list.get('95-100')+1
        pie=Pie(Use_type+"错误率分布",title_pos='center',width=1500,height=700)
        pie.add("",list(count_list.keys()),list(count_list.values()),radius=[20,70],center=['50%','58%'],legend_top="bottom",is_label_show=True)
        pie.render('./Logs/'+Use_type+'_net/predict_result/error_rate.html')
    print('测试完成，已保存测试报告')