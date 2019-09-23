'''
Created on 2019年9月20日

@author: juicemilk
'''
from Eye_net_project.Read_data import Data_To_Dict,Read_Label,Read_Rx_Pf,Read_Rx_Zero,Read_S,Read_Tx

"""
function declaration:

extract_data_main: Read the data from different dimensions and compose them into a dictionary to return the training set and the test set.
“从原始文件中读取不同为维度的数据，并将其组合成字典格式，同时将其划分成训练集和测试集，保存以上所有数据”
"""
def extract_data_main():
    
    #原始不同维度数据文件路径#
    Tx_input_file='./Data/PCIE4.0Data_20190511/InputPara_TX_Preset.xlsx'
    S_input_file='./Data/PCIE4.0Data_20190511/ChannelData/Channel_StepResponse/STEP_OUT_PCA_Compressed.txt'
    Rx_Zero_input_file='./Data/PCIE4.0Data_20190511/InputPara_RX_CTLE_ZERO.xlsx'
    Rx_Pf_input_file='./Data/PCIE4.0Data_20190511/InputPara_RX_CTLE_PF.xlsx'
    Label_input_file='./Data/PCIE4.0Data_20190511/BatchSimulationResults.csv'
    
    #读取不同维度数据的保存路径#
    Tx_output_file='./Data/temp_data/tx.txt'
    S_output_file='./Data/temp_data/s.txt'
    Rx_Zero_output_file='./Data/temp_data/rx_zero.txt'
    Rx_Pf_output_file='./Data/temp_data/rx_pf.txt'
    Label_output_file='./Data/temp_data/label.txt'
    
    #所有维度数据组合后的路径#
    data_file='./Data/temp_data/data.txt'
    data_dict_file='./Data/temp_data/data.npy'
    
    #训练数据、测试数据以及不同维度最大（小）值的保存路径#
    train_file='./Data/final_data/train_data/train_data.txt'
    train_data_file='./Data/final_data/train_data/train_data.npy'
    test_file='./Data/final_data/test_data/test_data.txt'
    test_data_file='./Data/final_data/test_data/test_data.npy'
    max_file='./Data/final_data/max.txt'
    max_dict_file='./Data/final_data/max.npy'
    min_file='./Data/final_data/min.txt'
    min_dict_file='./Data/final_data/min.npy'
    
    #训练数据占比
    train_rate=0.8
    
    #从原始文件中分别读取不同维度的数据并保存#
    Read_Tx.read_tx(Tx_input_file, Tx_output_file)
    Read_S.read_s(S_input_file, S_output_file)
    Read_Rx_Zero.read_rx_zero(Rx_Zero_input_file, Rx_Zero_output_file)
    Read_Rx_Pf.read_rx_pf(Rx_Pf_input_file, Rx_Pf_output_file)
    Read_Label.read_label(Label_input_file, Label_output_file)
    
    #将不同维度数据组合成字典格式，并将其划分成训练集和测试集#
    Data_To_Dict.data_to_dict(Tx_output_file,
                              S_output_file,
                              Rx_Pf_output_file, 
                              Rx_Zero_output_file,
                              Label_output_file,
                              data_file,
                              data_dict_file,
                              train_rate,
                              train_file,
                              train_data_file,
                              test_file,
                              test_data_file, 
                              max_file, 
                              max_dict_file,
                              min_file, 
                              min_dict_file)
    
if __name__=='__main__':
    extract_data_main()