% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% DATE: May 13, 2019
% 
% USAGE: 该文件用于读取Channel的阶跃响应TXT文件。
% 
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

clear all
clc

% 从TXT中加载数值型数据
fid = fopen('.\STEP_OUT.txt','r');
RawData = textscan(fid,'%f %f %f','HeaderLines',68,'commentStyle','DataFileList1.Index');
StepResponse_index = int32(RawData{1,1});                                   % Channel阶跃响应的序号向量：[1,2,3,...,64]
StepResponse_time = RawData{1,2};                                           % Channel阶跃响应的采样时刻向量，单位：sec
StepResponse_amp = RawData{1,3};                                            % % Channel阶跃响应的数据向量，单位：V

% 确定Channel阶跃响应的个数
Num_StepResponse = StepResponse_index(end);

% 确定每个Channel阶跃响应的采样点数
Len_StepResponse = length(StepResponse_index) / Num_StepResponse;

% 重排后的输出变量声明【每一列表示一个阶跃响应，即总共有Num_StepResponse个阶跃响应】
StepResponse_index_Matrix = zeros(Len_StepResponse,Num_StepResponse);       % Channel阶跃响应的序号矩阵，无单位
StepResponse_time_Matrix = zeros(Len_StepResponse,Num_StepResponse);        % Channel阶跃响应的采样时刻矩阵，单位：sec
StepResponse_data_Matrix = zeros(Len_StepResponse,Num_StepResponse);        % Channel阶跃响应的数据矩阵，单位：V

% 重排处理
for n = 1:Num_StepResponse
    start_index = 1 + (n-1) * Len_StepResponse;
    stop_index = n * Len_StepResponse;
    % 提取当前的Channel序号：[1,2,3,...,64]
    StepResponse_index_Matrix(:,n) = StepResponse_index(start_index:stop_index);
    % 提取当前的Channel阶跃响应采样时刻
    StepResponse_time_Matrix(:,n) = StepResponse_time(start_index:stop_index);
    % 提取当前的Channel阶跃响应结果
    StepResponse_data_Matrix(:,n) = StepResponse_amp(start_index:stop_index);    
end

