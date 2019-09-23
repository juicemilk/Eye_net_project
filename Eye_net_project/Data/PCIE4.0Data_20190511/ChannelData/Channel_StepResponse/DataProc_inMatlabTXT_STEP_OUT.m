% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% DATE: May 13, 2019
% 
% USAGE: ���ļ����ڶ�ȡChannel�Ľ�Ծ��ӦTXT�ļ���
% 
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

clear all
clc

% ��TXT�м�����ֵ������
fid = fopen('.\STEP_OUT.txt','r');
RawData = textscan(fid,'%f %f %f','HeaderLines',68,'commentStyle','DataFileList1.Index');
StepResponse_index = int32(RawData{1,1});                                   % Channel��Ծ��Ӧ�����������[1,2,3,...,64]
StepResponse_time = RawData{1,2};                                           % Channel��Ծ��Ӧ�Ĳ���ʱ����������λ��sec
StepResponse_amp = RawData{1,3};                                            % % Channel��Ծ��Ӧ��������������λ��V

% ȷ��Channel��Ծ��Ӧ�ĸ���
Num_StepResponse = StepResponse_index(end);

% ȷ��ÿ��Channel��Ծ��Ӧ�Ĳ�������
Len_StepResponse = length(StepResponse_index) / Num_StepResponse;

% ���ź���������������ÿһ�б�ʾһ����Ծ��Ӧ�����ܹ���Num_StepResponse����Ծ��Ӧ��
StepResponse_index_Matrix = zeros(Len_StepResponse,Num_StepResponse);       % Channel��Ծ��Ӧ����ž����޵�λ
StepResponse_time_Matrix = zeros(Len_StepResponse,Num_StepResponse);        % Channel��Ծ��Ӧ�Ĳ���ʱ�̾��󣬵�λ��sec
StepResponse_data_Matrix = zeros(Len_StepResponse,Num_StepResponse);        % Channel��Ծ��Ӧ�����ݾ��󣬵�λ��V

% ���Ŵ���
for n = 1:Num_StepResponse
    start_index = 1 + (n-1) * Len_StepResponse;
    stop_index = n * Len_StepResponse;
    % ��ȡ��ǰ��Channel��ţ�[1,2,3,...,64]
    StepResponse_index_Matrix(:,n) = StepResponse_index(start_index:stop_index);
    % ��ȡ��ǰ��Channel��Ծ��Ӧ����ʱ��
    StepResponse_time_Matrix(:,n) = StepResponse_time(start_index:stop_index);
    % ��ȡ��ǰ��Channel��Ծ��Ӧ���
    StepResponse_data_Matrix(:,n) = StepResponse_amp(start_index:stop_index);    
end

