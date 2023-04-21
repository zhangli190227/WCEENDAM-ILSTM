%%%%%%%%%%%%%WCEEMDAN_ILSTM����ʾ��
%% ��ջ���
tic;clc;clear;close all;format compact
%% ��������
[num,txt,raw]=xlsread('��������.����2020.xlsx',1);
Monitoring_site=txt(2:end,2);
[row,col]=find(strcmp(Monitoring_site,'ƽ�ŷ־�')) ;
index_n = 4;%PM2.5������num��4��
Total_PM2pot5_data=num(:,index_n);
PM2pot5_data = Total_PM2pot5_data(row);
[row,col]=size(PM2pot5_data);
%% �㳣ֵ����
%�������ֵ����0��ʹ��ǰ��ʱ������ƽ���ķ���������
for i=1:row
   if PM2pot5_data(i,end)==0
       PM2pot5_data(i,:)=mean([PM2pot5_data(i-1,:),PM2pot5_data(i+1,:)]);
   end
end

Mean = mean(PM2pot5_data);
SD = std2(PM2pot5_data);
Max_value = max(PM2pot5_data);
Min_value = min(PM2pot5_data);

ecg= PM2pot5_data;
%%%%%%%%%%%CEEMDAN
Nstd = 0.2;
NR = 50;
MaxIter = 500;

[modes its]=ceemdan(ecg,Nstd,NR,MaxIter);
t=1:length(ecg);

[a b]=size(modes);

figure;
subplot(a+1,1,1);
plot(t,ecg);% the ECG signal is in the first row of the subplot
ylabel('ECG')
set(gca,'xtick',[])
axis tight;

for i=2:a
    subplot(a+1,1,i);
    plot(t,modes(i-1,:));
    ylabel (['IMF ' num2str(i-1)]);
    set(gca,'xtick',[])
    xlim([1 length(ecg)])
end

subplot(a+1,1,a+1)
plot(t,modes(a,:))
ylabel(['IMF ' num2str(a)])
xlim([1 length(ecg)])

figure;
boxplot(its);
modes_data=modes';
features=modes_data((1:end-1),:);
output_data=ecg((2:end),:);
%%%%%%%%%%%%%%%%
%ѵ��������ǰ70%
input_train = features(1:(end-1464),:);%ѵ����������
output_train = output_data(1:(end-1464),:);%ѵ���������
%���Լ�������30%
input_test = features((end-1463):end,:);%������������
output_test = output_data((end-1463):end,:);%�����������
%���ݹ�һ����ͳһ����������λ
[inputn,inputps]=mapminmax(input_train');
[outputn,outputps]=mapminmax(output_train');
tn=mapminmax('apply',input_test',inputps);%���ݹ�һ��
yn=output_test';
%%%%%%%%%%%%%%%

%���㻥��Ϣ
X=input_train';
Y=output_train';
len_Y = length(Y);
[row,col]=size(X);

mi_all = [];
for i=1:row
mi_calc=calc_mi(X(i,:)',Y',len_Y);
mi_all = [mi_all;mi_calc];
end
mi_all_sum = sum(mi_all);

for j=1:length(mi_all)
lamda(j)=mi_all(j)/mi_all_sum;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter_pso = 1:10
iter_pso
%ѵ��������ǰ70%
input_train = features(1:(end-1464),:);%ѵ����������
output_train = output_data(1:(end-1464),:);%ѵ���������
%���Լ�������30%
input_test = features((end-1463):end,:);%������������
output_test = output_data((end-1463):end,:);%�����������
%���ݹ�һ����ͳһ����������λ
[inputn,inputps]=mapminmax(input_train');
[outputn,outputps]=mapminmax(output_train');
tn=mapminmax('apply',input_test',inputps);%���ݹ�һ��
yn=output_test';

XTrain = inputn.*lamda';
YTrain = outputn;

XTest = tn.*lamda';
YTest = yn;


%% 2.���PSO�Ż�LSTM������ѡ����ѵ�ѧϰ�ʡ����ز�ڵ���������С
pso_option = struct('c1',1,'c2',1,'maxgen',10,'sizepop',10,'k',0.6,'wV',0.9,'wP',0.9, ...
    'popLearnRatemax',1,'popLearnRatemin',10^(-3),'popHiddenUnitsmax',64,'popHiddenUnitsmin',10,'popMiniBatchSizemax',128,'popMiniBatchSizemin',1);%�����Ľ�����psoSVMcgForRegress����

[bestmse,bestLearnRate,bestHiddenUnits,bestMiniBatchSize,trace] = ampsolstmForRegress(YTrain,XTrain,YTest,XTest,outputps,pso_option);

Bestmse(iter_pso)=bestmse;
BestLearnRate(iter_pso)=bestLearnRate;
BestHiddenUnits(iter_pso)=bestHiddenUnits;
BestMiniBatchSize(iter_pso)=bestMiniBatchSize;

%�̶��������,ʹ��ѵ��������� 
setdemorandstream(pi);
% ���� LSTM �ع����硣
numFeatures = size(XTrain,1);
numResponses = size(YTrain,1);
numHiddenUnits = round(bestHiddenUnits);
layers = [ ...
    sequenceInputLayer(numFeatures)
%     fullyConnectedLayer(numFeatures)
   lstmLayer(numHiddenUnits,'OutputMode','sequence')
%    dropoutLayer(0.5)
%         lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% % ָ��ѵ��ѡ�ʹ������� 'adam' �Դ�СΪ 20 ��С�������� 60 ��ѵ����ָ��ѧϰ��Ϊ 0.01��Ҫ��ֹ�ݶȱ�ը���뽫�ݶ���ֵ����Ϊ 1��Ҫʹ���б��ְ����������뽫 'Shuffle' ����Ϊ 'never'��
maxEpochs = 60;
% miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',round(bestMiniBatchSize),...
    'InitialLearnRate',bestLearnRate, ...
    'GradientThreshold',1, ...
    'Plots','training-progress',...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.2, ...
    'ExecutionEnvironment','cpu', ...
    'Shuffle','never', ...
    'Verbose',0);

%% ѵ�� LSTM ����
% ʹ�� trainNetwork ��ָ����ѵ��ѡ��ѵ�� LSTM ���硣
net = trainNetwork(XTrain,YTrain,layers,options);

% load('net')

%% Ԥ�⽫��ʱ�䲽

YPred = predict(net,XTest);

%ʹ����ǰ����Ĳ�����Ԥ��ȥ��׼����
YPred = mapminmax('reverse',YPred,outputps);%���ݷ���һ��

Sum_predict_test(iter_pso,:)=YPred;

error=output_test-Sum_predict_test(iter_pso,:)';

%������
Error_max(iter_pso) =max(abs(error));

%���������RMSE
Rmse(iter_pso) = sqrt(mean(((error).^2)));

%Mapeƽ���ٷֱ����
Mape(iter_pso) = mean(abs((error)./output_test));

%MAEƽ���������
Mae(iter_pso) = mean(abs(error));

%����ϵ��R2
R2(iter_pso) = 1 - norm(output_test-Sum_predict_test(iter_pso,:)')^2/norm(output_test - mean(output_test))^2;

end
toc %������ʱ