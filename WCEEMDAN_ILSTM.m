%%%%%%%%%%%%%WCEEMDAN_ILSTM程序示例
%% 清空环境
tic;clc;clear;close all;format compact
%% 加载数据
[num,txt,raw]=xlsread('空气质量.信阳2020.xlsx',1);
Monitoring_site=txt(2:end,2);
[row,col]=find(strcmp(Monitoring_site,'平桥分局')) ;
index_n = 4;%PM2.5数据在num第4列
Total_PM2pot5_data=num(:,index_n);
PM2pot5_data = Total_PM2pot5_data(row);
[row,col]=size(PM2pot5_data);
%% 零常值处理
%对于输出值等于0，使用前后时刻数据平均的方法来处理
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
%训练集——前70%
input_train = features(1:(end-1464),:);%训练样本输入
output_train = output_data(1:(end-1464),:);%训练样本输出
%测试集——后30%
input_test = features((end-1463):end,:);%测试样本输入
output_test = output_data((end-1463):end,:);%测试样本输出
%数据归一化，统一基本度量单位
[inputn,inputps]=mapminmax(input_train');
[outputn,outputps]=mapminmax(output_train');
tn=mapminmax('apply',input_test',inputps);%数据归一化
yn=output_test';
%%%%%%%%%%%%%%%

%计算互信息
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
%训练集——前70%
input_train = features(1:(end-1464),:);%训练样本输入
output_train = output_data(1:(end-1464),:);%训练样本输出
%测试集——后30%
input_test = features((end-1463):end,:);%测试样本输入
output_test = output_data((end-1463):end,:);%测试样本输出
%数据归一化，统一基本度量单位
[inputn,inputps]=mapminmax(input_train');
[outputn,outputps]=mapminmax(output_train');
tn=mapminmax('apply',input_test',inputps);%数据归一化
yn=output_test';

XTrain = inputn.*lamda';
YTrain = outputn;

XTest = tn.*lamda';
YTest = yn;


%% 2.设计PSO优化LSTM，用于选择最佳的学习率、隐藏层节点和批处理大小
pso_option = struct('c1',1,'c2',1,'maxgen',10,'sizepop',10,'k',0.6,'wV',0.9,'wP',0.9, ...
    'popLearnRatemax',1,'popLearnRatemin',10^(-3),'popHiddenUnitsmax',64,'popHiddenUnitsmin',10,'popMiniBatchSizemax',128,'popMiniBatchSizemin',1);%参数的解释在psoSVMcgForRegress里面

[bestmse,bestLearnRate,bestHiddenUnits,bestMiniBatchSize,trace] = ampsolstmForRegress(YTrain,XTrain,YTest,XTest,outputps,pso_option);

Bestmse(iter_pso)=bestmse;
BestLearnRate(iter_pso)=bestLearnRate;
BestHiddenUnits(iter_pso)=bestHiddenUnits;
BestMiniBatchSize(iter_pso)=bestMiniBatchSize;

%固定随机种子,使其训练结果不变 
setdemorandstream(pi);
% 创建 LSTM 回归网络。
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

% % 指定训练选项。使用求解器 'adam' 以大小为 20 的小批量进行 60 轮训练。指定学习率为 0.01。要防止梯度爆炸，请将梯度阈值设置为 1。要使序列保持按长度排序，请将 'Shuffle' 设置为 'never'。
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

%% 训练 LSTM 网络
% 使用 trainNetwork 以指定的训练选项训练 LSTM 网络。
net = trainNetwork(XTrain,YTrain,layers,options);

% load('net')

%% 预测将来时间步

YPred = predict(net,XTest);

%使用先前计算的参数对预测去标准化。
YPred = mapminmax('reverse',YPred,outputps);%数据反归一化

Sum_predict_test(iter_pso,:)=YPred;

error=output_test-Sum_predict_test(iter_pso,:)';

%最大误差
Error_max(iter_pso) =max(abs(error));

%均方根误差RMSE
Rmse(iter_pso) = sqrt(mean(((error).^2)));

%Mape平均百分比误差
Mape(iter_pso) = mean(abs((error)./output_test));

%MAE平均绝对误差
Mae(iter_pso) = mean(abs(error));

%决定系数R2
R2(iter_pso) = 1 - norm(output_test-Sum_predict_test(iter_pso,:)')^2/norm(output_test - mean(output_test))^2;

end
toc %结束计时