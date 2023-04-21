function [bestCVmse,bestLearnRate,bestHiddenUnits,bestMiniBatchSize,fit_gen] = ampsolstmForRegress(train_label,train,T_test,P_test,outputps,pso_option)

%% 参数初始化
% c1:pso参数局部搜索能力
% c2:pso参数全局搜索能力
% maxgen:最大进化数量
% sizepop:种群最大数量
% k:k belongs to [0.1,1.0],速率和x的关系(V = kX)
% wV:(wV best belongs to [0.8,1.2]),速率更新公式中速度前面的弹性系数
% wP:种群更新公式中速度前面的弹性系数

% popLearnRatemax:LSTM 参数学习率LearnRate的变化的最大值.
% popLearnRatemin:LSTM 参数学习率LearnRate的变化的最小值.

% popHiddenUnitsmax:LSTM 参数隐藏层节点HiddenUnits的变化的最大值.
% popHiddenUnitsmin:LSTM 参数隐藏层节点HiddenUnits的变化的最小值.
% popMiniBatchSizemax:LSTM 参数批处理MiniBatchSize的变化的最大值.
% popMiniBatchSizemin:LSTM 参数批处理MiniBatchSize的变化的最小值.

VLearnRatemax = pso_option.k*pso_option.popLearnRatemax;
VLearnRatemin = -VLearnRatemax ;
VHiddenUnitsmax = pso_option.k*pso_option.popHiddenUnitsmax;
VHiddenUnitsmin = -VHiddenUnitsmax ;
VMiniBatchSizemax = pso_option.k*pso_option.popMiniBatchSizemax;
VMiniBatchSizemin = -VMiniBatchSizemax ;

wmax=0.9;wmin=0.6;
% [row,col]=size(P_test);
% P_test = P_test';
%% 产生初始粒子和速度
for i=1:pso_option.sizepop
    % 随机产生种群和速度
    i
    pop(i,1) = (pso_option.popLearnRatemax-pso_option.popLearnRatemin)*rand+pso_option.popLearnRatemin;
    pop(i,2) = (pso_option.popHiddenUnitsmax-pso_option.popHiddenUnitsmin)*rand+pso_option.popHiddenUnitsmin;
    pop(i,3) = (pso_option.popMiniBatchSizemax-pso_option.popMiniBatchSizemin)*rand+pso_option.popMiniBatchSizemin;
        
    V(i,1)=VLearnRatemax*rands(1,1);
    V(i,2)=VHiddenUnitsmax*rands(1,1);
    V(i,3)=VMiniBatchSizemax*rands(1,1);
    
    
    % 计算初始适应度

   %固定随机种子,使其训练结果不变 
setdemorandstream(pi);
% 创建 LSTM 回归网络。
numFeatures = size(train,1);
numResponses = size(train_label,1);
numHiddenUnits = round(pop(i,2));
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
    'MiniBatchSize', round(pop(i,3)),...
    'InitialLearnRate',pop(i,1), ...
    'GradientThreshold',1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.2, ...
    'ExecutionEnvironment','cpu', ...
    'Shuffle','never', ...
    'Verbose',0);
%     'Plots','training-progress',...

%% 训练 LSTM 网络
% 使用 trainNetwork 以指定的训练选项训练 LSTM 网络。
net = trainNetwork(train,train_label,layers,options);

%% 预测将来时间步
% 要预测将来多个时间步的值，请使用 predictAndUpdateState 函数一次预测一个时间步，并在每次预测时更新网络状态。对于每次预测，使用前一次预测作为函数的输入。

YPred = predict(net,P_test);

%使用先前计算的参数对预测去标准化。
YPred = mapminmax('reverse',YPred,outputps);%数据反归一化

fitness(i)=mse(YPred-T_test);%以均方差作为适应度函数，均方差越小，精度越高
end

% 找极值和极值点
[global_fitness bestindex]=min(fitness); % 全局极值
local_fitness=fitness;   % 个体极值初始化

global_x=pop(bestindex,:);   % 全局极值点
local_x=pop;    % 个体极值点初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1,pso_option.maxgen);

%% 迭代寻优
for i=1:pso_option.maxgen
    iter=i
    for j=1:pso_option.sizepop
        
        %速度更新
        V(j,:) = pso_option.wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
        % 边界判断
        if V(j,1) > VLearnRatemax
            V(j,1) = VLearnRatemax;
        end
        if V(j,1) < VLearnRatemin
            V(j,1) = VLearnRatemin;
        end
        if V(j,2) > VHiddenUnitsmax
            V(j,2) = VHiddenUnitsmax;
        end
        if V(j,2) < VHiddenUnitsmin
            V(j,2) = VHiddenUnitsmin;
        end
        if V(j,3) > VMiniBatchSizemax
            V(j,3) = VMiniBatchSizemax;
        end
        if V(j,3) < VMiniBatchSizemin
            V(j,3) = VMiniBatchSizemin;
        end
        
        %种群更新
        pop(j,:)=pop(j,:) + pso_option.wP*V(j,:);
        %边界判断
        if pop(j,1) > pso_option.popLearnRatemax
            pop(j,1) = (pso_option.popLearnRatemax-pso_option.popLearnRatemin)*rand+pso_option.popLearnRatemin;
        end
        if pop(j,1) < pso_option.popLearnRatemin
            pop(j,1) = (pso_option.popLearnRatemax-pso_option.popLearnRatemin)*rand+pso_option.popLearnRatemin;
        end
        if pop(j,2) > pso_option.popHiddenUnitsmax
            pop(j,2) = (pso_option.popHiddenUnitsmax-pso_option.popHiddenUnitsmin)*rand+pso_option.popHiddenUnitsmin;
        end
        if pop(j,2) < pso_option.popHiddenUnitsmin
            pop(j,2) = (pso_option.popHiddenUnitsmax-pso_option.popHiddenUnitsmin)*rand+pso_option.popHiddenUnitsmin;
        end
        if pop(j,3) > pso_option.popMiniBatchSizemax
            pop(j,3) = (pso_option.popMiniBatchSizemax-pso_option.popMiniBatchSizemin)*rand+pso_option.popMiniBatchSizemin;
        end
        if pop(j,3) < pso_option.popMiniBatchSizemin
            pop(j,3) = (pso_option.popMiniBatchSizemax-pso_option.popMiniBatchSizemin)*rand+pso_option.popMiniBatchSizemin;
        end
        
        % 自适应粒子变异
        if rand>0.8
            k=ceil(3*rand);
            if k == 1
                pop(j,k) = (pso_option.popLearnRatemax-pso_option.popLearnRatemin)*rand + pso_option.popLearnRatemin;
            end
            if k == 2
                pop(j,k) = (pso_option.popHiddenUnitsmax-pso_option.popHiddenUnitsmin)*rand + pso_option.popHiddenUnitsmin;
            end
            if k == 3
                pop(j,k) = (pso_option.popMiniBatchSizemax-pso_option.popMiniBatchSizemin)*rand + pso_option.popMiniBatchSizemin;
            end
        end
        
        %适应度值

%固定随机种子,使其训练结果不变 
setdemorandstream(pi);
% 创建 LSTM 回归网络。
numFeatures = size(train,1);
numResponses = size(train_label,1);
numHiddenUnits = round(pop(j,2));
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
    'MiniBatchSize', round(pop(j,3)),...
    'InitialLearnRate',pop(j,1), ...
    'GradientThreshold',1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',30, ...
    'LearnRateDropFactor',0.2, ...
    'ExecutionEnvironment','cpu', ...
    'Shuffle','never', ...
    'Verbose',0);
%     'Plots','training-progress',...

%% 训练 LSTM 网络
% 使用 trainNetwork 以指定的训练选项训练 LSTM 网络。
net = trainNetwork(train,train_label,layers,options);

%% 预测将来时间步

YPred = predict(net,P_test);

%使用先前计算的参数对预测去标准化。
YPred = mapminmax('reverse',YPred,outputps);%数据反归一化

fitness(j)=mse(YPred-T_test);%以均方差作为适应度函数，均方差越小，精度越高
        
        %个体最优更新
        if fitness(j) < local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        %群体最优更新
        if fitness(j) < global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
             %保存网络
            save ('net');
        end
        
                %----------------惯性权重改变--------------------------
        wfitness=fitness;
        [m,n]=find(isnan(wfitness)==1);
        wfitness(:,n)=[];
                fmin=min(wfitness);
        favg=mean(wfitness);
        if fitness(j)<=favg&&~isnan(fitness(j))
           pso_option.wV = wmin+(wmax-wmin)*(fitness(j)-fmin)/(favg-fmin);
           if isnan(pso_option.wV)
               pso_option.wV=wmax;
           end
        else
             pso_option.wV=wmax;
        end
        %-------------------惯性权重改变
        
    end
    
    fit_gen(i)=global_fitness;
    avgfitness_gen(i) = sum(fitness)/pso_option.sizepop;
end

%% 输出结果
% 最好的参数
bestLearnRate = global_x(1);
bestHiddenUnits = global_x(2);
bestMiniBatchSize = global_x(3);
bestCVmse = fit_gen(pso_option.maxgen);%最好的结果

