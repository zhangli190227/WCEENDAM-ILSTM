function [bestCVmse,bestLearnRate,bestHiddenUnits,bestMiniBatchSize,fit_gen] = ampsolstmForRegress(train_label,train,T_test,P_test,outputps,pso_option)

%% ������ʼ��
% c1:pso�����ֲ���������
% c2:pso����ȫ����������
% maxgen:����������
% sizepop:��Ⱥ�������
% k:k belongs to [0.1,1.0],���ʺ�x�Ĺ�ϵ(V = kX)
% wV:(wV best belongs to [0.8,1.2]),���ʸ��¹�ʽ���ٶ�ǰ��ĵ���ϵ��
% wP:��Ⱥ���¹�ʽ���ٶ�ǰ��ĵ���ϵ��

% popLearnRatemax:LSTM ����ѧϰ��LearnRate�ı仯�����ֵ.
% popLearnRatemin:LSTM ����ѧϰ��LearnRate�ı仯����Сֵ.

% popHiddenUnitsmax:LSTM �������ز�ڵ�HiddenUnits�ı仯�����ֵ.
% popHiddenUnitsmin:LSTM �������ز�ڵ�HiddenUnits�ı仯����Сֵ.
% popMiniBatchSizemax:LSTM ����������MiniBatchSize�ı仯�����ֵ.
% popMiniBatchSizemin:LSTM ����������MiniBatchSize�ı仯����Сֵ.

VLearnRatemax = pso_option.k*pso_option.popLearnRatemax;
VLearnRatemin = -VLearnRatemax ;
VHiddenUnitsmax = pso_option.k*pso_option.popHiddenUnitsmax;
VHiddenUnitsmin = -VHiddenUnitsmax ;
VMiniBatchSizemax = pso_option.k*pso_option.popMiniBatchSizemax;
VMiniBatchSizemin = -VMiniBatchSizemax ;

wmax=0.9;wmin=0.6;
% [row,col]=size(P_test);
% P_test = P_test';
%% ������ʼ���Ӻ��ٶ�
for i=1:pso_option.sizepop
    % ���������Ⱥ���ٶ�
    i
    pop(i,1) = (pso_option.popLearnRatemax-pso_option.popLearnRatemin)*rand+pso_option.popLearnRatemin;
    pop(i,2) = (pso_option.popHiddenUnitsmax-pso_option.popHiddenUnitsmin)*rand+pso_option.popHiddenUnitsmin;
    pop(i,3) = (pso_option.popMiniBatchSizemax-pso_option.popMiniBatchSizemin)*rand+pso_option.popMiniBatchSizemin;
        
    V(i,1)=VLearnRatemax*rands(1,1);
    V(i,2)=VHiddenUnitsmax*rands(1,1);
    V(i,3)=VMiniBatchSizemax*rands(1,1);
    
    
    % �����ʼ��Ӧ��

   %�̶��������,ʹ��ѵ��������� 
setdemorandstream(pi);
% ���� LSTM �ع����硣
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

% % ָ��ѵ��ѡ�ʹ������� 'adam' �Դ�СΪ 20 ��С�������� 60 ��ѵ����ָ��ѧϰ��Ϊ 0.01��Ҫ��ֹ�ݶȱ�ը���뽫�ݶ���ֵ����Ϊ 1��Ҫʹ���б��ְ����������뽫 'Shuffle' ����Ϊ 'never'��
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

%% ѵ�� LSTM ����
% ʹ�� trainNetwork ��ָ����ѵ��ѡ��ѵ�� LSTM ���硣
net = trainNetwork(train,train_label,layers,options);

%% Ԥ�⽫��ʱ�䲽
% ҪԤ�⽫�����ʱ�䲽��ֵ����ʹ�� predictAndUpdateState ����һ��Ԥ��һ��ʱ�䲽������ÿ��Ԥ��ʱ��������״̬������ÿ��Ԥ�⣬ʹ��ǰһ��Ԥ����Ϊ���������롣

YPred = predict(net,P_test);

%ʹ����ǰ����Ĳ�����Ԥ��ȥ��׼����
YPred = mapminmax('reverse',YPred,outputps);%���ݷ���һ��

fitness(i)=mse(YPred-T_test);%�Ծ�������Ϊ��Ӧ�Ⱥ�����������ԽС������Խ��
end

% �Ҽ�ֵ�ͼ�ֵ��
[global_fitness bestindex]=min(fitness); % ȫ�ּ�ֵ
local_fitness=fitness;   % ���弫ֵ��ʼ��

global_x=pop(bestindex,:);   % ȫ�ּ�ֵ��
local_x=pop;    % ���弫ֵ���ʼ��

% ÿһ����Ⱥ��ƽ����Ӧ��
avgfitness_gen = zeros(1,pso_option.maxgen);

%% ����Ѱ��
for i=1:pso_option.maxgen
    iter=i
    for j=1:pso_option.sizepop
        
        %�ٶȸ���
        V(j,:) = pso_option.wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
        % �߽��ж�
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
        
        %��Ⱥ����
        pop(j,:)=pop(j,:) + pso_option.wP*V(j,:);
        %�߽��ж�
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
        
        % ����Ӧ���ӱ���
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
        
        %��Ӧ��ֵ

%�̶��������,ʹ��ѵ��������� 
setdemorandstream(pi);
% ���� LSTM �ع����硣
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

% % ָ��ѵ��ѡ�ʹ������� 'adam' �Դ�СΪ 20 ��С�������� 60 ��ѵ����ָ��ѧϰ��Ϊ 0.01��Ҫ��ֹ�ݶȱ�ը���뽫�ݶ���ֵ����Ϊ 1��Ҫʹ���б��ְ����������뽫 'Shuffle' ����Ϊ 'never'��
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

%% ѵ�� LSTM ����
% ʹ�� trainNetwork ��ָ����ѵ��ѡ��ѵ�� LSTM ���硣
net = trainNetwork(train,train_label,layers,options);

%% Ԥ�⽫��ʱ�䲽

YPred = predict(net,P_test);

%ʹ����ǰ����Ĳ�����Ԥ��ȥ��׼����
YPred = mapminmax('reverse',YPred,outputps);%���ݷ���һ��

fitness(j)=mse(YPred-T_test);%�Ծ�������Ϊ��Ӧ�Ⱥ�����������ԽС������Խ��
        
        %�������Ÿ���
        if fitness(j) < local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        %Ⱥ�����Ÿ���
        if fitness(j) < global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
             %��������
            save ('net');
        end
        
                %----------------����Ȩ�ظı�--------------------------
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
        %-------------------����Ȩ�ظı�
        
    end
    
    fit_gen(i)=global_fitness;
    avgfitness_gen(i) = sum(fitness)/pso_option.sizepop;
end

%% ������
% ��õĲ���
bestLearnRate = global_x(1);
bestHiddenUnits = global_x(2);
bestMiniBatchSize = global_x(3);
bestCVmse = fit_gen(pso_option.maxgen);%��õĽ��

