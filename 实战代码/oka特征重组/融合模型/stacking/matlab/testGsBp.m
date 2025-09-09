%% 清空环境变量
warning off % 关闭报警信息
close all % 关闭开启的图窗
clear % 清空变量
clc % 清空命令行


%% 设置随机种子
%rng(100); % 在这里使用任何整数，100只是一个示例种子值
%rng(50); % 在这里使用任何整数，100只是一个示例种子值
rng(0);




%% 导入数据
x_train_data=xlsread('../../x_train_.xlsx');
y_train_data=xlsread('../../y_train.xlsx');
x_test_data=xlsread('../../x_test_.xlsx');
y_test_data=xlsread('../../y_test.xlsx');



%%  划分训练集和测试集
train_feature_num=7;



x_train = x_train_data(1: end, 1: train_feature_num)';
y_train = y_train_data(1: end, 1)';
M = size(x_train, 2);

x_test = x_test_data(1: end, 1: train_feature_num)';
y_test = y_test_data(1: end, 1)';
N = size(x_test, 2);


%% 数据归一化

[x_train_MinMaxScaler,ps_input]=mapminmax(x_train,0,1)
[y_train_MinMaxScaler,ps_output]=mapminmax(y_train,0,1)

x_test_MinMaxScaler=mapminmax('apply',x_test,ps_input)
y_test_MinMaxScaler=mapminmax('apply',y_test,ps_output)

























%% 创建网络---hiddennum、learning_rate先随便写个，之后gs会找最优
% sqrt(8+1)+1~10=4~13
hiddennum=13; %隐藏层结点个数
learning_rate=0.001;

net=newff(x_train_MinMaxScaler,y_train_MinMaxScaler,hiddennum,{'tansig','purelin'},'trainlm');

%% 参数设置
% 有空看下这段代码的意思？？？？？？
% 那怎么知道最优参数,怎么调参数？？
net.trainParam.epochs=1000; % % % % % % 最大迭代次数
net.trainParam.goal=1e-6;% % % % % % % 目标训练误差
net.trainParam.lr=0.001;% 学习率

%% 训练网络
net=train(net,x_train_MinMaxScaler,y_train_MinMaxScaler);


%{
%% 网格搜索，通过网格搜索方法来找最优的bpnn的超参数
% 定义超参数网格
%learning_rates = [0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]; % 学习率
%learning_rates = linspace(0,1,100); % 学习率
learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]; % 学习率
hidden_layer_sizes = [4,5,6,7,8,9,10,11,12,13]; % 隐藏层节点数
param_grid = cell(numel(learning_rates), numel(hidden_layer_sizes));



for i = 1:numel(learning_rates)
    for j = 1:numel(hidden_layer_sizes)
        param_grid{i,j} = struct('learning_rate', learning_rates(i), 'hidden_layer_size', hidden_layer_sizes(j));
    end
end




% 网格搜索
best_R2 = -inf;
best_params = struct();
best_net=net;
best_hiddennum=hiddennum;
best_learning_rate=learning_rate;
for i = 1:numel(learning_rates)
    for j = 1:numel(hidden_layer_sizes)
        params = param_grid{i,j};
        
        
        %net = trainNetwork(x_train_MinMaxScaler, y_train_MinMaxScaler, params.hidden_layer_size, 'learners', 'bp', 'LearnRate', params.learning_rate);
        % 使用交叉验证评估模型性能（均方误差）
        %y_pred = mapminmax('reverse', predict(net, x_test_MinMaxScaler), ps_output);
        
        net=newff(x_train_MinMaxScaler,y_train_MinMaxScaler,params.hidden_layer_size,{'tansig','purelin'},'trainlm');
        net.trainParam.epochs=1000; % % % % % % 最大迭代次数
        net.trainParam.goal=1e-6;% % % % % % % 目标训练误差
        net.trainParam.lr=params.learning_rate;% 学习率
        
        %setdemorandstream(pi);
        
        net=train(net,x_train_MinMaxScaler,y_train_MinMaxScaler);
        y_pred_MinMaxScaler=sim(net,x_test_MinMaxScaler);
        y_pred=mapminmax('reverse',y_pred_MinMaxScaler,ps_output);
        
        %{
        mse = mean((y_pred - y_test).^2);
        if mse < best_mse
            best_mse = mse;
            best_params = params;
        end
        %}
        ssres=sum((y_pred-y_test).^2);
        sstotal=sum((y_test-mean(y_test)).^2);
        R2=1-ssres/sstotal;
        if R2 > best_R2
            best_R2 = R2;
            best_params = params;
            best_hiddennum=params.hidden_layer_size;
            best_learning_rate=params.learning_rate;
            best_net = net; % 保存最优模型
            best_y_pred = y_pred; % 保存最优模型的预测值
        end
        
    end
end


% 打印最佳超参数组合和性能
disp('-----------------------------------------------------');
disp(['Best_hiddennum: ', num2str(best_hiddennum)]);
disp(['Best_learning_rate: ', num2str(best_learning_rate)]);

disp('Best parameters:');
disp(best_params);
disp(['Best R2: ', num2str(best_R2)]);


% 输出最优模型的预测值
disp('Best model predictions:');
disp(best_y_pred);


%save('save_bp_gs_best_net.mat','best_net')




%% 通过网格搜索方法找到的最优的bpnn的超参数
%{

Best_hiddennum: 9
Best_learning_rate: 0.3
Best parameters:
        learning_rate: 0.3000
    hidden_layer_size: 9

Best R2: 0.6746
Best model predictions:
  列 1 至 15

    0.3289   -0.0044   -0.0005    0.0906   -0.0391    0.1798    0.2614    0.0593    0.0030    0.0105   -0.0027   -0.0009   -0.0026   -0.0020   -0.0084

  列 16 至 20

    0.2457    0.0308   -0.0048    0.3310   -0.0007
%}



%%
%load('-mat','save_bp_gs_best_net')
%load('save_bp_gs_best_net.mat');
%}



%% 仿真测试
%y_pred_MinMaxScaler=sim(best_net,x_test_MinMaxScaler);
y_pred_MinMaxScaler=sim(net,x_test_MinMaxScaler);

%% 数据反归一化
y_pred=mapminmax('reverse',y_pred_MinMaxScaler,ps_output);
disp('-----------------------------------------------------');
disp(['y_test: ', num2str(y_test)]);
disp(['y_pred: ', num2str(y_pred)]);

%{
-----------------------------------------------------
y_test: 0.25428  0.00025224  3.4151e-05    0.038532  0.00022308      0.2184     0.37206    0.032994  0.00015855  2.0691e-06  3.3087e-05  8.7273e-05  9.7855e-06  2.1586e-05  4.8183e-05     0.23556     0.02808  0.00026017      0.1014  0.00041043
y_pred: 0.32895  -0.0043525 -0.00052696    0.090621   -0.039116     0.17984     0.26139    0.059262   0.0029893    0.010543  -0.0027134 -0.00085255  -0.0026125  -0.0020434  -0.0084367      0.2457    0.030836  -0.0048183     0.33096 -0.00073694
-----------------------------------------------------
%}

%% 性能评价
MAE=mean(abs(y_pred-y_test));
MSE=mean((y_pred-y_test).^2);
RMSE=sqrt(MSE);

ssres=sum((y_pred-y_test).^2);
sstotal=sum((y_test-mean(y_test)).^2);
R2=1-ssres/sstotal;
RelativeErro=mean(abs((y_pred-y_test)./y_test))*100;

disp('-----------------------------------------------------');
disp(['平均绝对误差 (MAE): ', num2str(MAE)]);
disp(['均方误差 (MSE): ', num2str(MSE)]);
disp(['均方根误差 (RMSE): ', num2str(RMSE)]);
disp(['决定系数 (R2): ', num2str(R2)]);
disp(['相对误差(Relative Erro,%): ', num2str(RelativeErro)]);

% 计算Explanatory Variance
ev = explanatoryVariance(y_test, y_pred);
disp(['Explanatory Variance: ', num2str(ev)]);



%{
平均绝对误差 (MAE): 0.031283
均方误差 (MSE): 0.0038663
均方根误差 (RMSE): 0.062179
决定系数 (R2): 0.6746
相对误差(Relative Erro,%): 29923.0571
Explanatory Variance: 0.68236
%}



%% 绘图
figure
plot(1:N,y_test,'r-*',1:N,y_pred,'b-o','LineWidth',1)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比'; ['RMSE=' num2str(RMSE) '%' ]};
title(string)
xlim([1,N])
grid


function ev = explanatoryVariance(y_true, y_pred)
    % 计算Residual Variance
    residual_variance = var(y_true - y_pred);

    % 计算Total Variance
    total_variance = var(y_true);

    % 计算Explanatory Variance
    ev = 1 - residual_variance / total_variance;
end

