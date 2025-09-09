%% 清空环境变量
warning off % 关闭报警信息
close all % 关闭开启的图窗
clear % 清空变量
clc % 清空命令行


%% 设置随机种子
%rng(100); % 在这里使用任何整数，100只是一个示例种子值
%rng(50); % 在这里使用任何整数，100只是一个示例种子值
rng(0);



%% 
% 导入数据
x_train_data=xlsread('x_train_.xlsx');
y_train_data=xlsread('y_train.xlsx');
x_test_data=xlsread('x_test_.xlsx');
y_test_data=xlsread('y_test.xlsx');
% 划分训练集和测试集
train_feature_num=7;
x_train = x_train_data(1: end, 1: train_feature_num)';
y_train = y_train_data(1: end, 1)';
M = size(x_train, 2);
x_test = x_test_data(1: end, 1: train_feature_num)';
y_test = y_test_data(1: end, 1)';
N = size(x_test, 2);
% 数据归一化
[x_train_MinMaxScaler,ps_input]=mapminmax(x_train,0,1);
[y_train_MinMaxScaler,ps_output]=mapminmax(y_train,0,1);
x_test_MinMaxScaler=mapminmax('apply',x_test,ps_input);
y_test_MinMaxScaler=mapminmax('apply',y_test,ps_output);



%% svr
c = 4.0;    % 惩罚因子
g = 0.8;    % 径向基函数参数
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
model = svmtrain(y_train_MinMaxScaler', x_train_MinMaxScaler', cmd);
% 仿真预测
[t_sim1, error_1] = svmpredict(y_train_MinMaxScaler', x_train_MinMaxScaler', model);
[t_sim2, error_2] = svmpredict(y_test_MinMaxScaler', x_test_MinMaxScaler', model);
%{
% 数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim1 = T_sim1';

T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim2 = T_sim2';
%}


%% bp
% 创建bp网络---hiddennum、learning_rate
hiddennum=7; %隐藏层结点个数
net=newff(x_train_MinMaxScaler,y_train_MinMaxScaler,hiddennum,{'tansig','purelin'},'trainlm');
net.trainParam.epochs=1000; % % % % % % 最大迭代次数
net.trainParam.goal=1e-6;% % % % % % % 目标训练误差
net.trainParam.lr=0.001;% 学习率
% 训练bp网络
net=train(net,x_train_MinMaxScaler,y_train_MinMaxScaler);
% 仿真测试
y_train_pred_MinMaxScaler=sim(net,x_train_MinMaxScaler);
y_test_pred_MinMaxScaler=sim(net,x_test_MinMaxScaler);
%{
% 数据反归一化
y_train_pred=mapminmax('reverse',y_train_pred_MinMaxScaler,ps_output);
y_test_pred=mapminmax('reverse',y_test_pred_MinMaxScaler,ps_output);
%}


%%
stacking_features = [t_sim1, y_train_pred_MinMaxScaler'];
meta_model = fitlm(stacking_features, y_train_MinMaxScaler);
stacking_test_features = [t_sim2, y_test_pred_MinMaxScaler'];
final_prediction_MinMaxScaler = predict(meta_model, stacking_test_features);
%disp('final_prediction_MinMaxScaler:');
%disp(final_prediction_MinMaxScaler);

% 数据反归一化
final_prediction=mapminmax('reverse',final_prediction_MinMaxScaler,ps_output);
disp(final_prediction);

%{
MAE: 0.03991700819824389
MSE: 0.005790047905890963
RMSE: 0.07609236430740579
MAPE: 52760.075322201774%
R2: 0.512690770201061
EV: 0.5940687148361891
%}
%{
%% 
% 数据反归一化
final_predictions=mapminmax('reverse',final_prediction_MinMaxScaler,ps_output);



% 性能评价
MAE=mean(abs(final_predictions-y_test));
MSE=mean((final_predictions-y_test).^2);
RMSE=sqrt(MSE);

ssres=sum((final_predictions-y_test).^2);
sstotal=sum((y_test-mean(y_test)).^2);
R2=1-ssres/sstotal;
RelativeErro=mean(abs((final_predictions-y_test)./y_test))*100;

disp('-----------------------------------------------------');
disp(['平均绝对误差 (MAE): ', num2str(MAE)]);
disp(['均方误差 (MSE): ', num2str(MSE)]);
disp(['均方根误差 (RMSE): ', num2str(RMSE)]);
disp(['决定系数 (R2): ', num2str(R2)]);
disp(['相对误差(Relative Erro,%): ', num2str(RelativeErro)]);

% 计算Explanatory Variance
ev = explanatoryVariance(y_test, final_predictions);
disp(['Explanatory Variance: ', num2str(ev)]);


%% 绘图

function ev = explanatoryVariance(y_true, final_predictions)
    % 计算Residual Variance
    residual_variance = var(y_true - final_predictions);

    % 计算Total Variance
    total_variance = var(y_true);

    % 计算Explanatory Variance
    ev = 1 - residual_variance / total_variance;
end
%}


%%
%{
% 准备训练数据和目标
% 这里假设你有训练数据 X_train 和相应的目标 y_train

% 使用 SVR 模型进行预测
svr_pred = predict(svrModel, X_train);

% 使用 BP 模型进行预测
bp_pred = predict(bpModel, X_train);

% 将 SVR 和 BP 的预测结果作为新的特征
stacking_features = [svr_pred, bp_pred];

% 定义元模型（这里使用简单的线性回归）
meta_model = fitlm(stacking_features, y_train);

% 输出元模型的系数
disp('Meta-model coefficients:');
disp(meta_model.Coefficients);
%}


%{
% 准备测试数据
% 这里假设你有测试数据 X_test

% 使用 SVR 模型进行测试集预测
svr_test_pred = predict(svrModel, X_test);

% 使用 BP 模型进行测试集预测
bp_test_pred = predict(bpModel, X_test);

% 将测试集预测结果作为新的特征
stacking_test_features = [svr_test_pred, bp_test_pred];

% 使用元模型进行最终的预测
final_prediction = predict(meta_model, stacking_test_features);

% 输出最终预测结果
disp('Final predictions:');
disp(final_prediction);
%}
