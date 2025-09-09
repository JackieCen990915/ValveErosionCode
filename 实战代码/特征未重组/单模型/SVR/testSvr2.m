%% 清空环境变量
warning off % 关闭报警信息
close all % 关闭开启的图窗
clear % 清空变量
clc % 清空命令行


%% 设置随机种子
%rng(100); % 在这里使用任何整数，100只是一个示例种子值
rng(0); % 在这里使用任何整数，100只是一个示例种子值


%% 导入数据
x_train_data=xlsread('../x_train_.xlsx');
y_train_data=xlsread('../y_train.xlsx');
x_test_data=xlsread('../x_test_.xlsx');
y_test_data=xlsread('../y_test.xlsx');


%%  划分训练集和测试集
train_feature_num=5;

P_train = x_train_data(1: end, 1: train_feature_num)';
T_train = y_train_data(1: end, 1)';
M = size(P_train, 2);

P_test = x_test_data(1: end, 1: train_feature_num)';
T_test = y_test_data(1: end, 1)';
N = size(P_test, 2);


%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  转置以适应模型
p_train = p_train'; 
p_test = p_test';
t_train = t_train'; 
t_test = t_test';

%%  创建模型

c = 4.0;    % 惩罚因子
g = 0.8;    % 径向基函数参数
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];

%{
c = 0.8;    % 惩罚因子
g = 1;    % 径向基函数参数
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
%}
model = svmtrain(t_train, p_train, cmd);

%%  仿真预测
[t_sim1, error_1] = svmpredict(t_train, p_train, model);
[t_sim2, error_2] = svmpredict(t_test , p_test , model);

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  测试集的相关指标计算
% MAE
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;
disp(['测试集数据的MAE为：', num2str(mae2)])
% MSE
mse2 = sum((T_sim2' - T_test).^2) ./ N ;
disp(['测试集数据的MSE为：', num2str(mse2)])
% RMSE
rmse2 = sqrt(mse2) ;
disp(['测试集数据的RMSE为：', num2str(rmse2)])
% MAPE(%)
MAPE2 = (100 / N) * sum(abs((T_test - T_sim2')./ T_test));
disp(['测试集数据的MAPE(%)为：', num2str(MAPE2)])
% R2
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;
disp(['测试集数据的R2为：', num2str(R2)])
% EV
EV2 = 1 - mse2 / var(T_test);
disp(['测试集数据的EV为：', num2str(EV2)])

%{
matlab:
Mean squared error = 0.0367092 (regression)
Squared correlation coefficient = 0.542045 (regression)
Mean squared error = 0.00970182 (regression)
Squared correlation coefficient = 0.678926 (regression)
测试集数据的MAE为：0.030694
测试集数据的MSE为：0.0040957
测试集数据的RMSE为：0.063998
测试集数据的MAPE(%)为：11912.5748
测试集数据的R2为：0.65529
测试集数据的EV为：0.67252----------------------

python:
MAE: 0.03069352953336251
MSE: 0.004095740632355003
RMSE: 0.06399797365819486
MAPE: 11912.574790362374%
R2: 0.6552891711002156
EV: 0.6554299930700114--------------------------------
%}


%{
% MSE
mse1 = sum((T_sim1' - T_train).^2) ./ M ;
mse2 = sum((T_sim2' - T_test).^2) ./ N ;
disp(['训练集数据的MSE为：', num2str(mse1)])
disp(['测试集数据的MSE为：', num2str(mse2)])

% RMSE
rmse1 = sqrt(mse1) ;
rmse2 = sqrt(mse2) ;
disp(['训练集数据的RMSE为：', num2str(rmse1)])
disp(['测试集数据的RMSE为：', num2str(rmse2)])


% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;
disp(['训练集数据的MAE为：', num2str(mae1)])
disp(['测试集数据的MAE为：', num2str(mae2)])


% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;
disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])


% EV
EV1 = 1 - mse1 / var(T_train);
EV2 = 1 - mse2 / var(T_test);
disp(['训练集数据的EV为：', num2str(EV1)])
disp(['测试集数据的EV为：', num2str(EV2)])
%}

%{
%%  均方根误差mse
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid
%}

%{
%%  绘制散点图
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('训练集真实值');
ylabel('训练集预测值');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('训练集预测值 vs. 训练集真实值')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('测试集真实值');
ylabel('测试集预测值');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('测试集预测值 vs. 测试集真实值')
%}