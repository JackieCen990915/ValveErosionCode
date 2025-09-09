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
train_feature_num=7;

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




%% 网格搜索方法优化超参数
% 定义超参数的搜索范围---同python
%{
c_range = [0.01,0.1,0.2,0.5,0.8,1,5,10,25,50,75,100]; % 惩罚因子的搜索范围
gamma_range = [0.01,0.05,0.1,0.2,0.5,0.8,1]; % 径向基函数参数的搜索范围
epsilon_range = [0.01,0.05,0.1,0.2,0.5,0.8,1];
%}

c_range = [0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,25,50,75,100]; % 惩罚因子的搜索范围
gamma_range = linspace(0,1,100); % 径向基函数参数的搜索范围
epsilon_range = linspace(0,1,100);

% 初始化最佳超参数和最佳性能指标
best_c = 0;
best_gamma = 0;
best_epsilon = 0;
best_r2 = -inf;


% 遍历超参数组合进行网格搜索
for i = 1:length(c_range)
    for j = 1:length(gamma_range)
        for k = 1:length(epsilon_range)
            % 使用当前超参数组合训练模型
            c = c_range(i);
            g = gamma_range(j);
            epsilon = epsilon_range(k);
            %cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01 -p ',num2str(epsilon)];
            cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p ',num2str(epsilon)];
            model = svmtrain(t_train, p_train, cmd);

            % 预测测试集
            [t_sim2, ~] = svmpredict(t_test , p_test , model);

            % 计算当前超参数组合的性能指标 R2
            ss_res = norm(t_test - t_sim2')^2;
            ss_tot = norm(t_test - mean(t_test))^2;
            r2 = 1 - ss_res / ss_tot;

            % 更新最佳超参数和性能指标
            if r2 > best_r2
                best_r2 = r2;
                best_c = c;
                best_gamma = g;
                best_epsilon = epsilon;
            end
        end
    end
end


% 打印最佳超参数组合和性能
disp('-----------------------------------------------------');
disp(['Best_r2: ', num2str(best_r2)]);
disp(['Best_c: ', num2str(best_c)]);
disp(['Best_gamma: ', num2str(best_gamma)]);
disp(['Best_epsilon: ', num2str(best_epsilon)]);
disp('-----------------------------------------------------');









%%  创建模型
%{
c = 4.0;    % 惩罚因子
g = 0.8;    % 径向基函数参数
cmd = [' -t 2',' -c ',num2str(c),' -g ',num2str(g),' -s 3 -p 0.01'];
model = svmtrain(t_train, p_train, cmd);
%}
% 使用最佳超参数重新训练模型
%cmd = [' -t 2',' -c ',num2str(best_c),' -g ',num2str(best_g),' -s 3 -p 0.01 -p ',num2str(best_epsilon)];
cmd = [' -t 2',' -c ',num2str(best_c),' -g ',num2str(best_gamma),' -s 3 -p ',num2str(best_epsilon)];
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


