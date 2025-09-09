import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 



#导入差分进化算法
from sko.DE import DE


import datetime

# 记录开始时间
start_time = datetime.datetime.now()



# 定义激活函数的候选列表
activation_functions = ['tanh','relu']


######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx',sheet_name='Sheet2')





######################2.提取特征变量######################
x=df.drop(columns='er')
y=df['er']


print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))




######################3.划分训练集和测试集######################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)




# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



'''
param_grid={
        'alpha':[0.01,0.05,0.1],
        'hidden_layer_sizes':np.arange(4,14,1),
        'max_iter':np.arange(1,501,50),
        
}
'''





# 按照这个格式，改成和网格搜索一样的参数？
'''
def train_MLP(x):
    alpha, hidden_layer_sizes, activation_index = x
    activation = activation_functions[int(activation_index)]
    clf = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=int(hidden_layer_sizes),
                        activation=activation,
                        solver='lbfgs',
                        random_state=90)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test, y_pred)
    return -score
'''


def train_MLP(x):
    alpha,hidden_layer_sizes= x
    clf = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=int(hidden_layer_sizes),
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score

'''
def train_MLP(x):
    #alpha,hidden_layer_sizes,max_iter = x
    alpha,hidden_layer_sizes= x
    clf = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=int(hidden_layer_sizes),
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score

'''

'''
def train_MLP(x):
    alpha,hidden_layer_sizes,max_iter = x
    clf = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=int(hidden_layer_sizes),
                        max_iter=int(max_iter),
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''

#A_DE = DE(func=train_MLP, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.7,lb=[0, 5, 0], ub=[1, 14, len(activation_functions) - 1])
A_DE = DE(func=train_MLP, n_dim=2, size_pop=200, max_iter=400,prob_mut=0.7,lb=[0,4], ub=[1,13])
#A_DE = DE(func=train_MLP, n_dim=1, size_pop=200, max_iter=400,prob_mut=0.7,lb=[0], ub=[1])
#A_DE = DE(func=train_MLP, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.7,lb=[0,4,1], ub=[1,13,500])
#200-max_iter用来指定神经网络的最大迭代次数,默认值为200

best_x, best_y = A_DE.run()#运行算法
print('best_x:', best_x, '\n', 'best_y:', best_y)




alpha = best_x[0]
hidden_layer_sizes = int(best_x[1])
#activation_index = int(best_x[2])
#activation = activation_functions[activation_index]

print('best_x[0]:', alpha)
print('best_x[1]:', hidden_layer_sizes)
#print('best_x[2]_activation:', activation)
'''
regressor = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver='lbfgs',
                        random_state=90)
'''


regressor=MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)





'''
regressor=MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
'''
'''
regressor=MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
'''
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)



######################6.评估模型(训练集)######################
MAE_train=metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
MAPE_train=metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train=metrics.r2_score(y_train, y_train_pred)
EV_train=metrics.explained_variance_score(y_train, y_train_pred)


print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('MAPE_train:', MAPE_train)
print('r2_score_train:', R2_train)
print('EV_train:', EV_train)
print('-------------------')


######################6.评估模型(测试集)######################
MAE_test=metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test=metrics.mean_squared_error(y_test, y_test_pred)
RMSE_test=np.sqrt(MSE_test)
MAPE_test=metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test=metrics.r2_score(y_test, y_test_pred)
EV_test=metrics.explained_variance_score(y_test, y_test_pred)


print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('MAPE_test:', MAPE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)
print('-------------------')


# 记录结束时间
end_time = datetime.datetime.now()
 
# 计算运行时长
duration = end_time - start_time
 
# 打印运行时长
print("Duration is ",duration)


'''
best_x: [5.42379845e-03 1.17712036e+01 7.82762779e-01] 
 best_y: [-0.96353674]
best_x[0]: 0.005423798452377173
best_x[1]: 11
best_x[2]_activation: tanh
MAE_train: 0.00173568112787501
MSE_train: 5.075867360304051e-06
RMSE_train: 0.002252968566204165
MAPE_train: 3.421241385368205
r2_score_train: 0.9034922112990214
EV_train: 0.9035370437728483
-------------------
MAE_test: 0.0009340685061996502
MSE_test: 2.0325128831177446e-06
RMSE_test: 0.0014256622612378238
MAPE_test: 1.5223944058206396
r2_score_test: 0.963536735905617
EV_test: 0.9662958838721204
-------------------
Duration is  0:51:16.041442

'''








'''
ga跑出来，给de用吧----de太慢了
best_x: [2.41090707e-03 5.59626350e+00] 
 best_y: 0.9547402647804266
best_x[0]: 0.0024109070656361817
best_x[1]: 5

MAE_train: 0.013366820534293706
MSE_train: 0.0007489706787205963
RMSE_train: 0.027367328673449227
MAPE_train: 888.3778589630825
r2_score_train: 0.9773989983969095


MAE_test: 0.014603444525327285
MSE_test: 0.0007496159129887115
RMSE_test: 0.027379114539895395
MAPE_test: 212.68772254038964
r2_score_test: 0.9547402647804266
EV_test: 0.9623725425431697
'''

