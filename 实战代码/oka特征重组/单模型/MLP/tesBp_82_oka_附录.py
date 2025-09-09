import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 

import shap

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


k=5
#隐藏层节点个数----4-13

#隐藏层节点个数----5-14----sqrt(1+11)+1-10
param_grid={
        'hidden_layer_sizes':np.arange(5,15,1),
        'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
        'activation':['sigmoid','tanh','relu'],
}

'''
param_grid={
        'alpha':[0.01,0.05,0.1],
        'hidden_layer_sizes':np.arange(4,14,1),
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],
}
'''

'''
param_grid={
        'hidden_layer_sizes':np.arange(4,14,1),
        'max_iter':np.arange(1,501,100),
        'alpha':np.linspace(0.1,1,10)
}
'''
'''
param_grid={
        'hidden_layer_sizes':np.arange(4,13,1),
        'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],
}
'''
'''
param_grid={
        'hidden_layer_sizes':np.arange(4,13,1),
        'alpha':[0.01,0.05,0.1],
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],
}
'''
'''
param_grid={
        'hidden_layer_sizes':np.arange(4,13,1),
        'max_iter':np.arange(1,501,50),
}
'''
        
regressor=MLPRegressor(
                        solver='lbfgs',
                        random_state=90)



GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)


regressor=GS.best_estimator_
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

'''
{'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 6}
0.6961603545220384
MAE_train: 0.002400368471547203
MSE_train: 9.57541254866315e-06
RMSE_train: 0.0030944163502449294
MAPE_train: 7.833874907076711
r2_score_train: 0.8179420726794311
EV_train: 0.8179577271639531
-------------------
MAE_test: 0.002417320793805746
MSE_test: 9.847258297846715e-06
RMSE_test: 0.003138034145423965
MAPE_test: 1.4576851327713223
r2_score_test: 0.8233402686386865
EV_test: 0.8271528368600944
-------------------

'''

#---42,90---数据集加入新的重构特征-----'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
'''

{'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': 6, 'max_iter': 51}
0.6961603545220384
MAE_train: 0.002400368471547203
MSE_train: 9.57541254866315e-06
RMSE_train: 0.0030944163502449294
MAPE_train: 7.833874907076711
r2_score_train: 0.8179420726794311
EV_train: 0.8179577271639531
-------------------
MAE_test: 0.002417320793805746
MSE_test: 9.847258297846715e-06
RMSE_test: 0.003138034145423965
MAPE_test: 1.4576851327713223
r2_score_test: 0.8233402686386865
EV_test: 0.8271528368600944
-------------------
'''


