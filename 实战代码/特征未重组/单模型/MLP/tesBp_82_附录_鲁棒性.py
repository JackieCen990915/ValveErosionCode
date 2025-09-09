import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 


######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/弯头冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand.xlsx',sheet_name='Sheet1_er_kg')





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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)



##### 将10%(6个)异常值添加到训练集#####################################
'''
random_selected_train_index=[19, 25, 9, 61, 32, 38]
random_multiple=[3.26035398,1.78945897,3.71130498,3.8388549,3.95684917,4.00810651]
'''

random_selected_train_index=[28, 29, 36, 15, 32, 69, 44, 54, 40, 64, 57, 10]
random_multiple=[3.77406847,2.3163381,2.39277276,1.94249484,3.8442951
 ,2.07089691
 ,4.48338812
 ,1.88319388
 ,3.33209855
 ,4.80352563
 ,3.50921626
 ,3.73180566]

y_train[random_selected_train_index]= y_train[random_selected_train_index]*random_multiple



# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


k=5

param_grid={
        'alpha':[0.01,0.05,0.1],
        'hidden_layer_sizes':np.arange(3,13,1),
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],

}

'''
param_grid={
        'hidden_layer_sizes':np.arange(4,14,1),
        'alpha':[0.01,0.05,0.1],
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],
        'solver':['lbfgs', 'sgd', 'adam']
        
}
'''

'''
param_grid={
        'hidden_layer_sizes':np.arange(4,13,1),
        'max_iter':np.arange(1,501,50),
}
'''
'''
regressor=MLPRegressor(
                        random_state=90)
'''

'''
regressor=MLPRegressor(
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
'''
'''
regressor=MLPRegressor(
                        solver='lbfgs',
                        random_state=90)


GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

'''
{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': 10, 'max_iter': 101}
0.9042817798547244
'''

#regressor=GS.best_estimator_
regressor=MLPRegressor(
                        alpha=0.01,
                        hidden_layer_sizes=10,
                        max_iter=101,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)


######################6.评估模型(训练集)######################
MAE_train=metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
MAPE_train=metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train=metrics.r2_score(y_train, y_train_pred)



print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('MAPE_train:', MAPE_train)
print('r2_score_train:', R2_train)


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

'''
MAE_test: 0.07756601112913494
MSE_test: 0.009966388253588579
RMSE_test: 0.09983179981142572
MAPE_test: 5608.302644623156
r2_score_test: 0.3982570465260671
EV_test: 0.4865097413582994
'''

'''
MAE_test: 0.05198216249126438
MSE_test: 0.0050538560932306765
RMSE_test: 0.0710904782177661
MAPE_test: 2729.405833062569
r2_score_test: 0.6948621491965409
EV_test: 0.7182771911271462
'''
