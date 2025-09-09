import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#网格搜索
from sklearn.model_selection import GridSearchCV
#随机搜索
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook

from sko.GA import GA #导入遗传算法的包

#导入粒子群算法
from sko.PSO import PSO


import datetime

# 记录开始时间
start_time = datetime.datetime.now()

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



def train_rf(x):
    n_estimators,max_depth = x
    #n_estimators,max_depth,max_features,min_samples_leaf,min_samples_split = x
    clf = RandomForestRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            #max_features=int(max_features),
                            #min_samples_leaf=int(min_samples_leaf),
                            #min_samples_split=int(min_samples_split), 
                            random_state=90)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''
def train_rf(x):
    n_estimators,max_depth,min_samples_leaf,min_samples_split = x
    clf = RandomForestRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            min_samples_leaf=int(min_samples_leaf),
                            min_samples_split=int(min_samples_split), 
                            random_state=42)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''

'''
def train_rf(x):
    #n_estimators,max_depth,max_features,min_samples_leaf,min_samples_split = x
    n_estimators,max_depth = x
    clf = RandomForestRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            #max_features=int(max_features),
                            #min_samples_leaf=int(min_samples_leaf),
                            #min_samples_split=int(min_samples_split), 
                            random_state=42)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''





######################调参######################
'''
param_grid = {
        'n_estimators': np.arange(1,201,10),
        'max_depth': np.arange(1,21,1),
        'max_features': np.arange(3,21,1),
        'min_samples_leaf': np.arange(1,21,1), 
        'min_samples_split': np.arange(2,21,1)    
    
}

'''
'''
param_grid = {
         'min_samples_split': np.arange(2,21,1),       
}

'''


#pso=PSO(func=train_rf,dim=2, pop=50, max_iter=50, lb=[1,1], ub=[100,20],w=1,c1=2,c2=2)
pso=PSO(func=train_rf,dim=2, pop=200, max_iter=400, lb=[1,1], ub=[100,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_rf,dim=3, pop=200, max_iter=400, lb=[1,1,1], ub=[100,20,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_rf,dim=4, pop=200, max_iter=400, lb=[1,1,1,2], ub=[100,20,20,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_rf,dim=5, pop=200, max_iter=400, lb=[1,1,1,1,2], ub=[100,20,8,20,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_rf,dim=2, pop=200, max_iter=400, lb=[1,1], ub=[200,20],w=1,c1=2,c2=2)


# 查看当前步长
#print(pso.steps_)

pso.run()
# 获取优化结果
print('best_x:', pso.gbest_x, '\n', 'best_y:', pso.gbest_y)




n_estimators=int(pso.gbest_x[0])
max_depth=int(pso.gbest_x[1])
#max_features=int(pso.gbest_x[2])
#min_samples_leaf=int(pso.gbest_x[3])
#min_samples_split=int(pso.gbest_x[4])

print('best_x[0]:', n_estimators)
print('best_x[1]:', max_depth)
#print('best_x[2]:', max_features)
#print('best_x[3]:', min_samples_leaf)
#print('best_x[4]:', min_samples_split)


######################5.使用数据进行预测######################

regressor=RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            #max_features=max_features,
                            #min_samples_leaf=min_samples_leaf,
                            #min_samples_split=min_samples_split, 
                            random_state=42)
                        

regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)




######################5.使用训练集数据进行预测######################
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




######################6.5.使用测试练集数据进行预测######################
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
best_x: [29.29962484 10.25098242] 
 best_y: -0.9693521392233893
best_x[0]: 29
best_x[1]: 10
MAE_train: 0.0006668061145320196
MSE_train: 8.445741751582454e-07
RMSE_train: 0.0009190071681756598
MAPE_train: 0.9593543699649264
r2_score_train: 0.9839420575336719
EV_train: 0.9839581127810084
-------------------
MAE_test: 0.0012004475862068971
MSE_test: 2.3993009242234105e-06
RMSE_test: 0.001548967696313713
MAPE_test: 0.8907683669379837
r2_score_test: 0.9569565615211958
EV_test: 0.9617655723724133

'''





