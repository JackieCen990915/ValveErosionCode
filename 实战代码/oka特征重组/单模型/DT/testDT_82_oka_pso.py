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


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook



import shap


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





# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


def train_dt(x):
    max_depth,max_features,min_samples_leaf,min_samples_split = x
    regressor = DecisionTreeRegressor(
                            max_depth=int(max_depth),
                            max_features=int(max_features),
                            min_samples_leaf=int(min_samples_leaf),
                            min_samples_split=int(min_samples_split), 
                            random_state=100)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score



'''
def train_dt(x):
    max_depth,max_features = x
    regressor=DecisionTreeRegressor(
                               max_depth=int(max_depth),
                               max_features=int(max_features),
                               random_state=100)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''


######################调参######################
# 设置要调优的超参数范围
'''
param_grid = {
    'max_features': ['auto', 'sqrt', 'log2', None,4,5]
}
'''


'''
param_grid = {
    'max_depth': np.arange(1,21,1),
    'max_features': ['auto', 'sqrt', 'log2', None,4,5]
    'min_samples_leaf': np.arange(1,21,1),
    'min_samples_split': np.arange(2,21,1)
}
'''

#pso=PSO(func=train_dt,dim=2, pop=200, max_iter=400, lb=[1,1], ub=[15,11],w=1,c1=2,c2=2)
pso=PSO(func=train_dt,dim=4, pop=200, max_iter=400, lb=[1,1,1,2], ub=[15,11,10,10],w=1,c1=2,c2=2)

#pso=PSO(func=train_dt,dim=2, pop=50, max_iter=50, lb=[1,1], ub=[100,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_dt,dim=3, pop=200, max_iter=400, lb=[1,1,1], ub=[100,20,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_dt,dim=4, pop=200, max_iter=400, lb=[1,1,1,2], ub=[100,20,20,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_dt,dim=5, pop=200, max_iter=400, lb=[1,1,1,1,2], ub=[100,20,8,20,20],w=1,c1=2,c2=2)
#pso=PSO(func=train_dt,dim=2, pop=200, max_iter=400, lb=[1,1], ub=[200,20],w=1,c1=2,c2=2)


# 查看当前步长
#print(pso.steps_)

pso.run()
# 获取优化结果
print('best_x:', pso.gbest_x, '\n', 'best_y:', pso.gbest_y)



max_depth=int(pso.gbest_x[0])
max_features=int(pso.gbest_x[1])
min_samples_leaf=int(pso.gbest_x[2])
min_samples_split=int(pso.gbest_x[3])


print('best_x[0]:', max_depth)
print('best_x[1]:', max_features)
print('best_x[2]:', min_samples_leaf)
print('best_x[3]:', min_samples_split)


'''
max_depth=int(pso.gbest_x[0])
max_features=int(pso.gbest_x[1])

print('best_x[0]:', max_depth)
print('best_x[1]:', max_features)
'''

######################5.使用数据进行预测######################
'''
regressor=DecisionTreeRegressor(
                           max_depth=max_depth,
                           max_features=max_features,
                           random_state=100)
'''
regressor=DecisionTreeRegressor(
                            max_depth=max_depth,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split, 
                            random_state=100)


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
best_x: [7.59556807 6.01993078 1.         5.4194255 ] 
 best_y: -0.9679825741816297
best_x[0]: 7
best_x[1]: 6
best_x[2]: 1
best_x[3]: 5
MAE_train: 0.001193012738095238
MSE_train: 3.275601429833482e-06
RMSE_train: 0.0018098622681943183
MAPE_train: 0.688900504686511
r2_score_train: 0.9377207818448476
EV_train: 0.9377207818448476
-------------------
MAE_test: 0.00108166244047619
MSE_test: 1.7846956951429059e-06
RMSE_test: 0.0013359250335040907
MAPE_test: 0.21397288955428698
r2_score_test: 0.9679825741816297
EV_test: 0.9682901987390513
-------------------
Duration is  0:01:03.371883
'''



'''
best_x: [7.00319448 3.48281358] 
 best_y: -0.9403788205821425
best_x[0]: 7
best_x[1]: 3
MAE_train: 0.0003072916666666666
MSE_train: 4.122892113095238e-07
RMSE_train: 0.0006420975091911849
MAPE_train: 0.043631134270459934
r2_score_train: 0.9921611190237309
EV_train: 0.9921611190237309
-------------------
MAE_test: 0.0013388021428571428
MSE_test: 3.323367182921429e-06
RMSE_test: 0.001823010472521052
MAPE_test: 0.6434625968078025
r2_score_test: 0.9403788205821425
EV_test: 0.9419034984645887
-------------------
Duration is  0:00:59.071673

'''


