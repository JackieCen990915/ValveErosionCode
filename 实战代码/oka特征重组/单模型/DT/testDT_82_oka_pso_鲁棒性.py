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



##### 将10%(6个)异常值添加到训练集#####################################
random_selected_train_index=[52, 13, 2, 60, 25, 23]
random_multiple=[2.08767221,
 1.3688075 ,
 2.00090929,
 2.65562694,
 1.42969978,
 2.32287012]

y_train[random_selected_train_index]= y_train[random_selected_train_index]*random_multiple


# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



regressor=DecisionTreeRegressor(
                            max_depth=7,
                            max_features=6,
                            min_samples_leaf=1,
                            min_samples_split=5, 
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

'''
MAE_train: 0.0013149758275226191
MSE_train: 5.9690848242166695e-06
RMSE_train: 0.002443171059139468
MAPE_train: 0.2743776706295819
r2_score_train: 0.9299423241867306
EV_train: 0.9299423241867306
-------------------
MAE_test: 0.002585255427571429
MSE_test: 1.7354800766990884e-05
RMSE_test: 0.004165909356550005
MAPE_test: 0.3033876565543738
r2_score_test: 0.6886550196417471
EV_test: 0.7163566738666363
-------------------
'''
