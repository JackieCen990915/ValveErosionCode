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

import datetime

import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import pandas as pd
# 记录开始时间
#start_time = datetime.datetime.now()





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




# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


print("---------------x_train------------------")
print(x_train)
print("---------------y_train------------------")
print(y_train)
print("---------------x_test------------------")
print(x_test)
print("type of x_test:",type(x_test))
print("---------------y_test------------------")
print(y_test)



######################5.使用数据进行预测######################
regressor=RandomForestRegressor(
                            n_estimators=73,
                            max_depth=10,
                            random_state=42)
                      
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)


print('y_test_pred:', y_test_pred)
print('-------------------')





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




######################6.使用测试练集数据进行预测######################
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
y_test_pred: [0.02009239 0.01332408 0.00271589 0.02117609 0.00196682 0.00874576
 0.02835501 0.00716924 0.00083416 0.00983142 0.02085849 0.01764275
 0.00039509 0.008244  ]
-------------------
MAE_train: 0.0011797722778562148
MSE_train: 4.25066484779088e-06
RMSE_train: 0.00206171405577759
MAPE_train: 1.8447749309458625
r2_score_train: 0.9501109954595974
EV_train: 0.9501335939879957
-------------------
MAE_test: 0.002088529149604394
MSE_test: 7.393677239612607e-06
RMSE_test: 0.0027191317069264237
MAPE_test: 1.0211866284407003
r2_score_test: 0.8673574922668741
EV_test: 0.9340396819544605
-------------------
'''
