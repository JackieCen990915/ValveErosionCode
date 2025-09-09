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









######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand.xlsx',sheet_name='Sheet1')







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









######################调参######################
# 设置要调优的超参数范围

param_grid = {

    
      'max_depth': [1,3,5,7,9,11,13,15,17,19]
       
    
}


'''
param_grid = {
    'max_depth': np.arange(1,21,1),
    'max_features': ['auto','sqrt','log2',None,4,5]
    'min_samples_leaf': np.arange(1,21,1),
    'min_samples_split': np.arange(2,21,1)
}
'''









#调参#
scorel=[]


k=5
#k=10
#{'max_depth': 6}---0.7702258617168747--90,90
#{'max_depth': 10}---0.7151295424221573--------90，42
#{'max_depth': 7}--------0.8097804435290609----42，42-----'max_depth': [1,3,5,7,9,11,13,15,17,19]


#{'max_depth': 5}-----0.7984693778169809----42，100-----'max_depth': [1,3,5,7,9,11,13,15,17,19]

regressor=DecisionTreeRegressor(
                           random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)




#{'max_features': 'auto'}-----0.7702258617168747---不变---默认取值
#{'max_features': 5}--0.7488953199155786
#{'max_features': 'auto'}----0.8097804435290609


#{'max_features': 'auto'}
'''
regressor=DecisionTreeRegressor(
                           max_depth=5,
                           random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'min_samples_leaf': 2}-----0.8366148456283918
#{'min_samples_leaf': 1}
#{'min_samples_leaf': 2}----0.8393459916266295



#{'min_samples_leaf': 2}
'''
regressor=DecisionTreeRegressor(
                           max_depth=5,
                           max_features='auto',
                           random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'min_samples_split': 2}---0.8268269813939882---不变---默认取值
#{'min_samples_split': 4}----0.8036199135863319
#{'min_samples_split': 2}


#{'min_samples_split': 2}
'''
regressor=DecisionTreeRegressor(
                           max_depth=5,
                           max_features='auto',
                           min_samples_leaf=2,
                           random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''




######################构建预测模型######################
'''
# 初始化决策树回归模型
regressor = DecisionTreeRegressor()
# 初始化网格搜索对象
grid_search = GridSearchCV(regressor, param_grid, cv=5)  # cv表示交叉验证的折数
# 执行网格搜索
grid_search.fit(x_train, y_train)
# 获取最佳模型
best_params=grid_search.best_params_
best_model=grid_search.best_estimator_
print("best_params: ", best_params)
print("best_model: ", best_model)
regressor=best_model
'''
'''
regressor=DecisionTreeRegressor(
                           max_depth=5,
                           max_features='auto',
                           min_samples_leaf=2,
                           min_samples_split=2,
                           random_state=100)

regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)





#####################评估模型(训练集)######################
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



#####################评估模型(测试集)######################
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
print('----------------------------------')
'''

'''
MAE_train: 0.0008119525765306121
MSE_train: 1.2147694008753827e-06
RMSE_train: 0.0011021657774016496
MAPE_train: 0.45654233719513443
r2_score_train: 0.9784558846654211
EV_train: 0.9784558846654211
-------------------
MAE_test: 0.0018875146938775508
MSE_test: 8.55261418092257e-06
RMSE_test: 0.002924485284784755
MAPE_test: 1.7299603742862042
r2_score_test: 0.7822100923194287
EV_test: 0.7883507613405271
----------------------------------
'''

#---90,42---
'''
MAE_train: 0.0007725934523809524
MSE_train: 1.598983059375e-06
RMSE_train: 0.0012645090190959495
MAPE_train: 0.4074097304546143
r2_score_train: 0.9710362154515194
EV_train: 0.9710362154515194
-------------------
MAE_test: 0.0007947252380952384
MSE_test: 1.0294776338142868e-06
RMSE_test: 0.001014631772523553
MAPE_test: 0.39646556397954125
r2_score_test: 0.9788258064709927
EV_test: 0.9788492928090378
----------------------------------
'''

#---42,100--- 数据集加入新的重构特征----'max_depth': [1,3,5,7,9,11,13,15,17,19]
'''
MAE_train: 0.0011075702197802198
MSE_train: 3.4468405712707876e-06
RMSE_train: 0.0018565668776725464
MAPE_train: 0.220342055877831
r2_score_train: 0.9344650011661779
EV_train: 0.9344650011661779
-------------------
MAE_test: 0.001248601886446886
MSE_test: 4.368805970458772e-06
RMSE_test: 0.0020901688856307214
MAPE_test: 0.2280053873432156
r2_score_test: 0.921623657492592
EV_test: 0.921895630842903
----------------------------------
'''



#######将y_test和y_pred写入xlsx，后续画实际和预测的图用###############
'''
# 创建一个新的Excel工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 写入数据
for col1,col2 in zip(y_test,y_pred):
    sheet.append([col1,col2])

# 指定Excel文件路径
excel_file_path = 'DT_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')
'''









