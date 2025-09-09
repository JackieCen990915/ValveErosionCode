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
'''
random_selected_train_index=[4, 23, 7, 64, 33, 73]
random_multiple=[2.24464873,2.94976447,2.00922533,1.71312197,2.34816149,1.92963569]
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



# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)









######################调参######################
# 设置要调优的超参数范围

param_grid = {
    'max_features': ['auto','sqrt','log2',None,4,5]
    
}


'''
param_grid = {
    'max_depth': np.arange(1,21,1),
    'max_features': np.arange(2,21,1),
    'max_features': ['auto','sqrt','log2',None]
    'min_samples_leaf': np.arange(1,21,1),
    'min_samples_split': np.arange(2,21,1)
}
'''









#调参#
scorel=[]


k=5
#{'max_depth': 14}----0.8021837655635032
'''
regressor=DecisionTreeRegressor(
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''




#{'max_features': 2}---0.8076676093698112
#{'max_features': 'sqrt'}-----0.8076676093698112
'''
regressor=DecisionTreeRegressor(
                           max_depth=14,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

#{'min_samples_leaf': 1}----0.8076676093698112
#{'min_samples_leaf': 1}-------0.8076676093698112
'''
regressor=DecisionTreeRegressor(
                           max_depth=14,
                           max_features='sqrt',
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

#{'min_samples_split': 7}---0.834070513719557
#{'min_samples_split': 7}-------0.834070513719557
'''
regressor=DecisionTreeRegressor(
                           max_depth=14,
                           max_features='sqrt',
                           min_samples_leaf=1,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''













######################5.使用测试集数据进行预测######################
'''
# 初始化决策树回归模型
regressor = DecisionTreeRegressor()


# 初始化网格搜索对象
grid_search = GridSearchCV(regressor, param_grid, cv=5)  # cv表示交叉验证的折数

# 执行网格搜索
grid_search.fit(x_train, y_train)

# 打印最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 获取最佳模型
best_model = grid_search.best_estimator_

regressor=best_model

'''


regressor=DecisionTreeRegressor(
                           max_depth=14,
                           max_features='sqrt',
                           min_samples_leaf=1,
                           min_samples_split=7,
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

print('----------------------------------')
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

print('----------------------------------')
print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('MAPE_test:', MAPE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)



'''
----------------------------------
MAE_train: 0.06750508736397881
MSE_train: 0.0482047378720237
RMSE_train: 0.21955577394371503
MAPE_train: 14.010322953202676
r2_score_train: 0.5881253319070183
----------------------------------
MAE_test: 0.04587486159762132
MSE_test: 0.006932267405119249
RMSE_test: 0.08326023904072849
MAPE_test: 7.159350392681712
r2_score_test: 0.5814488702940576
EV_test: 0.5835890316269945
'''


'''
----------------------------------
MAE_train: 0.023038917493216838
MSE_train: 0.005018855971442572
RMSE_train: 0.07084388450277534
MAPE_train: 0.49198241085347943
r2_score_train: 0.8861234130369287
----------------------------------
MAE_test: 0.029937185436880797
MSE_test: 0.0035296409443376837
RMSE_test: 0.05941078138130893
MAPE_test: 895.6774878051682
r2_score_test: 0.7868900435638239
EV_test: 0.803052689946892
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













######################7.特征重要性######################

#获取特征名称
features=x.columns
#获取特征重要性
importances=regressor.feature_importances_


#通过二维表格形式显示
importances_df=pd.DataFrame()
importances_df['特征名称']=features
importances_df['特征重要性']=importances
importances_df.sort_values('特征重要性',ascending=False)
print(importances_df)
'''

'''
