import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 

from openpyxl import Workbook









######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand.xlsx',sheet_name='Sheet1_er_kg')








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





# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)












######################调参######################

param_grid = [
            {
             'weights':['uniform'],
             'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
                            33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,
                            63,65,67,69,71,73,75,77,79,81,83,85,87,89,91]
            },
            {
             'weights':['distance'],
             'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
                            33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,
                            63,65,67,69,71,73,75,77,79,81,83,85,87,89,91],
             'p':[1,2]
            },

    
]

'''
param_grid = {
    'n_neighbors': range(1, 21),  # 邻居数量的范围
    'weights': ['uniform', 'distance'],  # 权重类型
    'metric': ['euclidean', 'manhattan']  # 距离度量方法
}
'''

'''
param_grid = [
            {
             'weights':['uniform'],
             'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
            },
            {
             'weights':['distance'],
             'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
             'p':[1,2]
            },

    
]
'''

'''
param_grid = [
            {
             'weights':['uniform'],
             'n_neighbors':[3,5,7,9,11]
            },
            {
             'weights':['distance'],
             'n_neighbors':[3,5,7,9,11],
             'p':[i for i in range(1,4)]
            },

    
]
'''






#调参#
scorel=[]


k=5
#n_estimators
#{'n_neighbors': 3, 'weights': 'uniform'}----0.6923980847727724


regressor=KNeighborsRegressor()
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

#regressor=KNeighborsRegressor(n_neighbors=3, weights='uniform')





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
MAE: 0.04196976464333333
MSE: 0.010999189469846496
RMSE: 0.1048770206949382
MAPE: 81.4880907704293
r2_score: 0.07427250411692299
EV: 0.20892347200102446
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
excel_file_path = 'KNN_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')

'''


