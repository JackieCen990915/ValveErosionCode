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
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand_oka.xlsx',sheet_name='Sheet1_er_kg')






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
#{'n_neighbors': 3, 'weights': 'uniform'}----  0.6787027168655915

regressor=KNeighborsRegressor()
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''
{'n_neighbors': 3, 'weights': 'uniform'}
0.6787027168655915
'''

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

'''
MAE_train: 0.02659234007850877
MSE_train: 0.004374259185217326
RMSE_train: 0.06613818250615393
MAPE_train: 3.701210759631934
r2_score_train: 0.8484765161063283
'''





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
MAE: 0.026818550733333335
MSE: 0.005662052398506093
RMSE: 0.07524661054496802
MAPE: 13.583290601626775
r2_score: 0.5234632876543253
EV: 0.5754025909273528

MAE_test: 0.02691636919298246
MSE_test: 0.005927265598427467
RMSE_test: 0.07698873682836643
MAPE_test: 14.294669793613528
r2_score_test: 0.1829203789256597
EV_test: 0.2679492636062909


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









