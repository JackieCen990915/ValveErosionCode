from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold



from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import VotingRegressor

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from openpyxl import Workbook

######################1.导入数据######################
df=pd.read_excel('../../0114_cx_整理数据_17_最终.xlsx',sheet_name='Sheet1')








######################2.提取特征变量######################
x=df.drop(columns='ROP ')
y=df['ROP ']


print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))






######################3.划分训练集和测试集######################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)

# 使用MinMaxScaler进行归一化,对结果没影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

 

knn=KNeighborsRegressor(n_neighbors=1,
                        p=1,
                        weights='distance')


rf=RandomForestRegressor(n_estimators=181,
                           max_depth=20,
                           max_features=4,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           random_state=90)


 

# 软投票
#regressor= VotingRegressor(estimators=[('knn', knn),('rf', rf)], weights=[2,1])
regressor= VotingRegressor(estimators=[('knn', knn),('rf', rf)], weights=[0.55642844,0.44404596])


regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print("---------------x_train------------------")
print(x_train)
print("---------------y_train------------------")
print(y_train)
print("---------------x_test------------------")
print(x_test)
print("---------------y_test------------------")
print(y_test)
print("---------------y_pred------------------")
print(y_pred)


MAE=metrics.mean_absolute_error(y_test, y_pred)
MSE=metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(MSE)
MAPE=metrics.mean_absolute_percentage_error(y_test, y_pred)
R2=metrics.r2_score(y_test, y_pred)


print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)
print('r2_score:', R2)

'''
MAE: 0.30941301293526907
MSE: 0.23256258429480312
RMSE: 0.4822474305735626
MAPE: 0.014190231541077635
r2_score: 0.9969066355029229

regressor= VotingRegressor(estimators=[('knn', knn),('rf', rf)], weights=[2,1])
----------------------------
MAE: 0.300948689174301
MSE: 0.22401832577601094
RMSE: 0.4733057423864736
MAPE: 0.013760680649785278
r2_score: 0.9970202845064203

regressor= VotingRegressor(estimators=[('knn', knn),('rf', rf)], weights=[0.55642844,0.44404596])


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
excel_file_path = 'RF_LGBM_XGB_yTest_yPred_软投票.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')

'''

