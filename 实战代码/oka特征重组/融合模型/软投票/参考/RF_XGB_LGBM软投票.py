from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np
from sklearn.svm import SVR
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

 

xgb= xgb.XGBRegressor(
                  learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90)

lgbm=LGBMRegressor(learning_rate=0.1,
                   n_estimators=201,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0.25,
                   reg_lambda=0.0,
                   min_split_gain=0,  
                   random_state=90)


# 软投票
#regressor= VotingRegressor(estimators=[('knn', knn),('rf', rf)], weights=[2,1])
regressor= VotingRegressor(estimators=[('xgb', xgb),('lgbm', lgbm)], weights=[0.69069845,0.31053728]
)


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
MAE: 0.38542848601960567
MSE: 0.3148668241676758
RMSE: 0.56112995301238
MAPE: 0.017970054862383567
r2_score: 0.9958118892678238
regressor= VotingRegressor(estimators=[('rf', rf),('xgb', xgb),('lgbm', lgbm)], weights=[0.27617455,0.55746202,0.16768997]



----------------------------
MAE: 0.3891397280722864
MSE: 0.32470136525189713
RMSE: 0.569825732353232
MAPE: 0.018133334544044914
r2_score: 0.9956810779409406
regressor= VotingRegressor(estimators=[('rf', rf),('xgb', xgb)], weights=[0.41003794,0.59130654]



---------------------------
MAE: 0.38541256888769304
MSE: 0.3595823820079624
RMSE: 0.5996518840193553
MAPE: 0.017496063171646543
r2_score: 0.9952171181032808
regressor= VotingRegressor(estimators=[('rf', rf),('lgbm', lgbm)], weights=[0.71474238,0.28657082]


------------------------
MAE: 0.408545394628959
MSE: 0.3403141383801489
RMSE: 0.5833644987314097
MAPE: 0.019139972292063497
r2_score: 0.9954734091181922
regressor= VotingRegressor(estimators=[('xgb', xgb),('lgbm', lgbm)], weights=[0.69069845,0.31053728]

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

