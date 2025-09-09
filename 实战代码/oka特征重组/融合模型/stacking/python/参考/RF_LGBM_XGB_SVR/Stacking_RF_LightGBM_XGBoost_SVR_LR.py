from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler 


import pandas as pd

from openpyxl import Workbook



######################1.导入数据######################
df=pd.read_excel('../../../../0114_cx_整理数据_17_最终.xlsx',sheet_name='Sheet1')








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

 
 


print("---------------y_train------------------")
print(y_train)
print("---------------y_test------------------")
print(y_test)



clf1=RandomForestRegressor(n_estimators=181,
                           max_depth=20,
                           max_features=4,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           random_state=90)


clf2=LGBMRegressor(n_estimators=201,
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

clf3=xgb.XGBRegressor(
                  learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90)


clf4=SVR(kernel='rbf',C=100,epsilon=0.8,gamma=1)

# 软投票
estimators=[ ('rf',clf1),('lgbm',clf2),( 'xgb',clf3),( 'svr',clf4)]
final_estimator=LinearRegression()

eclf=StackingRegressor(estimators=estimators,
                       final_estimator=final_estimator)



eclf.fit(x_train,y_train)
y_pred = eclf.predict(x_test)




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
excel_file_path = '0214_X5_yTest_yPred_网格搜索_.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')
'''




######################6.评估模型######################

MAE=metrics.mean_absolute_error(y_test, y_pred)
MSE=metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(MSE)
MAPE=metrics.mean_absolute_percentage_error(y_test, y_pred)
R2=metrics.r2_score(y_test, y_pred)
EV=metrics.explained_variance_score(y_test, y_pred)

print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)
print('r2_score:', R2)
print('EV:', EV)


'''

XGB-LGBM-LR:
MAE: 0.4128856559067309
MSE: 0.33484739106816513
RMSE: 0.5786599960841989
MAPE: 0.019884731821169448
r2_score: 0.9955461234892534
EV: 0.9955494000502721

RF-LGBM-LR:
MAE: 0.39333974026805085
MSE: 0.35913930559477725
RMSE: 0.5992823254483459
MAPE: 0.018317856158539782
r2_score: 0.9952230115570804
EV: 0.9952253119044965


RF-XGB-LR:
MAE: 0.39022314064414415
MSE: 0.3158163723339049
RMSE: 0.561975419688357
MAPE: 0.01900492205236019
r2_score: 0.9957992591252985
EV: 0.9958024612758244



RF-XGB-LGBM-LR:
MAE: 0.38790214197216805
MSE: 0.3074917685291035
RMSE: 0.5545194032034438
MAPE: 0.01879254091627034
r2_score: 0.9959099864546327
EV: 0.9959130584061153


RF-XGB-LGBM-SVR--LR:
MAE: 0.3651595460809015
MSE: 0.27339837502244135
RMSE: 0.5228751046114563
MAPE: 0.017604069982444695
r2_score: 0.9963634699475952
EV: 0.9963639168809789
'''
