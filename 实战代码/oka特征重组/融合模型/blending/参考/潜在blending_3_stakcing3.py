from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler 


import pandas as pd

from sklearn.ensemble import StackingRegressor


# 划分训练集和测试集
######################1.导入数据######################
df=pd.read_excel('../../../0114_cx_整理数据_17_最终.xlsx',sheet_name='Sheet1')








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













# 训练基本模型

base_model_rf = RandomForestRegressor(n_estimators=181,
                           max_depth=20,
                           max_features=4,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           random_state=90)


base_model_lgbm = LGBMRegressor(n_estimators=201,
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

base_model_xgb = xgb.XGBRegressor(
                  learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90)


# stacking
estimators=[ ('rf',base_model_rf),( 'lgbm',base_model_lgbm),( 'xgb',base_model_xgb)]
final_estimator=LinearRegression()

base_model_stacking=StackingRegressor(estimators=estimators,
                       final_estimator=final_estimator)



base_model_rf.fit(x_train, y_train)
base_model_lgbm.fit(x_train, y_train)
base_model_xgb.fit(x_train, y_train)
base_model_stacking.fit(x_train, y_train)

# 生成基本模型的预测结果
predictions_rf = base_model_rf.predict(x_test)
predictions_lgbm = base_model_lgbm.predict(x_test)
predictions_xgb = base_model_xgb.predict(x_test)
predictions_stacking = base_model_stacking.predict(x_test)








# 使用基本模型的预测结果和元模型进行训练
meta_model = LinearRegression()
meta_model.fit(np.column_stack((predictions_rf,predictions_lgbm,predictions_xgb,predictions_stacking)), y_test)



# 预测
#blend_predictions = meta_model.predict(np.column_stack((base_model_rf.predict(x_test), base_model_lgbm.predict(x_test),base_model_xgb.predict(x_test))))
y_pred = meta_model.predict(np.column_stack((base_model_rf.predict(x_test),base_model_lgbm.predict(x_test),base_model_xgb.predict(x_test),base_model_stacking.predict(x_test))))


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
MAE: 0.40641632507810627
MSE: 0.3316661914403608
RMSE: 0.5759046721813956
MAPE: 0.019168136001912996
r2_score: 0.9955884373034751
'''
