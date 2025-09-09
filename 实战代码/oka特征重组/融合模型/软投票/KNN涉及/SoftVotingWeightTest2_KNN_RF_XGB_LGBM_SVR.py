from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



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


# 定义基础模型
models = [
    KNeighborsRegressor(n_neighbors=1,
                        p=1,
                        weights='distance'),
    RandomForestRegressor(n_estimators=181,
                           max_depth=20,
                           max_features=4,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           random_state=90),
    xgb.XGBRegressor(
                  learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90),
    LGBMRegressor(learning_rate=0.1,
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
                   random_state=90),
    SVR(kernel='rbf',C=100,epsilon=0.8,gamma=1)
]



# 使用交叉验证得到基础模型的预测结果
predictions = []
for model in models:
    preds = cross_val_predict(model, x_train, y_train, cv=5, method='predict')
    predictions.append(preds)
print("predictions:", predictions)

'''
predictions: [
array([21.255, 21.724, 32.77 , ..., 35.296, 39.759, 29.024]),
array([21.23772013, 21.85195028, 31.10619337, ..., 35.13314096,
       39.43263536, 26.42427935]),
array([21.308525, 22.095673, 31.561045, ..., 34.74696 , 40.049156,
       26.671701], dtype=float32),
array([21.69676394, 21.17509074, 31.67575762, ..., 35.44149893,
       40.98199427, 26.21195224])]
'''

# 定义均方误差损失函数
def mse_loss(weights):
    ensemble_preds = np.dot(weights, predictions)
    return mean_squared_error(y_train, ensemble_preds)
print("mse_loss:", mse_loss)




# 初始权重
initial_weights = np.ones(len(models)) / len(models)

# 最小化均方误差损失函数，得到优化后的权重
result = minimize(mse_loss, initial_weights, method='L-BFGS-B')

# 优化后的权重: [ 0.44374185  0.18051005  0.30402761  0.13923685 -0.06707814]
optimized_weights = result.x
print("优化后的权重:", optimized_weights)
