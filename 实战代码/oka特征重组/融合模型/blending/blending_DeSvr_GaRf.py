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

from sklearn.svm import SVR

from sklearn.neural_network import MLPRegressor


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
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)





# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)







# 训练基本模型
base_model_1 = SVR(kernel='rbf',
         C=14.063456167204542,
         epsilon=3.363135491649024e-05,
         gamma=10.421706507320557)


base_model_3 = RandomForestRegressor(
                            n_estimators=20,
                            max_depth=6,
                            max_features=4,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=90)

base_model_1.fit(x_train, y_train)
base_model_3.fit(x_train, y_train)



# 生成基本模型的预测结果
predictions_1 = base_model_1.predict(x_test)
predictions_3 = base_model_3.predict(x_test)





# 使用基本模型的预测结果和元模型进行训练
meta_model = LinearRegression()
meta_model.fit(np.column_stack((predictions_1, predictions_3)), y_test)



# 预测
#blend_predictions = meta_model.predict(np.column_stack((base_model_1.predict(x_test), base_model_lgbm.predict(x_test),base_model_xgb.predict(x_test))))
y_pred = meta_model.predict(np.column_stack((base_model_1.predict(x_test), base_model_3.predict(x_test))))


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
MAE: 0.0060864303427844175
MSE: 8.743255483486303e-05
RMSE: 0.00935053767624424
MAPE: 228.75535784105946
r2_score: 0.9947210641972374
EV: 0.9947210641972374
'''
