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
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/冲蚀/周报&开会讨论_研二下/实战所用数据/弯头气固/0229_冲蚀数据统计_Num_GA_oka.xlsx',sheet_name='Sheet1')








######################2.提取特征变量######################
x=df.drop(columns='erosion_rate')
y=df['erosion_rate']


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

 
 


clf1=RandomForestRegressor(n_estimators=51,
                           max_depth=3,
                           max_features=3,
                           min_samples_leaf=1,
                           min_samples_split=9,      
                           random_state=90)


clf2=SVR(kernel='rbf',C=5, epsilon=0.01, gamma=0.5)



clf3=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,                        
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15,
                   feature_fraction=0.8,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0,
                   reg_lambda=0.05,     
                   min_split_gain=0,      
                   random_state=90)


estimators=[ ('rf',clf1),( 'svr',clf2),( 'lgbm',clf3)]
final_estimator=LinearRegression()

'''
final_estimator=MLPRegressor(
                        hidden_layer_sizes=11,
                        max_iter=101,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
'''
eclf=StackingRegressor(estimators=estimators,
                       final_estimator=final_estimator)



eclf.fit(x_train,y_train)
y_train_pred=eclf.predict(x_train)
y_test_pred=eclf.predict(x_test)



######################6.评估模型(训练集)######################
MAE_train=metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
MAPE_train=metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train=metrics.r2_score(y_train, y_train_pred)


print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('MAPE_train:', MAPE_train)
print('r2_score_train:', R2_train)


'''
MAE_train: 0.039721967682954246
MSE_train: 0.006232323131882562
RMSE_train: 0.07894506401215064
MAPE_train: 3707.295414879252
r2_score_train: 0.7899262867755639
'''




######################6.评估模型(测试集)######################
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


'''
MAE: 0.7855659989974344
MSE: 1.7876948134665216
RMSE: 1.337047049832773
MAPE: 0.03368850533401385
r2_score: 0.9762214902953772
'''
