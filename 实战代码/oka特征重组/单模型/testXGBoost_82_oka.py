
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt


from openpyxl import Workbook



#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np




from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler








######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx',sheet_name='Sheet1')




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






param_grid={
    
       'max_depth': np.arange(1,21,1)

       
}

'''
param_grid = {

        'max_depth': np.arange(1,21,1),
        'min_child_weight':np.arange(1,21,1),
        'gamma':np.linspace(0,1,21),
        'subsample':np.linspace(0,1,21), 
        'colsample_bytree':np.linspace(0,1,21),
        'alpha':np.linspace(0,1,21),
        'reg_lambda':np.linspace(0,1,21),
        'learning_rate':np.linspace(0,1,21)
        
}
'''



#调参#
scorel=[]
k=5


#{'max_depth': 4}---0.8907667107619457---123,123

regressor=xgb.XGBRegressor(learning_rate=0.1,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)




#{'min_child_weight': 1}------0.8907667107619457----默认取值，不变
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                          max_depth=4,
                          random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'gamma': 0.0}---0.8907667107619457----默认取值，不变
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                          max_depth=4,
                          min_child_weight=1,
                          random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'subsample': 0.9500000000000001}---0.8990625292256217
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0,
                          random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'colsample_bytree': 1.0}---0.8990625292256217----默认取值，不变
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0,
                          subsample=0.9500000000000001,
                          random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'alpha': 0.0}---0.8990625292256217----默认取值，不变
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0,
                          subsample=0.9500000000000001,
                          colsample_bytree=1.0,
                          random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'reg_lambda': 0.2}----0.8992241725075367
'''
regressor=xgb.XGBRegressor(learning_rate=0.1,
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0,
                          subsample=0.9500000000000001,
                          colsample_bytree=1.0,
                          alpha=0,
                          random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'learning_rate': 0.55}---0.9082560718007546
'''
regressor=xgb.XGBRegressor(
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0,
                          subsample=0.9500000000000001,
                          colsample_bytree=1.0,
                          alpha=0,
                          random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''








regressor=xgb.XGBRegressor(learning_rate=0.55,
                          max_depth=4,
                          min_child_weight=1,
                          gamma=0,
                          subsample=0.9500000000000001,
                          colsample_bytree=1.0,
                          alpha=0,
                          random_state=123)



regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)







######################6.评估模型(训练集)######################
MAE_train=metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
MAPE_train=metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train=metrics.r2_score(y_train, y_train_pred)
EV_train=metrics.explained_variance_score(y_train, y_train_pred)


print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('MAPE_train:', MAPE_train)
print('r2_score_train:', R2_train)
print('EV_train:', EV_train)
print('-------------------')


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
print('-------------------')


'''

MAE_train: 0.0006333547762268659
MSE_train: 6.521173604470104e-07
RMSE_train: 0.0008075378383995454
MAPE_train: 0.5654835883107845
r2_score_train: 0.9884346019787563
EV_train: 0.9884359791295698
-------------------
MAE_test: 0.0017096081661875692
MSE_test: 5.542084558534157e-06
RMSE_test: 0.002354163239568182
MAPE_test: 2.660523008707363
r2_score_test: 0.8588723799731987
EV_test: 0.8638612327575638
-------------------
'''


