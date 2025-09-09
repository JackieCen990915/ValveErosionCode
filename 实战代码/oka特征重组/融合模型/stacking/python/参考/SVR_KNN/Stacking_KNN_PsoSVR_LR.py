from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler 


import pandas as pd

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)




# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

 
 


clf1=SVR(kernel='rbf',C=100.0,gamma=0.9854073762340687,epsilon=0.01)
clf2=KNeighborsRegressor(n_neighbors=3, weights='uniform')




estimators=[ ('rf',clf1),( 'svr',clf2)]
final_estimator=LinearRegression()

eclf=StackingRegressor(estimators=estimators,
                       final_estimator=final_estimator)



eclf.fit(x_train,y_train)
y_pred = eclf.predict(x_test)




######################6.评估模型######################

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
MAE: 0.028816245457518756
MSE: 0.004720776396151706
RMSE: 0.06870790635837848
MAPE: 291.68758898857163
r2_score: 0.6026841319704583
'''
