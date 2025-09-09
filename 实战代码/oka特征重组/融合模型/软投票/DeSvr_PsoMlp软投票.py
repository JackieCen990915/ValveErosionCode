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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)

# 使用MinMaxScaler进行归一化,对结果没影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

 

svr = SVR(kernel='rbf',
         C=14.063456167204542,
         epsilon=3.363135491649024e-05,
         gamma=10.421706507320557)


mlp = MLPRegressor(
                    alpha=9.577699892831528e-05,
                    hidden_layer_sizes=13,
                    activation='relu',
                    solver='lbfgs',
                    random_state=90)
rf = RandomForestRegressor(
                            n_estimators=20,
                            max_depth=6,
                            max_features=4,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=90)


# 软投票
#R2
#regressor= VotingRegressor(estimators=[('svr', svr),('mlp', mlp)], weights=[0.78657457,0.21440629])
regressor= VotingRegressor(estimators=[('svr', svr),('mlp', mlp),('rf', rf)], weights=[0.62223953,0.12769325,0.27890609])


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
EV=metrics.explained_variance_score(y_test, y_pred)



print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)
print('r2_score:', R2)
print('EV:', EV)

'''
svr+mlp
MAE: 0.007204458025069375
MSE: 0.0001350275149432314
RMSE: 0.011620134032928855
MAPE: 331.7953116048323
r2_score: 0.9918474121642884
EV: 0.9926300136889777
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

