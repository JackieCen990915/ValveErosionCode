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

from openpyxl import Workbook

######################1.导入数据######################
df = pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx', sheet_name='Sheet2')







######################2.提取特征变量######################
x=df.drop(columns='er')
y=df['er']


print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))






#将原数据集划分为训练集和测试集两部分，再将训练集进一步划分为训练集和验证集两部分
x_train_temp, x_test, y_train_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size=0.25, random_state=42)



# 使用 MinMaxScaler 进行归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)




# 训练基本模型
'''
base_model_1 = SVR(kernel='rbf',
         C=14.063456167204542,
         epsilon=3.363135491649024e-05,
         gamma=10.421706507320557)


base_model_2 = MLPRegressor(
                    alpha=0.0064975691995336855,
                    hidden_layer_sizes=6,
                    activation='relu',
                    solver='lbfgs',
                    random_state=90)


base_model_3 = RandomForestRegressor(
                            n_estimators=20,
                            max_depth=6,
                            max_features=4,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=90)
'''
base_model_1 = DecisionTreeRegressor(
                            max_depth=13,
                            max_features=6,
                            min_samples_leaf=2,
                            min_samples_split=5, 
                            random_state=100)


base_model_2 = RandomForestRegressor(
                            n_estimators=73,
                            max_depth=10,
                            random_state=42)


base_model_3 = MLPRegressor(
                    alpha=0.005423798452377173,
                    hidden_layer_sizes=1,
                    activation='tanh',
                    solver='lbfgs',
                    random_state=90)


# 对基学习器，用训练集进行训练
base_model_1.fit(x_train, y_train)
base_model_2.fit(x_train, y_train)
base_model_3.fit(x_train, y_train)



# 对基学习器，用验证集和测试集进行预测
predictions_1_val = base_model_1.predict(x_val)
predictions_2_val = base_model_2.predict(x_val)
predictions_3_val = base_model_3.predict(x_val)


# 对基学习器，用测试集进行预测
predictions_1_test = base_model_1.predict(x_test)
predictions_2_test = base_model_2.predict(x_test)
predictions_3_test = base_model_3.predict(x_test)



# 对基学习器，将验证集的预测结果存储起来并将其作为元学习器的新训练集，元学习器使用新的训练集进行训练
meta_model = LinearRegression()
meta_model.fit(np.column_stack((predictions_1_val, predictions_2_val, predictions_3_val)), y_val)




# 对基学习器，将测试集的预测结果存储起来并将其作为元学习器的新测试集，元学习器在新测试集上的预测结果即为最终预测结果
y_pred = meta_model.predict(np.column_stack((predictions_1_test, predictions_2_test, predictions_3_test)))


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
MAE: 0.005598504629772414
MSE: 7.796032905532702e-05
RMSE: 0.008829514655705998
MAPE: 318.51719913229977
r2_score: 0.9952929709874928
EV: 0.9952929709874928
'''

'''

# 创建一个新的Excel工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 写入数据
for col1,col2 in zip(y_test,y_pred):
    sheet.append([col1,col2])

# 指定Excel文件路径
excel_file_path = 'BFM4结果.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')
'''
