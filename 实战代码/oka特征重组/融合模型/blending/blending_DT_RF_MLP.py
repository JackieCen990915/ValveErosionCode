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







######################3.划分训练集和测试集######################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)





# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



# 第一步：训练基学习器
# 训练决策树基学习器
dt_model = DecisionTreeRegressor(
                            max_depth=13,
                            max_features=6,
                            min_samples_leaf=2,
                            min_samples_split=5, 
                            random_state=100)
dt_model.fit(x_train, y_train)



# 训练随机森林基学习器
rf_model = RandomForestRegressor(
                            n_estimators=73,
                            max_depth=10,
                            random_state=42)
rf_model.fit(x_train, y_train)




# 训练多层感知机基学习器
mlp_model = MLPRegressor(
                    alpha=0.005423798452377173,
                    hidden_layer_sizes=1,
                    activation='tanh',
                    solver='lbfgs',
                    random_state=90)
mlp_model.fit(x_train, y_train)



# 用基学习器对训练集进行预测
dt_train_pred = dt_model.predict(x_train)
rf_train_pred = rf_model.predict(x_train)
mlp_train_pred = mlp_model.predict(x_train)



# 将基学习器的预测结果横向堆叠，作为新的特征
new_train_features = np.column_stack((dt_train_pred, rf_train_pred, mlp_train_pred))

# 第二步：训练元学习器（线性回归，这里以简单的线性回归替代逻辑回归用于回归任务）
lr_model = LinearRegression()
lr_model.fit(new_train_features, y_train)

# 第三步：用训练好的模型进行预测并评估

# 用基学习器对测试集进行预测
dt_test_pred = dt_model.predict(x_test)
rf_test_pred = rf_model.predict(x_test)
mlp_test_pred = mlp_model.predict(x_test)

# 将基学习器的测试集预测结果横向堆叠，作为新的特征用于元学习器预测
new_test_features = np.column_stack((dt_test_pred, rf_test_pred, mlp_test_pred))

# 用元学习器进行最终预测
final_pred = lr_model.predict(new_test_features)

# 计算均方误差（MSE）来评估最终融合模型的性能，也可根据需求选用其他评估指标，如平均绝对误差（MAE）等
MAE=metrics.mean_absolute_error(y_test, final_pred)
MSE=metrics.mean_squared_error(y_test, final_pred)
RMSE=np.sqrt(MSE)
MAPE=metrics.mean_absolute_percentage_error(y_test, final_pred)
R2=metrics.r2_score(y_test, final_pred)
EV=metrics.explained_variance_score(y_test, final_pred)



print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)
print('r2_score:', R2)
print('EV:', EV)


'''
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

import shap

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







######################3.划分训练集和测试集######################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)





# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)







# 定义基础模型
base_model_1 = DecisionTreeRegressor(
                            max_depth=13,
                            max_features=6,
                            min_samples_leaf=2,
                            min_samples_split=5, 
                            random_state=100)

# 训练基础模型
base_model_2 = MLPRegressor(
                    alpha=0.005423798452377173,
                    hidden_layer_sizes=1,
                    activation='tanh',
                    solver='lbfgs',
                    random_state=90)


base_model_3 = RandomForestRegressor(
                            n_estimators=73,
                            max_depth=10,
                            random_state=42)


base_model_1.fit(x_train, y_train)
base_model_2.fit(x_train, y_train)
base_model_3.fit(x_train, y_train)



# 在测试集上进行预测
predictions_1 = base_model_1.predict(x_test)
predictions_2 = base_model_2.predict(x_test)
predictions_3 = base_model_3.predict(x_test)



# 定义元模型
meta_model = LinearRegression()


# 将基础模型的预测结果作为输入，训练元模型
meta_model.fit(np.column_stack((predictions_1, predictions_2, predictions_3)), y_test)



# 使用元模型进行预测
#blend_predictions = meta_model.predict(np.column_stack((base_model_1.predict(x_test), base_model_lgbm.predict(x_test),base_model_xgb.predict(x_test))))
y_pred = meta_model.predict(np.column_stack((base_model_1.predict(x_test), base_model_2.predict(x_test), base_model_3.predict(x_test))))


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

'''
MAE: 0.0006755108871618922
MSE: 6.474177972932213e-07
RMSE: 0.0008046227670736277
MAPE: 0.3964430456451455
r2_score: 0.9883853301407395
EV: 0.9883853301407394
'''


'''
#########使用 SHAP 模型输出特征重要性##########


# 创建一个可调用的对象，这里使用元模型
def model_function(x):
    return meta_model.predict(np.column_stack((base_model_1.predict(x), base_model_2.predict(x), base_model_3.predict(x))))


# 计算 SHAP 值
explainer = shap.KernelExplainer(model_function, x_test)
shap_values = explainer.shap_values(x_test)

# 获取特征重要性
feature_importances = np.abs(shap_values).mean(axis=0)
feature_names = x.columns

# 打印特征重要性
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance}")
    
shap.summary_plot(shap_values, x_test, feature_names=feature_names)
shap.summary_plot(shap_values, x_test, feature_names=feature_names, plot_type='bar')
'''


