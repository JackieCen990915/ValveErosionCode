from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
import datetime
from sko.GA import GA

# 记录开始时间
start_time = datetime.datetime.now()

######################1.导入数据######################
df = pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx', sheet_name='Sheet2')

######################2.提取特征变量######################
x = df.drop(columns='er')
y = df['er']

######################3.划分训练集和测试集######################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=90)

# 使用MinMaxScaler进行归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义参数的候选列表
n_estimators_list = list(range(1, 101))
max_depth_list = list(range(1, 20))
max_features_list = list(range(1, 12))
min_samples_leaf_list = list(range(1, 10))
min_samples_split_list = list(range(2, 11))

def train_rf(x):
    n_estimators_index, max_depth_index, max_features_index, min_samples_leaf_index, min_samples_split_index = x
    n_estimators = n_estimators_list[int(n_estimators_index)]
    max_depth = max_depth_list[int(max_depth_index)]
    max_features = max_features_list[int(max_features_index)]
    min_samples_leaf = min_samples_leaf_list[int(min_samples_leaf_index)]
    min_samples_split = min_samples_split_list[int(min_samples_split_index)]
    
    clf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=90
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test, y_pred)
    return -score

# 遗传算法参数
ga = GA(func=train_rf, n_dim=5, size_pop=200, max_iter=400, prob_mut=0.01,
        lb=[0, 0, 0, 0, 1], ub=[len(n_estimators_list)-1, len(max_depth_list)-1, len(max_features_list)-1, len(min_samples_leaf_list)-1, len(min_samples_split_list)-1],
        precision=1e-5)

best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', -best_y[0])

# 解析最佳参数
n_estimators_index, max_depth_index, max_features_index, min_samples_leaf_index, min_samples_split_index = best_x
n_estimators = n_estimators_list[n_estimators_index]
max_depth = max_depth_list[max_depth_index]
max_features = max_features_list[max_features_index]
min_samples_leaf = min_samples_leaf_list[min_samples_leaf_index]
min_samples_split = min_samples_split_list[min_samples_split_index]

print('best_x[0]:', n_estimators)
print('best_x[1]:', max_depth)
print('best_x[2]:', max_features)
print('best_x[3]:', min_samples_leaf)
print('best_x[4]:', min_samples_split)

######################5.使用数据进行预测######################
regressor = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    max_features=max_features,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    random_state=90
)
regressor.fit(x_train, y_train)
y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

######################6.使用训练集数据进行预测######################
MAE_train = metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train = metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train = np.sqrt(MSE_train)
MAPE_train = metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train = metrics.r2_score(y_train, y_train_pred)
EV_train = metrics.explained_variance_score(y_train, y_train_pred)

print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('MAPE_train:', MAPE_train)
print('r2_score_train:', R2_train)
print('EV_train:', EV_train)
print('-------------------')

######################7.使用测试集数据进行预测######################
MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
RMSE_test = np.sqrt(MSE_test)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test = metrics.r2_score(y_test, y_test_pred)
EV_test = metrics.explained_variance_score(y_test, y_test_pred)

print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('MAPE_test:', MAPE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)
print('-------------------')

# 记录结束时间
end_time = datetime.datetime.now()
# 计算运行时长
duration = end_time - start_time
# 打印运行时长
print("Duration is ", duration)


'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#网格搜索
from sklearn.model_selection import GridSearchCV
#随机搜索
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook

from sko.GA import GA #导入遗传算法的包

import datetime

import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import pandas as pd




# 记录开始时间
start_time = datetime.datetime.now()


######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx',sheet_name='Sheet2')



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



# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


print("---------------x_train------------------")
print(x_train)
print("---------------y_train------------------")
print(y_train)
print("---------------x_test------------------")
print(x_test)
print("type of x_test:",type(x_test))
print("---------------y_test------------------")
print(y_test)



def train_rf(x):
    
    n_estimators,max_depth,max_features,min_samples_leaf,min_samples_split = x
    clf = RandomForestRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            max_features=int(max_features),
                            min_samples_leaf=int(min_samples_leaf),
                            min_samples_split=int(min_samples_split),
                            random_state=42)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score







######################调参######################
ga = GA(func=train_rf, n_dim=5,size_pop=200, max_iter=400, prob_mut=0.01,lb=[1,1,1,1,2], ub=[100,19,11,9,10], precision=1e-5)




best_x, best_y = ga.run()
#拿到最佳预测值，方便之后做拟合
print('best_x:', best_x, '\n','best_y:',-best_y[0])


n_estimators=int(best_x[0])
max_depth=int(best_x[1])
max_features=int(best_x[2])
min_samples_leaf=int(best_x[3])
min_samples_split=int(best_x[4])

print('best_x[0]:', n_estimators)
print('best_x[1]:', max_depth)
print('best_x[2]:', max_features)
print('best_x[3]:', min_samples_leaf)
print('best_x[4]:', min_samples_split)




######################5.使用数据进行预测######################

regressor=RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split, 
                            random_state=90)
                      
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)


print('y_test_pred:', y_test_pred)
print('-------------------')


regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)


######################5.使用训练集数据进行预测######################
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




######################6.使用测试练集数据进行预测######################
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




# 记录结束时间
end_time = datetime.datetime.now()
 
# 计算运行时长
duration = end_time - start_time
 
# 打印运行时长
print("Duration is ",duration)

'''





'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# 网格搜索
from sklearn.model_selection import GridSearchCV
# 随机搜索
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook

from sko.GA import GA  # 导入遗传算法的包

import datetime

import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import pandas as pd

# 记录开始时间
start_time = datetime.datetime.now()

# 定义学习器个数和最大深度的候选列表
n_estimators_list = list(range(1, 101))
max_depth_list = list(range(1, 20))

######################1.导入数据######################
df = pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx', sheet_name='Sheet2')

######################2.提取特征变量######################
x = df.drop(columns='er')
y = df['er']

print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))

######################3.划分训练集和测试集######################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("---------------x_train------------------")
print(x_train)
print("---------------y_train------------------")
print(y_train)
print("---------------x_test------------------")
print(x_test)
print("type of x_test:", type(x_test))
print("---------------y_test------------------")
print(y_test)

def train_rf(x):
    n_estimators_index, max_depth_index = x
    n_estimators = n_estimators_list[int(n_estimators_index)]
    max_depth = max_depth_list[int(max_depth_index)]
    clf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test, y_pred)
    return -score

######################调参######################
ga = GA(func=train_rf, n_dim=2, size_pop=200, max_iter=400, prob_mut=0.01, lb=[0, 0], ub=[len(n_estimators_list) - 1, len(max_depth_list) - 1], precision=1e-5)

best_x, best_y = ga.run()

# 拿到最佳预测值，方便之后做拟合
print('best_x:', best_x, '\n', 'best_y:', -best_y[0])

n_estimators_index = int(best_x[0])
max_depth_index = int(best_x[1])
n_estimators = n_estimators_list[n_estimators_index]
max_depth = max_depth_list[max_depth_index]

print('best_n_estimators:', n_estimators)
print('best_max_depth:', max_depth)

######################5.使用数据进行预测######################
regressor = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)

regressor.fit(x_train, y_train)
y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

######################5.使用训练集数据进行预测######################
MAE_train = metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train = metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train = np.sqrt(MSE_train)
MAPE_train = metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train = metrics.r2_score(y_train, y_train_pred)
EV_train = metrics.explained_variance_score(y_train, y_train_pred)

print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('MAPE_train:', MAPE_train)
print('r2_score_train:', R2_train)
print('EV_train:', EV_train)
print('-------------------')

######################6.使用测试练集数据进行预测######################
MAE_test = metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test = metrics.mean_squared_error(y_test, y_test_pred)
RMSE_test = np.sqrt(MSE_test)
MAPE_test = metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test = metrics.r2_score(y_test, y_test_pred)
EV_test = metrics.explained_variance_score(y_test, y_test_pred)

print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('MAPE_test:', MAPE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)
print('-------------------')

# 记录结束时间
end_time = datetime.datetime.now()

# 计算运行时长
duration = end_time - start_time

# 打印运行时长
print("Duration is ", duration)

'''
