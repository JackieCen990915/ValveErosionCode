import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from openpyxl import Workbook
import shap


from sko.GA import GA #导入遗传算法的包

import datetime

# 记录开始时间
start_time = datetime.datetime.now()


######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/0207_最开始70个数据跑阀芯/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx',sheet_name='Sheet2')








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

'''
def train_dt(x):
    max_depth,max_features = x
    regressor=DecisionTreeRegressor(
                               max_depth=int(max_depth),
                               max_features=int(max_features),
                               random_state=100)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''


'''


def train_dt(x):
    max_depth,max_features,min_samples_leaf,min_samples_split = x
    regressor = DecisionTreeRegressor(
                            max_depth=int(max_depth),
                            max_features=int(max_features),
                            min_samples_leaf=int(min_samples_leaf),
                            min_samples_split=int(min_samples_split), 
                            random_state=100)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''

######################调参######################
# 设置要调优的超参数范围
'''
param_grid = {
    'max_features': ['auto', 'sqrt', 'log2', None,4,5]
}
'''


'''
param_grid = {
    'max_depth': np.arange(1,21,1),
    'max_features': ['auto', 'sqrt', 'log2', None,4,5]
    'min_samples_leaf': np.arange(1,21,1),
    'min_samples_split': np.arange(2,21,1)
}
'''


#ga = GA(func=train_dt, n_dim=4,size_pop=200, max_iter=400, prob_mut=0.01,lb=[1,1,1,2], ub=[19,11,9,10], precision=1e-5)

#ga = GA(func=train_dt, n_dim=4,size_pop=200, max_iter=400, prob_mut=0.01,lb=[1,1,1,2], ub=[15,11,10,10], precision=1e-5)
#ga = GA(func=train_dt, n_dim=2,size_pop=200, max_iter=400, prob_mut=0.01,lb=[1,1], ub=[15,11], precision=1e-5)

#ga = GA(func=train_dt, n_dim=4,size_pop=200, max_iter=400, prob_mut=0.01,lb=[1,1,1,2], ub=[20,8,20,20], precision=1e-5)
#ga = GA(func=train_dt, n_dim=4,size_pop=200, max_iter=400, prob_mut=0.01,lb=[1,1,1,2], ub=[15,11,10,10], precision=1e-5)





######################5.使用测试集数据进行预测######################
#print(ga.get_params())
'''
best_x, best_y = ga.run()


print('best_x:', best_x, '\n','best_y:',-best_y[0])


max_depth=int(best_x[0])
max_features=int(best_x[1])
min_samples_leaf=int(best_x[2])
min_samples_split=int(best_x[3])


print('best_x[0]:', max_depth)
print('best_x[1]:', max_features)
print('best_x[2]:', min_samples_leaf)
print('best_x[3]:', min_samples_split)
'''

'''
max_depth=int(best_x[0])
max_features=int(best_x[1])


print('best_x[0]:', max_depth)
print('best_x[1]:', max_features)
'''

######################5.使用数据进行预测######################
'''
regressor=DecisionTreeRegressor(
                            max_depth=9,
                            max_features=6,
                            min_samples_leaf=2,
                            min_samples_split=5, 
                            random_state=100)
'''
'''
regressor=DecisionTreeRegressor(
                            max_depth=max_depth,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split, 
                            random_state=100)
'''

'''
regressor=DecisionTreeRegressor(
                           max_depth=max_depth,
                           max_features=max_features,
                           random_state=100)
'''


regressor=DecisionTreeRegressor(
                            max_depth=13,
                            max_features=6,
                            min_samples_leaf=2,
                            min_samples_split=5,
                            random_state=100)

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
# 记录结束时间
end_time = datetime.datetime.now()
 
# 计算运行时长
duration = end_time - start_time
 
# 打印运行时长
print("Duration is ",duration)
'''



#ga = GA(func=train_dt, n_dim=4,size_pop=200, max_iter=400, prob_mut=0.01,lb=[1,1,1,2], ub=[19,11,9,10], precision=1e-5)
'''
best_x: [9.29175486 6.52322914 2.47120044 5.59389076] 
 best_y: 0.9804181076618754
best_x[0]: 9
best_x[1]: 6
best_x[2]: 2
best_x[3]: 5
MAE_train: 0.0012176481547619047
MSE_train: 3.3542183093721725e-06
RMSE_train: 0.0018314525135455116
MAPE_train: 0.7679479162730988
r2_score_train: 0.9362260341179497
EV_train: 0.9362260341179497
-------------------
MAE_test: 0.000853135059523809
MSE_test: 1.0915218217996517e-06
RMSE_test: 0.0010447592171403187
MAPE_test: 0.3910523317332793
r2_score_test: 0.9804181076618754
EV_test: 0.9814430156539368
-------------------
Duration is  0:01:02.118232
'''


'''
best_x: [7.87979359 3.60612736] 
 best_y: 0.9403788205821425
MAE_train: 0.0003072916666666666
MSE_train: 4.122892113095238e-07
RMSE_train: 0.0006420975091911849
MAPE_train: 0.043631134270459934
r2_score_train: 0.9921611190237309
EV_train: 0.9921611190237309
-------------------
MAE_test: 0.0013388021428571428
MSE_test: 3.323367182921429e-06
RMSE_test: 0.001823010472521052
MAPE_test: 0.6434625968078025
r2_score_test: 0.9403788205821425
EV_test: 0.9419034984645887
-------------------
Duration is  0:01:00.785053
'''

#######将测试集的反归一化的输入参数和输出参数（预测值和真实值）写入xlsx文件###############
# 反归一化 x_test
x_test_inverse = scaler.inverse_transform(x_test)
x_test_inverse_df = pd.DataFrame(x_test_inverse, columns=x.columns)

# 创建一个新的Excel工作簿
workbook = Workbook()
# 获取默认的工作表
sheet = workbook.active

# 写入表头
sheet.append(list(x_test_inverse_df.columns) + ['y_test', 'y_test_pred'])

# 写入数据
for x_row, y_true, y_pred in zip(x_test_inverse_df.values, y_test, y_test_pred):
    sheet.append(list(x_row) + [y_true, y_pred])

# 指定Excel文件路径
excel_file_path = '0313_阀门最优预测模型_测试集_反归一化_输入输出参数.xlsx'
# 保存工作簿到文件
workbook.save(excel_file_path)
print(f'Data has been written to {excel_file_path}')