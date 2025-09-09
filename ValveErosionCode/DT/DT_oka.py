import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler



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



# 4.使用MinMaxScaler进行归一化------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)







######################调参######################
# 设置要调优的超参数范围

param_grid = {
    
       'max_depth': [1,3,5,7,9,11,13,15,17,19]
    
}


'''
param_grid = {
    'max_depth': np.arange(1,21,1),
    'max_features': ['auto', 'sqrt', 'log2', None,4,5],
    'min_samples_leaf': np.arange(1,21,1),
    'min_samples_split': np.arange(2,21,1)
}
'''









#调参#
scorel=[]
k=5
regressor=DecisionTreeRegressor(
                               random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)


'''
regressor=DecisionTreeRegressor(
                               max_depth=7,
                               random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

'''
regressor=DecisionTreeRegressor(
                               max_depth=7,
                               max_features=5,
                               random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


'''
regressor=DecisionTreeRegressor(
                               max_depth=7,
                               max_features=5,
                               min_samples_leaf=1,
                               random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


######################构建预测模型######################
'''
# 初始化决策树回归模型
regressor = DecisionTreeRegressor()
# 初始化网格搜索对象
grid_search = GridSearchCV(regressor, param_grid, cv=5)  # cv表示交叉验证的折数
# 执行网格搜索
grid_search.fit(x_train, y_train)
# 打印最佳参数组合
print("最佳参数组合:", grid_search.best_params_)
# 获取最佳模型
best_model = grid_search.best_estimator_
regressor=best_model

'''
'''
regressor=DecisionTreeRegressor(
                               max_depth=7,
                               max_features=5,
                               min_samples_leaf=1,
                               min_samples_split=2,
                               random_state=100)

regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)

'''

'''

######################评估模型(训练集)######################
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
R2_train=metrics.r2_score(y_train, y_train_pred)
EV_train=metrics.explained_variance_score(y_train, y_train_pred)

print('RMSE_train:', RMSE_train)
print('r2_score_train:', R2_train)
print('EV_train:', EV_train)
print('-------------------')



######################评估模型(测试集)######################
MSE_test=metrics.mean_squared_error(y_test, y_test_pred)
RMSE_test=np.sqrt(MSE_test)
R2_test=metrics.r2_score(y_test, y_test_pred)
EV_test=metrics.explained_variance_score(y_test, y_test_pred)

print('RMSE_test:', RMSE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)
print('-------------------')
'''