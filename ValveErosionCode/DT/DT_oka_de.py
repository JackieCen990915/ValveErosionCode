import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import csv
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
#导入差分进化算法
from sko.DE import DE
import datetime

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







# 使用MinMaxScaler进行归一化
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



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









######################调参######################
'''
param_grid = {
        'n_estimators': np.arange(1,201,10),
        'max_depth': np.arange(1,21,1),
        'max_features': np.arange(3,21,1),
        'min_samples_leaf': np.arange(1,21,1), 
        'min_samples_split': np.arange(2,21,1)    
    
}

'''
'''
param_grid = {
         'min_samples_split': np.arange(2,21,1),       
}

'''
A_DE = DE(func=train_dt, n_dim=4, size_pop=200, max_iter=400,prob_mut=0.7,lb=[1,1,1,2], ub=[19,11,9,10])

#A_DE = DE(func=train_dt, n_dim=4, size_pop=200, max_iter=400,prob_mut=0.7,lb=[1,1,1,2], ub=[15,11,10,10])
#A_DE = DE(func=train_dt, n_dim=2, size_pop=200, max_iter=400,prob_mut=0.7,lb=[1,1], ub=[15,11])


#A_DE = DE(func=train_dt, n_dim=2, size_pop=50, max_iter=50,prob_mut=0.3,lb=[10,1], ub=[100,20])
#A_DE = DE(func=train_dt, n_dim=2, size_pop=50, max_iter=50,prob_mut=0.5,lb=[10,1], ub=[100,20])
#A_DE = DE(func=train_dt, n_dim=2, size_pop=200, max_iter=400,prob_mut=0.5,lb=[1,1], ub=[100,20])
#A_DE = DE(func=train_dt, n_dim=4, size_pop=200, max_iter=400,prob_mut=0.7,lb=[1,1,1,2], ub=[100,20,20,20])
#A_DE = DE(func=train_dt, n_dim=5, size_pop=200, max_iter=400,prob_mut=0.7,lb=[1,1,1,1,2], ub=[100,20,8,20,20])




best_x, best_y = A_DE.run()#运行算法
print('best_x:', best_x, '\n', 'best_y:', best_y)



max_depth=int(best_x[0])
max_features=int(best_x[1])
min_samples_leaf=int(best_x[2])
min_samples_split=int(best_x[3])


print('best_x[0]:', max_depth)
print('best_x[1]:', max_features)
print('best_x[2]:', min_samples_leaf)
print('best_x[3]:', min_samples_split)

'''
max_depth=int(best_x[0])
max_features=int(best_x[1])


print('best_x[0]:', max_depth)
print('best_x[1]:', max_features)
'''

######################5.使用数据进行预测######################
regressor=DecisionTreeRegressor(
                            max_depth=max_depth,
                            max_features=max_features,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split, 
                            random_state=100)

'''
regressor=DecisionTreeRegressor(
                           max_depth=max_depth,
                           max_features=max_features,
                           random_state=100)
'''
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)




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



# 记录结束时间
end_time = datetime.datetime.now()
 
# 计算运行时长
duration = end_time - start_time
 
# 打印运行时长
print("Duration is ",duration)

