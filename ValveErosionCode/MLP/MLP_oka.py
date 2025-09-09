import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 


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


k=5
#隐藏层节点个数----4-13

#隐藏层节点个数----5-14----sqrt(1+11)+1-10
param_grid={
        'hidden_layer_sizes':np.arange(5,15,1),
        'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
        'activation':['sigmoid','tanh','relu'],
}

'''
param_grid={
        'alpha':[0.01,0.05,0.1],
        'hidden_layer_sizes':np.arange(4,14,1),
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],
}
'''

'''
param_grid={
        'hidden_layer_sizes':np.arange(4,14,1),
        'max_iter':np.arange(1,501,100),
        'alpha':np.linspace(0.1,1,10)
}
'''
'''
param_grid={
        'hidden_layer_sizes':np.arange(4,13,1),
        'alpha':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],
}
'''
'''
param_grid={
        'hidden_layer_sizes':np.arange(4,13,1),
        'alpha':[0.01,0.05,0.1],
        'max_iter':np.arange(1,501,50),
        'activation':['sigmoid','tanh','relu'],
}
'''
'''
param_grid={
        'hidden_layer_sizes':np.arange(4,13,1),
        'max_iter':np.arange(1,501,50),
}
'''
        
regressor=MLPRegressor(
                        solver='lbfgs',
                        random_state=90)



GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)


regressor=GS.best_estimator_
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
