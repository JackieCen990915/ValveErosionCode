import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR




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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=99)






# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


'''
param_grid={
    'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,
          0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
          1,2,3,4,5,6,7,8,9,
          10,30,50,70,100],

    
    'gamma': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'epsilon': [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
}

'''

'''
param_grid={
    'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,5,10,25,50,75,100],
    'gamma': [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    'epsilon': [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
}
'''


param_grid={
    'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01,0.1,0.2,0.5,0.8,1,5,10,25,50,75,100],
    'gamma': [0.01,0.05,0.1,0.2,0.5,0.8,1],
    'epsilon': [0.01,0.05,0.1,0.2,0.5,0.8,1]
}




k=5
regressor=SVR()
grid_search=GridSearchCV(regressor,
                         param_grid,
                         cv=k,
                         n_jobs=-1,
                         scoring='r2')
grid_search.fit(x_train, y_train)
best_params=grid_search.best_params_
best_model=grid_search.best_estimator_
print("best_params: ", best_params)
print("best_model: ", best_model)


regressor=best_model
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


