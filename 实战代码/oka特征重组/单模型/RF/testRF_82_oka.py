import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook

import math

import shap


from lime.lime_tabular import LimeTabularExplainer


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












######################调参######################

param_grid = {

    
         'n_estimators':[10,30,50,70,100],
     
          
}



'''
param_grid = {
        'n_estimators': np.arange(1,101,1),
        'n_estimators': np.arange(1,201,10),
        'max_depth': np.arange(1,21,1),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_leaf': np.arange(1,21,1), 
        'min_samples_split': np.arange(2,21,1)   
    
}

'''


#调参#
scorel=[]


k=5
#k=10
#对于回归模型，默认的scoring指标是R2
#{'n_estimators': 350}--0.8487084357320859---90,90--- 数据集加入新的重构特征---'n_estimators': np.arange(10,501,10)
#{'n_estimators': 91}----0.8425253102313814----90,42-----'n_estimators': np.arange(1,101,10)


#{'n_estimators': 10}---0.875664974240982----42,42-----'n_estimators': [10,30,50,70,100]






regressor=RandomForestRegressor(
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)






#{'max_depth': 8}---0.8497194034464173
#{'max_depth': 7}---0.8429951818967693


#{'max_depth': 9}---0.8780146019645493
'''
regressor=RandomForestRegressor(n_estimators=10,
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

#{'max_features': 'auto'}---不变--默认取值
#{'max_features': 'auto'}


#{'max_features': 'auto'}
'''
regressor=RandomForestRegressor(n_estimators=10,
                               max_depth=9,
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'min_samples_leaf': 1}---不变--默认取值
#{'min_samples_leaf': 1}


#{'min_samples_leaf': 1}
'''
regressor=RandomForestRegressor(n_estimators=10,
                               max_depth=9,
                               max_features='auto',
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'min_samples_split': 2}---不变--默认取值
#{'min_samples_split': 3}---0.8516396964229205

#{'min_samples_split': 2}
'''
regressor=RandomForestRegressor(n_estimators=10,
                               max_depth=9,
                               max_features='auto',  
                               min_samples_leaf=1,
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

######################构建预测模型#####################
#--90,90--- 数据集加入新的重构特征---'n_estimators': np.arange(10,501,10)
'''
regressor=RandomForestRegressor(n_estimators=350,
                               max_depth=8,
                               max_features='auto',  
                               min_samples_leaf=1,
                               min_samples_split=2,   
                               random_state=90)
'''
#--90,42--- 数据集加入新的重构特征---'n_estimators': np.arange(1,101,10)
'''
regressor=RandomForestRegressor(n_estimators=91,
                               max_depth=7,
                               max_features='auto',  
                               min_samples_leaf=1,
                               min_samples_split=3,   
                               random_state=42)
'''

regressor=RandomForestRegressor(n_estimators=10,
                               max_depth=9,
                               max_features='auto',  
                               min_samples_leaf=1,
                               min_samples_split=2,   
                               random_state=42)
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)



######################使用训练集数据进行预测######################
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




######################使用测试练集数据进行预测######################
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




#--90,90--- 数据集加入新的重构特征---'n_estimators': np.arange(10,501,10)
'''

MAE_train: 0.0006074140026053864
MSE_train: 7.2358796208493e-07
RMSE_train: 0.0008506397369538588
MAPE_train: 0.7833328254981683
r2_score_train: 0.9868930157121902
EV_train: 0.9868961781751531
-------------------
MAE_test: 0.0008252747292310819
MSE_test: 1.187910951681615e-06
RMSE_test: 0.0010899132771379633
MAPE_test: 1.8007191588268703
r2_score_test: 0.9755671657547916
EV_test: 0.9781755244343759
-------------------
'''


#--90,42--- 数据集加入新的重构特征---'n_estimators': np.arange(1,101,10)
'''
MAE_train: 0.0007480026859818183
MSE_train: 1.1476379901485907e-06
RMSE_train: 0.001071278670630845
MAPE_train: 0.98710519979623
r2_score_train: 0.9792118251088241
EV_train: 0.9792354444003306
-------------------
MAE_test: 0.0007857192334479589
MSE_test: 1.0706622887199606e-06
RMSE_test: 0.0010347281230931923
MAPE_test: 2.099181853632563
r2_score_test: 0.9779787245871765
EV_test: 0.9813060529331129
-------------------
'''

#--42,42--- 数据集加入新的重构特征---'n_estimators': [10,30,50,70,100]
'''
MAE_train: 0.0007094488988095241
MSE_train: 1.0517293984472725e-06
RMSE_train: 0.001025538589448136
MAPE_train: 0.7744706273865525
r2_score_train: 0.9800034021082307
EV_train: 0.980306992989975
-------------------
MAE_test: 0.0014365156190476185
MSE_test: 3.903858083046789e-06
RMSE_test: 0.0019758183325009385
MAPE_test: 1.3928110052324845
r2_score_test: 0.929964818697348
EV_test: 0.9382987625845607
-------------------
'''



'''
#########使用 SHAP 模型输出特征重要性##########
#测试集
# 计算 SHAP 值
explainer = shap.TreeExplainer(regressor)
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

#训练集
'''
# 计算 SHAP 值
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(x_train)

# 获取特征重要性
feature_importances = np.abs(shap_values).mean(axis=0)
feature_names = x.columns

# 打印特征重要性
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance}")
    
shap.summary_plot(shap_values, x_train, feature_names=feature_names)
shap.summary_plot(shap_values, x_train, feature_names=feature_names, plot_type='bar')
'''



#########使用LIME模型输出特征重要性##########
'''
# Initialize a LimeTabularExplaine
explainer = LimeTabularExplainer(training_data=x_train, mode="regression")
 
# Select a sample instance for explanation
sample_instance = x_test[0]
print(x_test[0])
 
# Explain the prediction for the sample instance
explaination = explainer.explain_instance(sample_instance, regressor.predict)
 
# show the result of the model's explaination 
explaination.show_in_notebook(show_table = True, show_all = False)
#print(explaination.as_list())

'''
