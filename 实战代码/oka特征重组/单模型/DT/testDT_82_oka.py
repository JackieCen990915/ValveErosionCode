import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt



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
#k=10
#{'max_depth': 6}---0.7695156860937801
#{'max_depth': 11}--0.793440455123757---90,90--数据集加入新的重构特征
#{'max_depth': 10}---0.7802474187963995--------90，42--数据集加入新的重构特征


#{'max_depth': 7}---0.7827986551875634--------42，100--数据集加入新的重构特征---'max_depth': [1,3,5,7,9,11,13,15,17,19]


regressor=DecisionTreeRegressor(
                               random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)



#{'max_features': 'auto'}---0.7695156860937801--不变--默认取值
#{'max_features': 'auto'}
#{'max_features': 'auto'}
#{'max_features': 5}----0.81238292724201

#{'max_features': 5}
'''
regressor=DecisionTreeRegressor(
                               max_depth=7,
                               random_state=100)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'min_samples_leaf': 2}---0.827292310370126
#{'min_samples_leaf': 1}
#{'min_samples_leaf': 1}

#{'min_samples_leaf': 1}
'''
regressor=DecisionTreeRegressor(
                               max_depth=7,
                               max_features=5,
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'min_samples_split': 5}---0.828410424619819
#{'min_samples_split': 3}---0.8007262678852521
#{'min_samples_split': 3}----0.7971145687752338


#{'min_samples_split': 2}
'''
regressor=DecisionTreeRegressor(
                               max_depth=7,
                               max_features=5,
                               min_samples_leaf=1,
                               random_state=42)
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
#---90,90--- 数据集加入新的重构特征
regressor=DecisionTreeRegressor(
                               max_depth=11,
                               max_features='auto',
                               min_samples_leaf=1,
                               min_samples_split=3,
                               random_state=90)
'''
#---42,100--- 数据集加入新的重构特征----'max_depth': [1,3,5,7,9,11,13,15,17,19]
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


######################评估模型(测试集)######################
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
print('----------------------------------')
'''



'''
MAE_train: 0.0008365954336734694
MSE_train: 1.2755796044914543e-06
RMSE_train: 0.001129415603084823
MAPE_train: 0.497490072813632
r2_score_train: 0.9773774066931575
EV_train: 0.9773774066931575
-------------------
MAE_test: 0.0018556611224489793
MSE_test: 8.31297737404757e-06
RMSE_test: 0.0028832234346383165
MAPE_test: 1.5506440621241633
r2_score_test: 0.7883123760121258
EV_test: 0.7944530450332241
----------------------------------
'''
#---90,90--- 数据集加入新的重构特征

'''

MAE_train: 0.0004200375
MSE_train: 7.292158450892859e-07
RMSE_train: 0.0008539413592801826
MAPE_train: 0.09277963174505191
r2_score_train: 0.9867910729243374
EV_train: 0.9867910729243374
-------------------
MAE_test: 0.0006376299999999999
MSE_test: 7.965335481000001e-07
RMSE_test: 0.0008924872817581213
MAPE_test: 0.36165426776080956
r2_score_test: 0.983616977245706
EV_test: 0.9844493708419866
----------------------------------
'''


#---90,42--- 数据集加入新的重构特征
'''
MAE_train: 0.0004200375
MSE_train: 7.292158450892859e-07
RMSE_train: 0.0008539413592801826
MAPE_train: 0.09277963174505191
r2_score_train: 0.9867910729243374
EV_train: 0.9867910729243374
-------------------
MAE_test: 0.0006376299999999999
MSE_test: 7.965335481000001e-07
RMSE_test: 0.0008924872817581213
MAPE_test: 0.36165426776080956
r2_score_test: 0.983616977245706
EV_test: 0.9844493708419866
----------------------------------
'''
#---42,42--- 数据集加入新的重构特征----'max_depth': [1,3,5,7,9,11,13,15,17,19]
'''
MAE_train: 0.00033834192176870746
MSE_train: 4.999524982631803e-07
RMSE_train: 0.000707073191588523
MAPE_train: 0.07501183001984023
r2_score_train: 0.9904943713777384
EV_train: 0.9904943713777384
-------------------
MAE_test: 0.0020500796938775516
MSE_test: 8.805964678888388e-06
RMSE_test: 0.0029674845709604606
MAPE_test: 1.1455167575328904
r2_score_test: 0.8420210674386583
EV_test: 0.8649775661850202
----------------------------------
'''


#---42,100--- 数据集加入新的重构特征----'max_depth': [1,3,5,7,9,11,13,15,17,19]
'''
MAE_train: 0.000327202380952381
MSE_train: 8.928842261904762e-07
RMSE_train: 0.0009449255135673268
MAPE_train: 0.05139058260321556
r2_score_train: 0.9830235354632147
EV_train: 0.9830235354632147
-------------------
MAE_test: 0.001604552142857143
MSE_test: 4.1096514888738085e-06
RMSE_test: 0.0020272275375186203
MAPE_test: 0.6775957750873035
r2_score_test: 0.9262728867209846
EV_test: 0.9275054811131334
----------------------------------
'''


#########使用 SHAP 模型输出特征重要性##########
'''
# 计算 SHAP 值
explainer = shap.KernelExplainer(regressor.predict,x_test)
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



