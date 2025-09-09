import pandas as pd
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




##### 将10%(6个)异常值添加到训练集#####################################


random_selected_train_index=[52, 13, 2, 60, 25, 23]
random_multiple=[2.08767221,
 1.3688075 ,
 2.00090929,
 2.65562694,
 1.42969978,
 2.32287012]

y_train[random_selected_train_index]= y_train[random_selected_train_index]*random_multiple



# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)





print("---------------x_train------------------")
print(x_train)
print(type(x_train))
print("---------------x_test------------------")
print(x_test)
print(type(x_test))
print("---------------y_train------------------")
print(y_train)
print(type(y_train))
print("---------------y_test------------------")
print(y_test)
print(type(y_test))


'''
---------------y_test------------------
22    0.015090
0     0.008190
49    0.003840
4     0.020600
54    0.000491
18    0.007440
10    0.023600
33    0.006770
45    0.000186
12    0.007080
31    0.019250
9     0.014050
67    0.000063
5     0.007710
'''



######################5.使用数据进行预测######################

regressor=RandomForestRegressor(n_estimators=10,
                           max_depth=9,
                           max_features='auto',  
                           min_samples_leaf=1,
                           min_samples_split=2,   
                           random_state=42)


#auto----all
print(regressor.get_params()['max_features'])  # 输出默认值

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




######################6.5.使用测试练集数据进行预测######################
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
MAE_train: 0.0012959173355487498
MSE_train: 4.0679345187634745e-06
RMSE_train: 0.002016912124700398
MAPE_train: 1.1495735323692122
r2_score_train: 0.9522556562458401
EV_train: 0.9534084477308693
-------------------
MAE_test: 0.0030415159552035728
MSE_test: 1.539047800921212e-05
RMSE_test: 0.00392306997251032
MAPE_test: 2.3593462215756875
r2_score_test: 0.7238949534588577
EV_test: 0.8739199187622657
-------------------

'''




#######将y_test和y_pred写入xlsx，后续画实际和预测的图用###############
'''
# 创建一个新的Excel工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 写入数据
for col1,col2 in zip(y_test,y_pred):
    sheet.append([col1,col2])

# 指定Excel文件路径
excel_file_path = '0306_RF_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')
'''










######################7.特征重要性######################
'''
#获取特征名称
features=x.columns
#获取特征重要性
importances=regressor.feature_importances_


#通过二维表格形式显示
importances_df=pd.DataFrame()
importances_df['特征名称']=features
importances_df['特征重要性']=importances
importances_df.sort_values('特征重要性',ascending=False)
print(importances_df)
print('-------------------')
'''

'''

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



'''

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
