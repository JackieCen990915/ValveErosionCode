
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR

from openpyxl import Workbook


import shap



######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/弯头冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand.xlsx',sheet_name='Sheet1_er_kg')







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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)



##### 将10%(6个)异常值添加到训练集#####################################
'''
random_selected_train_index=[19, 25, 9, 61, 32, 38]
random_multiple=[3.26035398,1.78945897,3.71130498,3.8388549,3.95684917,4.00810651]
'''
'''
random_selected_train_index=[4, 23, 7, 64, 33, 73]
random_multiple=[2.24464873,2.94976447,2.00922533,1.71312197,2.34816149,1.92963569]
'''
random_selected_train_index=[61, 72, 74, 10, 11, 53, 25, 4, 44, 56, 36, 26]
random_multiple=[2.04304729
 ,1.35678778
 ,1.54663711
 ,1.38771819
 ,2.52230581
 ,2.61733407
 ,2.21996588
 ,2.25217997
 ,2.09358257
 ,2.56126378
 ,1.81633444
 ,2.4137071 ]

y_train[random_selected_train_index]= y_train[random_selected_train_index]*random_multiple





# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



'''
param_grid={
    'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01,0.1,0.2,0.5,0.8,1,5,10,25,50,75,100],
    'gamma': [0.01,0.05,0.1,0.2,0.5,0.8,1],
    'epsilon': [0.01,0.05,0.1,0.2,0.5,0.8,1],
}


k=5

regressor=SVR()


grid_search=GridSearchCV(regressor, param_grid, cv=k, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_params=grid_search.best_params_
best_model=grid_search.best_estimator_
print("best_params: ", best_params)
print("best_model: ", best_model)
'''

'''
best_params:  {'C': 100, 'epsilon': 0.01, 'gamma': 0.2, 'kernel': 'poly'}
best_model:  SVR(C=100, epsilon=0.01, gamma=0.2, kernel='poly')
'''


#regressor=best_model

regressor=SVR(kernel='poly',
              C=100,
              gamma=0.2,
              epsilon=0.01)

regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)




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
excel_file_path = 'SVR_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')

'''



######################6.评估模型(训练集)######################

MAE_train=metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
MAPE_train=metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train=metrics.r2_score(y_train, y_train_pred)

print('----------------------------------')
print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('MAPE_train:', MAPE_train)
print('r2_score_train:', R2_train)






######################6.评估模型(测试集)######################

MAE_test=metrics.mean_absolute_error(y_test, y_test_pred)
MSE_test=metrics.mean_squared_error(y_test, y_test_pred)
RMSE_test=np.sqrt(MSE_test)
MAPE_test=metrics.mean_absolute_percentage_error(y_test, y_test_pred)
R2_test=metrics.r2_score(y_test, y_test_pred)
EV_test=metrics.explained_variance_score(y_test, y_test_pred)

print('----------------------------------')
print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('MAPE_test:', MAPE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)



'''
----------------------------------
MAE_train: 0.06294477349801314
MSE_train: 0.06683174506886652
RMSE_train: 0.25851836505143405
MAPE_train: 1523.8401617072755
r2_score_train: 0.42897100921091325
----------------------------------
MAE_test: 0.027062831278390258
MSE_test: 0.0026214528716100487
RMSE_test: 0.05120012569916258
MAPE_test: 238.49305433238948
r2_score_test: 0.841723927142075
EV_test: 0.8418416726655578
'''
'''
----------------------------------
MAE_train: 0.03790857548310093
MSE_train: 0.008775558616008433
RMSE_train: 0.09367795160019476
MAPE_train: 2012.9511561190586
r2_score_train: 0.8008847694431495
----------------------------------
MAE_test: 0.030364426681566015
MSE_test: 0.0030425312867862616
RMSE_test: 0.05515914508752163
MAPE_test: 693.679669399852
r2_score_test: 0.8163003772316024
EV_test: 0.8166976012209386
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
'''
hv: 0.029247056737676184
D: 0.002068211926227043
R/D: 0.005816812734174539
dp: 0.01724903001018529
fp: 0.024838209018354127
u0: 0.06833400273716454
'''


