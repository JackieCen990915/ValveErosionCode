
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
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/弯头冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand_oka.xlsx',sheet_name='Sheet1_er_kg')






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
random_selected_train_index=[4, 23, 7, 64, 33, 73]
random_multiple=[2.24464873,2.94976447,2.00922533,1.71312197,2.34816149,1.92963569]
'''
'''
random_selected_train_index=[4, 23, 7, 64, 33, 73]
random_multiple=[0.24511851,0.55879534,0.61794635,1.87577546,1.48501098,1.60618524]
'''
'''
random_selected_train_index=[39, 14, 47, 16, 66, 26]
random_multiple=[3.96993843,
 1.98953235,
 1.85345469,
 4.74959304,
 2.10447381,
 3.72937687]

y_train[random_selected_train_index]= y_train[random_selected_train_index]*random_multiple
'''

##### 将20%(12个)异常值添加到训练集#####################################
random_selected_train_index=[13, 33, 58, 73, 54, 24, 38, 16, 65, 68, 40, 14]
random_multiple=[1.5750534,
 2.27984681,
 3.60275699,
 3.96971608,
 2.93352731,
 2.393262  ,
 2.56428517,
 3.55800567,
 2.2032054 ,
 3.60821066,
 2.51886394,
 34.99522315]

y_train[random_selected_train_index]=y_train[random_selected_train_index]*random_multiple

'''
MAE_test: 0.013020167917743065
MSE_test: 0.0005578562808348095
RMSE_test: 0.02361898136742585
MAPE_test: 302.5577162827205
r2_score_test: 0.9663181809194868
EV_test: 0.9688040306242154
'''

# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)




param_grid={
    'kernel':['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.01,0.1,0.2,0.5,0.8,1,5,10,25,50,75,100],
    'gamma': [0.01,0.05,0.1,0.2,0.5,0.8,1],
    'epsilon': [0.01,0.05,0.1,0.2,0.5,0.8,1],
}


#k=5
c=50
gamma=1
epsilon=0.0001

regressor=SVR(kernel='rbf',C=c,gamma=gamma,epsilon=epsilon)

#regressor=SVR(kernel='rbf',C=0.8, epsilon=0.01, gamma=1)
#regressor=SVR()
'''
grid_search=GridSearchCV(regressor, param_grid, cv=k, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_params=grid_search.best_params_
best_model=grid_search.best_estimator_
print("best_params: ", best_params)
print("best_model: ", best_model)
'''


'''
best_params:  {'C': 50, 'epsilon': 0.01, 'gamma': 1, 'kernel': 'rbf'}
best_model:  SVR(C=50, epsilon=0.01, gamma=1)
'''




#regressor=best_model
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)





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
print('-------------------')


'''
----------------------------------
MAE_train: 0.018865423011611315
MSE_train: 0.005936809309156118
RMSE_train: 0.07705069311275609
MAPE_train: 28.12679546590069
r2_score_train: 0.8652952813501507
----------------------------------
MAE_test: 0.021957266355787274
MSE_test: 0.001856398535122525
RMSE_test: 0.04308594359095
MAPE_test: 1498.877290197999
r2_score_test: 0.8879157916663454
EV_test: 0.900092203196217
-------------------
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
hv: 0.027544260746538766
D: 0.006211738407955177
R/D: 0.02008212700032427
dp: 0.04805852717070106
fp: 0.028802900537548384
u0: 0.040914842731830985
(hv)^k1: 0.033245663141849034

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
excel_file_path = 'SVR_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')

'''






