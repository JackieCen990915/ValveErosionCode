import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from openpyxl import Workbook
import shap



######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand.xlsx',sheet_name='Sheet1')







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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=199)




# 使用MinMaxScaler进行归一化,对结果有影响------------------
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
#k=10
regressor=SVR()
grid_search=GridSearchCV(regressor,
                         param_grid,
                         cv=k,
                         n_jobs=-1)

grid_search.fit(x_train, y_train)
best_params=grid_search.best_params_
best_model=grid_search.best_estimator_
print("best_params: ", best_params)
print("best_model: ", best_model)


regressor=best_model
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


#---90,90--- 
'''
best_params:  {'C': 0.1, 'epsilon': 0.01, 'gamma': 0.2, 'kernel': 'sigmoid'}
best_model:  SVR(C=0.1, epsilon=0.01, gamma=0.2, kernel='sigmoid')
MAE_train: 0.006267192945002157
MSE_train: 4.7392397684382654e-05
RMSE_train: 0.006884213657665097
MAPE_train: 17.388136202204617
r2_score_train: 0.1415398757864842
EV_train: 0.4791589786229472
-------------------
MAE_test: 0.005249324162417321
MSE_test: 3.739029682091716e-05
RMSE_test: 0.006114760569385948
MAPE_test: 20.05312868242159
r2_score_test: 0.23096009569456377
EV_test: 0.6222936888828814
-------------------

'''



 
'''
best_params:  {'C': 0.1, 'epsilon': 0.01, 'gamma': 0.3, 'kernel': 'sigmoid'}
best_model:  SVR(C=0.1, epsilon=0.01, gamma=0.3, kernel='sigmoid')
MAE_train: 0.00576847343209876
MSE_train: 4.241342962669489e-05
RMSE_train: 0.0065125593760590695
MAPE_train: 14.852994464729742
r2_score_train: 0.23172829726544786
EV_train: 0.4605316187292603
-------------------
MAE_test: 0.004282761006673284
MSE_test: 2.7820869834595503e-05
RMSE_test: 0.005274549254163383
MAPE_test: 15.381395356514746
r2_score_test: 0.42778311769586774
EV_test: 0.6364473831475477
-------------------
'''





'''
#########使用 SHAP 模型输出特征重要性##########

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


