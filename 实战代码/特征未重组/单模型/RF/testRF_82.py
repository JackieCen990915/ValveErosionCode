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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)







# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)












######################调参######################
param_grid = {

    
    
          'max_depth': [1,3,5,7,9,11,13,15,17,19],
    
    
}


'''
param_grid = {
        'n_estimators': np.arange(1,101,1),
        'n_estimators': [10,30,50,70,100]
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

#{'n_estimators': 161}--0.8708186966642992---90,90----'n_estimators': np.arange(1,501,10)
#{'n_estimators': 60}---0.8704360562153772------90,90----'n_estimators': np.arange(10,501,10)


#{'n_estimators': 91}---0.8636059242709736-----90,42----'n_estimators': np.arange(1,101,10)

#{'n_estimators': 10}----0.8841637328330485----42,42----'n_estimators':  [10,30,50,70,100]



'''

regressor=RandomForestRegressor(
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'max_depth': 7}---0.8734882332105771
#{'max_depth': 7}-----0.8739487700616276



#{'max_depth': 9}--0.8847004175214161
'''
regressor=RandomForestRegressor(n_estimators=10,
                               random_state=42)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'max_features': 'auto'}---0.8734882332105771---不变--默认取值
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


#{'min_samples_leaf': 1}-----0.8734882332105771---不变--默认取值
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

#{'min_samples_split': 2}----0.8734882332105771---不变--默认取值
#{'min_samples_split': 2}


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






######################构建预测模型######################
#--90,90--- 数据集加入新的重构特征---'n_estimators': np.arange(10,501,10)
'''
regressor=RandomForestRegressor(n_estimators=60,
                               max_depth=7,
                               max_features='auto',
                               min_samples_leaf=1,
                               min_samples_split=2,      
                               random_state=90)
'''
#--90,90--- 数据集加入新的重构特征---'n_estimators': np.arange(1,101,10)
'''
regressor=RandomForestRegressor(n_estimators=91,
                               max_depth=7,
                               max_features='auto',
                               min_samples_leaf=1,
                               min_samples_split=2,      
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





#----90,90-----'n_estimators': np.arange(10,501,10)
'''
MAE_train: 0.0006294098444264059
MSE_train: 7.554048732805311e-07
RMSE_train: 0.0008691403070163822
MAPE_train: 0.9952789626994337
r2_score_train: 0.986316688054768
EV_train: 0.9864571278995214
-------------------
MAE_test: 0.0008552268479437213
MSE_test: 1.3103828681632223e-06
RMSE_test: 0.0011447195587405774
MAPE_test: 1.8756057412684846
r2_score_test: 0.9730481755637743
EV_test: 0.9752837045657528
-------------------
'''
#----90,42-----'n_estimators': np.arange(1,101,10)
'''
MAE_train: 0.0006251007966381686
MSE_train: 7.863058635920852e-07
RMSE_train: 0.0008867388925676403
MAPE_train: 0.8332705694257861
r2_score_train: 0.9857569512767759
EV_train: 0.9857727750100334
-------------------
MAE_test: 0.0007956207085544852
MSE_test: 1.206408788981107e-06
RMSE_test: 0.0010983664183600603
MAPE_test: 2.217388956362585
r2_score_test: 0.9751867040779348
EV_test: 0.9785351683637303
-------------------
'''

#----42,42-----'n_estimators':  [10,30,50,70,100]
'''
MAE_train: 0.0007057015714285718
MSE_train: 1.1103403228200525e-06
RMSE_train: 0.0010537268729704356
MAPE_train: 0.5455101028225932
r2_score_train: 0.9788890288783126
EV_train: 0.9790041847552194
-------------------
MAE_test: 0.0012510044047619043
MSE_test: 4.39127119730496e-06
RMSE_test: 0.0020955360167043083
MAPE_test: 0.8436127522954732
r2_score_test: 0.921220631511189
EV_test: 0.9213853032582059
-------------------

'''



#######将y_test和y_pred写入xlsx，后续画实际和预测的图用###############
'''
# 创建一个新的Excel工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 写入数据
for col1,col2 in zip(y_test,y_test_pred):
    sheet.append([col1,col2])

# 指定Excel文件路径
excel_file_path = '1126_RF_yTest_yPred_gs.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')

'''






