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
random_selected_train_index=[4, 23, 7, 64, 33, 73]
random_multiple=[2.24464873,2.94976447,2.00922533,1.71312197,2.34816149,1.92963569]
'''
'''
random_selected_train_index=[4, 23, 7, 64, 33, 73]
random_multiple=[0.24511851,0.55879534,0.61794635,1.87577546,1.48501098,1.60618524]
'''
random_selected_train_index=[28, 29, 36, 15, 32, 69, 44, 54, 40, 64, 57, 10]
random_multiple=[3.77406847,2.3163381,2.39277276,1.94249484,3.8442951
 ,2.07089691
 ,4.48338812
 ,1.88319388
 ,3.33209855
 ,4.80352563
 ,3.50921626
 ,3.73180566]

y_train[random_selected_train_index]= y_train[random_selected_train_index]*random_multiple




# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)








######################调参######################
param_grid = {
    
       'min_samples_split': np.arange(2,21,1)
    
}


'''
param_grid = {
        'n_estimators': np.arange(1,201,10),
        'max_depth': np.arange(1,21,1),
        'max_features': np.arange(3,21,1),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_leaf': np.arange(1,21,1), 
        'min_samples_split': np.arange(2,21,1)    
    
}

'''






#调参#
scorel=[]


k=5

#{'n_estimators': 47}---0.8161184542105884
'''
regressor=RandomForestRegressor(
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''


#{'max_depth': 13}----0.8166030357904195
'''
regressor=RandomForestRegressor(n_estimators=47,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'max_features': 4}----0.8722192538095855
#{'max_features': 1}---------0.9276049504879426
#{'max_features': 'sqrt'}--------0.8896603286597891
'''
regressor=RandomForestRegressor(n_estimators=47,
                           max_depth=13,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'min_samples_leaf': 1}----0.8722192538095855
#{'min_samples_leaf': 1}--------0.9276049504879426
#{'min_samples_leaf': 1}-----------0.8896603286597891
'''
regressor=RandomForestRegressor(n_estimators=47,
                           max_depth=13,
                           max_features='sqrt',
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'min_samples_split': 2}-----0.8722192538095855
#{'min_samples_split': 2}--------0.9276049504879426
#{'min_samples_split': 3}-----------0.9040946937123374
'''
regressor=RandomForestRegressor(n_estimators=47,
                           max_depth=13,
                           max_features='sqrt',
                           min_samples_leaf=1,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''









######################5.使用测试集数据进行预测######################
regressor=RandomForestRegressor(n_estimators=47,
                           max_depth=13,
                           max_features='sqrt',
                           min_samples_leaf=1,
                           min_samples_split=3,      
                           random_state=90)


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




######################6.使用测试练集数据进行预测######################
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
MAE_train: 0.050565586001584
MSE_train: 0.028199959783207537
RMSE_train: 0.16792843649366695
MAPE_train: 459.4549984744741
r2_score_train: 0.7590517117470962
EV_train: 0.7590532580058884
-------------------
MAE_test: 0.038036962571753534
MSE_test: 0.004539463053847972
RMSE_test: 0.06737553750322124
MAPE_test: 835.8504090870472
r2_score_test: 0.7259197779873046
EV_test: 0.7535387072087335
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











'''
######################7.特征重要性######################

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

'''
