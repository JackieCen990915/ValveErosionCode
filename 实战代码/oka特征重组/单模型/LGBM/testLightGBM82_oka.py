
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor



from openpyxl import Workbook



######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx',sheet_name='Sheet1')






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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)




# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)






param_grid={

    
         'n_estimators': np.arange(1,101,1)


         
}


'''
param_grid = {
        'n_estimators': np.arange(1,101,1),
        'max_depth': np.arange(1,21,1),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_samples_leaf': np.arange(1,21,1), 
        'min_samples_split': np.arange(2,21,1)    
}

'''


#调参#
scorel=[]

k=5
#{'n_estimators': 16}----0.3701839477915015
regressor=LGBMRegressor(
                   learning_rate=0.1,
                   random_state=123)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)




#{'max_depth': 3}---0.3701839477915015
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''

#{'num_leaves': 3}---0.3701839477915015
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''

#{'min_data_in_leaf': 3}---0.7469612683883997
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''


#{'max_bin': 20}----0.7798000151024558
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''




#{'feature_fraction': 1.0}----0.7798000151024558
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''


#{'bagging_fraction': 0.1}---0.7798000151024558
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   feature_fraction=1,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''

#{'bagging_freq': 0}---0.7798000151024558
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   feature_fraction=1,
                   bagging_fraction=0.1,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

#{'reg_alpha': 0.0}---0.7798000151024558
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   feature_fraction=1,
                   bagging_fraction=0.1,
                   bagging_freq=0,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

#{'reg_lambda': 0.0}----0.7798000151024558
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   feature_fraction=1,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0,    
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''




#{'min_split_gain': 0.0}---0.7798000151024558
'''
regressor=LGBMRegressor(n_estimators=16,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   feature_fraction=1,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0,      
                   reg_lambda=0,      
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''

#{'learning_rate': 0.6000000000000001}---0.8934489213760604
'''
regressor=LGBMRegressor(n_estimators=16,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   feature_fraction=1,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0,      
                   reg_lambda=0,
                   min_split_gain=0,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''
'''

######################预测######################
regressor=LGBMRegressor(
                   learning_rate=0.6,                        
                   n_estimators=16,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=3,
                   max_bin=20,     
                   feature_fraction=1,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0,      
                   reg_lambda=0,
                   min_split_gain=0,     
                   random_state=90)

regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)

'''






'''

######################6.评估模型(训练集)######################

MAE_train=metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
MAPE_train=metrics.mean_absolute_percentage_error(y_train, y_train_pred)
R2_train=metrics.r2_score(y_train, y_train_pred)


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


print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('MAPE_test:', MAPE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)



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
excel_file_path = 'LGBM_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')

'''



 
