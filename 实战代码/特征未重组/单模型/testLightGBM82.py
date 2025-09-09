
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
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)

# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)











param_grid={
    
       'learning_rate': np.linspace(0,1,21),
      
}




#调参#
scorel=[]

k=5
#n_estimators
#{'n_estimators': 21}---0.32098013714765344
#{'n_estimators': 201}----0.44403153766198955
'''
regressor=LGBMRegressor(
                   learning_rate=0.1,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'max_depth': 3}---0.44403153766198955
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''





#{'num_leaves': 3}---0.32098013714765344
#{'num_leaves': 3}----0.44403153766198955
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''





#{'min_data_in_leaf': 1}---0.6778346643886456
#{'min_data_in_leaf': 11}---0.5145844787025473
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'max_bin': 15}----0.7208043093832325
#{'max_bin': 15}------0.5423541651329173
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'feature_fraction': 1.0}---0.7208043093832325
#{'feature_fraction': 0.30000000000000004}----0.5430583658693724
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'bagging_fraction': 0.1}----0.7208043093832325
#{'bagging_fraction': 0.1}-------0.5430583658693724
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15, 
                   feature_fraction=0.3,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'bagging_freq': 0}----0.7208043093832325
#{'bagging_freq': 0}------0.5430583658693724
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15, 
                   feature_fraction=0.3, 
                   bagging_fraction=0.1,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'reg_alpha': 0.1}----0.7235358218144589
#{'reg_alpha': 0.0}------0.5430583658693724
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15, 
                   feature_fraction=0.3, 
                   bagging_fraction=0.1,
                   bagging_freq=0,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'reg_lambda': 0.4}----0.7271914592939538
#{'reg_lambda': 0.2}------0.543866340202812
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15, 
                   feature_fraction=0.3, 
                   bagging_fraction=0.1,
                   bagging_freq=0, 
                   reg_alpha=0,    
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'min_split_gain': 0.0}---0.7271914592939538
#{'min_split_gain': 0.0}--------0.543866340202812
'''
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.1,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15, 
                   feature_fraction=0.3, 
                   bagging_fraction=0.1,
                   bagging_freq=0, 
                   reg_alpha=0,
                   reg_lambda=0.2,     
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'learning_rate': 0.1}----0.7271914592939538
#{'learning_rate': 0.2}-------0.5715672169255673
'''
regressor=LGBMRegressor(n_estimators=201,
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15, 
                   feature_fraction=0.3, 
                   bagging_fraction=0.1,
                   bagging_freq=0, 
                   reg_alpha=0,
                   reg_lambda=0.2,
                   min_split_gain=0,      
                   random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''











######################预测######################
regressor=LGBMRegressor(n_estimators=201,
                   learning_rate=0.2,                        
                   max_depth=3,
                   num_leaves=3,
                   min_data_in_leaf=11,
                   max_bin=15, 
                   feature_fraction=0.3, 
                   bagging_fraction=0.1,
                   bagging_freq=0, 
                   reg_alpha=0,
                   reg_lambda=0.2,
                   min_split_gain=0,      
                   random_state=90)
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)








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


'''
MAE_train: 0.042458469190009354
MSE_train: 0.007786974232370382
RMSE_train: 0.08824383396232499
MAPE_train: 861.1261216382585
r2_score_train: 0.7375234632157268
'''


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
MAE_test: 0.029667903235161868
MSE_test: 0.002963118262672807
RMSE_test: 0.05443453189541366
MAPE_test: 1646.0309735486735
r2_score_test: 0.5699006412190029
EV_test: 0.6337627004099421

'''
















'''
mae_scores = cross_val_score(regressor, x, y, cv=k, scoring='neg_mean_absolute_error').mean()
mse_scores = cross_val_score(regressor, x, y, cv=k, scoring='neg_mean_squared_error').mean()
rmse_scores = np.sqrt(-mse_scores).mean()
r2_scores = cross_val_score(regressor, x, y, cv=k, scoring='r2').mean()


mae_scores=-mae_scores
mse_scores=-mse_scores 


print('MAE:', mae_scores)
print('MSE:', mse_scores)
print('RMSE:', rmse_scores)
print('r2_score:', r2_scores)

'''
'''
MAE: 4.782470188632756
MSE: 36.54618406779342
RMSE: 6.0453439991280415
r2_score: 0.17188748314290672
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






#########################################
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

            特征名称  特征重要性
0             hv     83
1  pipe_diameter     49
2            r/d    155
3             dp     33
4             fp     80

'''


