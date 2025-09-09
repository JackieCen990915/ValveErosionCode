from catboost import CatBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook
from skopt import BayesSearchCV


######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand_oka.xlsx',sheet_name='Sheet1_er_kg')








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



# 使用MinMaxScaler进行归一化,对结果影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)





######################调参######################
# 设置要调优的超参数范围


'''
param_distributions = {
    'iterations': [100, 500, 1000],
    'depth': np.arange(1,11,1),
    'learning_rate': np.linspace(0.1,1,10),
    'l2_leaf_reg': [1, 4, 9]
}
'''

param_grid = {
    'max_depth': np.arange(1,21,1),
    'learning_rate': np.linspace(0,1,21),
    'l2_leaf_reg': np.linspace(0,1,11)
}


'''
param_grid = {
    'iterations': [100, 500, 1000],
    'depth': np.arange(1,11,1),
    'learning_rate': np.linspace(0.1,1,10),
    'l2_leaf_reg': [1, 4, 9],
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
}
'''









#调参#
scorel=[]


k=5


regressor= CatBoostRegressor(
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)



#{'max_depth': 17}--0.966307667094263
'''
regressor= CatBoostRegressor(
                           learning_rate=0.1,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'max_features': 5}---0.9758893777944195
'''
regressor=DecisionTreeRegressor(
                           max_depth=17,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''



#{'min_samples_leaf': 1}---0.9758893777944195
'''
regressor=DecisionTreeRegressor(
                           max_depth=17,
                           max_features=5,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''


#{'min_samples_split': 2}---0.9758893777944195
'''
regressor=DecisionTreeRegressor(
                           max_depth=17,
                           max_features=5,
                           min_samples_leaf=1,
                           random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)
'''














######################5.使用测试集数据进行预测######################

#网格搜索
'''
regressor = CatBoostRegressor(random_state=90)


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
#贝叶斯搜索
regressor = CatBoostRegressor(random_state=90)
bayes_search = BayesSearchCV(regressor, param_distributions, cv=k)

# 执行搜索
bayes_search.fit(x_train, y_train)

# 输出最优参数
best_params = bayes_search.best_params_

# 使用最优参数手动构建并训练最终模型
best_model = CatBoostRegressor(**best_params)
'''

'''
regressor=DecisionTreeRegressor(
                           max_depth=17,
                           max_features=5,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           random_state=90)
'''
'''
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

print("---------------x_train------------------")
print(x_train)
print("---------------y_train------------------")
print(y_train)
print("---------------x_test------------------")
print(x_test)
print("---------------y_test------------------")
print(y_test)
print("---------------y_pred------------------")
print(y_pred)

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
excel_file_path = 'DT_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')
'''




'''
######################6.评估模型######################

MAE=metrics.mean_absolute_error(y_test, y_pred)
MSE=metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(MSE)
MAPE=metrics.mean_absolute_percentage_error(y_test, y_pred)
R2=metrics.r2_score(y_test, y_pred)


print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)
print('r2_score:', R2)
'''


'''
MAE: 0.6015702997052934
MSE: 1.429788381244348
RMSE: 1.1957375887895922
MAPE: 0.02747362716438873
r2_score: 0.9809820800268197
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
MAE: 5.1713563990873554
MSE: 43.005781717871535
RMSE: 6.557879361338659
r2_score: 0.055697465098718243

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
'''

'''
         特征名称     特征重要性
0         WOH  0.057605
1         WOB  0.033967
2         SPP  0.080028
3         RPM  0.043480
4      Torque  0.090354
5   Flow Rate  0.059865
6     Bit Run  0.049148
7    Bit Time  0.042660
8          GR  0.051627
9          DT  0.014824
10       RMSL  0.016055
11         RD  0.012463
12      Depth  0.130211
13      Angle  0.142384
14       Azim  0.174322
15   Bit Size  0.001006


1	Azim	0.174322
2	Angle	0.142384
3	Depth	0.130211
4	Torque	0.090354
5	SPP	0.080028
6   Flow Rate	0.059865
7	WOH	0.057605
8	GR	0.051627
9	Bit Run	0.049148
10	RPM	0.04348
11  Bit Time	0.04266
12	WOB	0.033967
13	RMSL	0.016055
14	DT	0.014824
15	RD	0.012463
16  Bit Size	0.001006



'''



'''
#获取特征名称
feature_names=x.columns

# 获取特征重要性
feature_importance=regressor.feature_importances_

# 打印特征重要性
for i, importance in enumerate(feature_importance):
    print(f"Feature {i+1}: {importance}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance, tick_label=feature_names)
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.show()

'''



'''

######################8.画图######################
#查看测试集上真实值和预测值的拟合程度（散点图）

plt.scatter(y_test,y_pred)
plt.xlabel('Actual value')
plt.ylabel('predicition value')
plt.title('Actual value VS predicition value')
plt.show()

#不太对
#可视化测试集上真实值与预测值的分布
plt.plot(y_test,color="red")
plt.plot(y_pred,color="blue")
plt.legend(['Actual value','predicition value'])
plt.show()
'''


