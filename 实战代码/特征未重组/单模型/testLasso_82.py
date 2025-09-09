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



from sklearn.linear_model import Lasso




######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/冲蚀/周报&开会讨论_研二下/实战所用数据/弯头气固/0229_冲蚀数据统计_Num_GA.xlsx',sheet_name='Sheet1')








######################2.提取特征变量######################
x=df.drop(columns='erosion_rate')
y=df['erosion_rate']


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









######################调参######################
# 设置要调优的超参数范围


array_0_1=np.linspace(0,1,1000)
array_1_100=np.arange(1,101,5)
array_0_100=np.concatenate((array_0_1, array_1_100))

param_grid = {
    'alpha': array_0_100,
    'normalize': [True, False]
}


'''
param_grid = {
    'alpha': [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,25,50,75,100],
    'normalize': [True, False]
}
'''









#调参#
scorel=[]

k=5

regressor=Lasso(random_state=90)
GS=GridSearchCV(regressor,param_grid,cv=k)
GS.fit(x_train,y_train)
print(GS.best_params_)
print(GS.best_score_)

'''
{'alpha': 0.01, 'normalize': False}
0.16724794420329187
'''















######################5.使用测试集数据进行预测######################
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

regressor=GS.best_estimator_



'''
regressor=Lasso(alpha=0.01,
                normalize=False,
                random_state=90)
'''
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
MAE_test: 0.06634451273922466
MSE_test: 0.008488540843746761
RMSE_test: 0.0921332776131771
MAPE_test: 713.8281687437375
r2_score_test: 0.28557684359149627
EV_test: 0.35039750683581816
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

'''
            特征名称     特征重要性
0              p  0.000000
1             hv  0.000000
2  pipe_diameter  0.000000
3            r/d  0.000000
4             dp  0.000000
5             fp  0.997727
6        (hv)^k1  0.002273
7   (dp/dref)^k3  0.000000

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
