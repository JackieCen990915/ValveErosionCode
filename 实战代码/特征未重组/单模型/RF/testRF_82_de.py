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

from sko.GA import GA #导入遗传算法的包

#导入粒子群算法
from sko.PSO import PSO


#导入差分进化算法
from sko.DE import DE




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

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)







# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)




def train_rf(x):
    #n_estimators,max_depth,max_features,min_samples_leaf,min_samples_split = x
    n_estimators,max_depth = x
    clf = RandomForestRegressor(
                            n_estimators=int(n_estimators),
                            max_depth=int(max_depth),
                            #max_features=int(max_features),
                            #min_samples_leaf=int(min_samples_leaf),
                            #min_samples_split=int(min_samples_split), 
                            random_state=90)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score






######################调参######################
'''
param_grid = {
        'n_estimators': np.arange(1,201,10),
        'max_depth': np.arange(1,21,1),
        'max_features': np.arange(3,21,1),
        'min_samples_leaf': np.arange(1,21,1), 
        'min_samples_split': np.arange(2,21,1)    
    
}

'''
'''
param_grid = {
         'min_samples_split': np.arange(2,21,1),       
}

'''


A_DE = DE(func=train_rf, n_dim=2, size_pop=50, max_iter=50,prob_mut=0.3,lb=[10,1], ub=[100,20])




best_x, best_y = A_DE.run()#运行算法
print('best_x:', best_x, '\n', 'best_y:', best_y)


n_estimators=int(best_x[0])
max_depth=int(best_x[0])


print('best_x[0]:', n_estimators)
print('best_x[1]:', max_depth)





######################5.使用数据进行预测######################

regressor=RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            #max_features=max_features,
                            #min_samples_leaf=min_samples_leaf,
                            #min_samples_split=min_samples_split, 
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


'''
MAE_train: 0.043600699738718864
MSE_train: 0.007852250740531886
RMSE_train: 0.08861292648666946
MAPE_train: 426.9575707459526
r2_score_train: 0.7353231796030782
EV_train: 0.7353240198850599

'''


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


'''
MAE_test: 0.016492585758684283
MSE_test: 0.0010885513365276125
RMSE_test: 0.03299320136827605
MAPE_test: 220.9804727108332
r2_score_test: 0.8419957658327114
EV_test: 0.8804739219732881
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
MAE: 0.0858966075177188
MSE: 0.02775199694928207
RMSE: 0.16658930622726678
r2_score: -56.43515923408199
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

'''
            特征名称     特征重要性
0             hv  0.001154
1  pipe_diameter  0.029400
2            r/d  0.018522
3             dp  0.000000
4             fp  0.945763
5        (hv)^k1  0.005161
6   (dp/dref)^k3  0.000000

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
