from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

import datetime

import shap
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

import pandas as pd
# 记录开始时间
#start_time = datetime.datetime.now()





######################1.导入数据######################
df = pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx', sheet_name='Sheet2')







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


print("---------------x_train------------------")
print(x_train)
print("---------------y_train------------------")
print(y_train)
print("---------------x_test------------------")
print(x_test)
print("type of x_test:",type(x_test))
print("---------------y_test------------------")
print(y_test)





######################使用数据进行预测######################
regressor=RandomForestRegressor(
                            n_estimators=73,
                            max_depth=10,
                            random_state=90)
                      
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)


print('y_test_pred:', y_test_pred)
print('-------------------')





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





######################使用 SHAP 模型解释######################

# 测试集？

# 计算 SHAP 值
explainer = shap.TreeExplainer(regressor)
#输出numpy.array数组
#shap_values = explainer.shap_values(x_test)
#shap_values = explainer.shap_values(x)
shap_values = explainer.shap_values(x_train)



# 获取特征重要性
feature_importances = np.abs(shap_values).mean(axis=0)
feature_names = x.columns
# 打印特征重要性
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance}")



# 1）可视化所有样本的SHAP值摘要 点图默认
#shap.summary_plot(shap_values, x_test, feature_names=feature_names)
#shap.summary_plot(shap_values, x, feature_names=feature_names)
shap.summary_plot(shap_values, x_train, feature_names=feature_names)




# 2）可视化所有样本的SHAP特征重要性 条形图
#shap.summary_plot(shap_values, x_test, feature_names=feature_names, plot_type='bar')
#shap.summary_plot(shap_values, x, feature_names=feature_names, plot_type='bar')
shap.summary_plot(shap_values, x_train, feature_names=feature_names, plot_type='bar')

'''
Hv: 0.005203914260846203
ρw: 0.0
Vo: 0.00062044350330581
dp: 0.0006440236998828418
u0: 0.0019510430723761517
mp: 8.12516665364508e-05
Hv': 0.0
dp': 0.000530462425711419
ρw': 0.0
ρw'_Hv': 0.0
ρw'_Hv'_dp': 0.0006248773930809742

'''



'''
# 可视化单样本的 SHAP 值瀑布图-------咋不行----终于可以了。除了x_test的数值是归一化的，别的都可以显示出来了，包括特征名

feature_names = x.columns
x_train_columns=pd.DataFrame(x_train, columns=['hv', 'D', 'R/D','dp','mp','u0',"hv'","dp'"])
print("x_train_colums:",x_train_columns)


#训练集
explainer2 = shap.KernelExplainer(regressor.predict,x_train_columns)
shap_values2 = explainer2(x_train_columns)
# 选择一个训练样本进行可视化
sample_index = 22 
# 获取特定样本的 SHAP 值和基准线值
specific_sample_shap_value = shap_values2[sample_index]
base_value2 = explainer2.expected_value
print("所有训练样本的基准线为",base_value2)

# 输出特定样本各个特征的 SHAP 值和基准线值
for feature_name, value in zip(feature_names, specific_sample_shap_value.values):
    print(f"测试样本 {sample_index} 的特征 '{feature_name}' 的 SHAP 值为：{value}")
#所有训练样本的基准线为 0.08998946337309963


shap.plots.waterfall(specific_sample_shap_value)
'''
'''
所有训练样本的基准线为 0.10605899669799411
测试样本 22 的特征 'hv' 的 SHAP 值为：0.04473814254608563
测试样本 22 的特征 'D' 的 SHAP 值为：0.010545083268827193
测试样本 22 的特征 'R/D' 的 SHAP 值为：0.017204915963918932
测试样本 22 的特征 'dp' 的 SHAP 值为：0.03832866856965919
测试样本 22 的特征 'mp' 的 SHAP 值为：0.04406744073337732
测试样本 22 的特征 'u0' 的 SHAP 值为：0.16907883750663139
测试样本 22 的特征 'hv'' 的 SHAP 值为：0.04533154521759064
测试样本 22 的特征 'dp'' 的 SHAP 值为：0.030034019972180248
'''













'''
# 3）可视化单样本的SHAP值力图
# 基础值（通常是模型的平均预测值）
base_value = explainer.expected_value
# 选择一个测试样本进行可视化
single_sample_index = 1 
single_shap_value = shap_values[single_sample_index]

print("shap_values:",shap_values)
print("single_shap_value:",single_shap_value)
print("x_test[single_sample_index]:",x_test[single_sample_index])

'''

'''
#matplotlib=True
#shap.force_plot(base_value, single_shap_value, x_test[single_sample_index], feature_names=feature_names, matplotlib=True)
#shap.force_plot(base_value, single_shap_value, x.iloc[single_sample_index], feature_names=feature_names, matplotlib=True)




#可视化单样本的SHAP值的决策图
#可视化单样本的SHAP值的局部条形图---和力图、瀑布图、决策图相通
#输出shap.Explanation对象
#shap_values2 = explainer(x_test) 
#shap.plots.bar(shap_values2[single_sample_index],show_data=True)

'''




# 可视化单样本的 SHAP 值瀑布图-------咋不行----终于可以了。除了x_test的数值是归一化的，别的都可以显示出来了，包括特征名
#explainer2 = shap.TreeExplainer(regressor,x_test)的x_test要传入！！！！！！
'''
explainer2 = shap.TreeExplainer(regressor,x)
shap_values2 = explainer2(x) 
shap.plots.waterfall(shap_values2[single_sample_index])
'''

'''
#x_test_columns=pd.DataFrame(x_test, columns=['hv', 'D', 'R/D','dp','fp','u0',"hv'","dp'"])
x_test_columns=pd.DataFrame(x_test, columns=['Hv','ρw','Vo','dp','u0','mp',"Hv'","dp'","ρw'","ρw'_Hv'","ρw'_Hv'_dp'"])
print("x_test_colums:",x_test_columns)

#x_test=x_test_columns.values
explainer2 = shap.TreeExplainer(regressor,x_test_columns)
shap_values2 = explainer2(x_test_columns) 
shap.plots.waterfall(shap_values2[single_sample_index])

'''

'''
# 可视化单样本的 SHAP 值决策图
#shap.decision_plot(base_value,single_shap_value, feature_names)


# 可视化所有样本的单输入特征和输出特征ER的依赖图
#print("x_test.columns:",x_test.columns)
#shap.dependence_plot('hv',shap_values,x_test,interaction_index=None,show=False)报错无'hv'列
#shap.dependence_plot('u0',shap_values,x,interaction_index=None,show=False)

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






