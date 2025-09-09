import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import shap



######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx',sheet_name='Sheet2')








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

print("---------------x_train------------------")
print(x_train)
print(type(x_train))
print("---------------x_test------------------")
print(x_test)
print(type(x_test))
print("---------------y_train------------------")
print(y_train)
print(type(y_train))
print("---------------y_test------------------")
print(y_test)
print(type(y_test))



# 使用MinMaxScaler进行归一化
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



######################使用数据进行预测######################
regressor=DecisionTreeRegressor(
                            max_depth=13,
                            max_features=6,
                            min_samples_leaf=2,
                            min_samples_split=5,
                            random_state=100)


regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)
print("---------------y_train_pred------------------")
print(y_train_pred)
print(type(y_train_pred))




######################评估模型(训练集)######################
MSE_train=metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train=np.sqrt(MSE_train)
R2_train=metrics.r2_score(y_train, y_train_pred)
EV_train=metrics.explained_variance_score(y_train, y_train_pred)

print('RMSE_train:', RMSE_train)
print('r2_score_train:', R2_train)
print('EV_train:', EV_train)
print('-------------------')



######################评估模型(测试集)######################
MSE_test=metrics.mean_squared_error(y_test, y_test_pred)
RMSE_test=np.sqrt(MSE_test)
R2_test=metrics.r2_score(y_test, y_test_pred)
EV_test=metrics.explained_variance_score(y_test, y_test_pred)

print('RMSE_test:', RMSE_test)
print('r2_score_test:', R2_test)
print('EV_test:', EV_test)
print('-------------------')






######################使用 SHAP 模型解释######################
# 计算 SHAP 值
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(x_train)


# 获取特征重要性
feature_importances = np.abs(shap_values).mean(axis=0)
feature_names = x.columns
# 打印特征重要性
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance}")


# 1）可视化所有样本的SHAP值摘要 点图默认
shap.summary_plot(shap_values, x_train, feature_names=feature_names)


# 2）可视化所有样本的SHAP特征重要性 条形图
shap.summary_plot(shap_values, x_train, feature_names=feature_names, plot_type='bar')


# 可视化单样本的 SHAP 值瀑布图
feature_names = x.columns
x_train_columns=pd.DataFrame(x_train, columns=['Hv','ρw','Vo','dp','u0','mp',"Hv'","dp'","ρw'","ρw'_Hv'","ρw'_Hv'_dp'"])
print("x_train_colums:",x_train_columns)

explainer2 = shap.KernelExplainer(regressor.predict,x_train_columns)
shap_values2 = explainer2(x_train_columns)
# 选择一个训练样本进行可视化
sample_index = 3
# 获取特定样本的 SHAP 值和基准线值
specific_sample_shap_value = shap_values2[sample_index]
base_value2 = explainer2.expected_value
print("所有训练样本的基准线为",base_value2)

# 输出特定样本各个特征的 SHAP 值和基准线值
for feature_name, value in zip(feature_names, specific_sample_shap_value.values):
    print(f"样本 {sample_index} 的特征 '{feature_name}' 的 SHAP 值为：{value}")


shap.plots.waterfall(specific_sample_shap_value)


