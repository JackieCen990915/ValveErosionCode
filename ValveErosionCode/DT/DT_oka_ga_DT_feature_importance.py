import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt


######################1.导入数据######################
df=pd.read_excel('D:/A研项目/A研2项目/课题论文/2机理数据融合模型/阀门冲蚀/实战/0207_最开始70个数据跑阀芯/实战所用数据/1125_冲蚀数据整合_Sand_oka.xlsx',sheet_name='Sheet2')








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


######################基于DT算法的特征重要性得分和可视化######################
# 获取特征重要性得分
feature_importances = regressor.feature_importances_
feature_names = x.columns

# 创建一个DataFrame来存储特征名称和重要性得分
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# 可视化特征重要性得分
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()
plt.show()
