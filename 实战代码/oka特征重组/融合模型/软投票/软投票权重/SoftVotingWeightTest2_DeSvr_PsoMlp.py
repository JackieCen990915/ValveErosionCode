from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPRegressor


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

# 使用MinMaxScaler进行归一化,对结果没影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# 定义基础模型
'''
models = [
       SVR(kernel='rbf',
         C=14.063456167204542,
         epsilon=3.363135491649024e-05,
         gamma=10.421706507320557),
    
        MLPRegressor(
                    alpha=9.577699892831528e-05,
                    hidden_layer_sizes=13,
                    activation='relu',
                    solver='lbfgs',
                    random_state=90)
]
'''

models = [
       SVR(kernel='rbf',
         C=14.063456167204542,
         epsilon=3.363135491649024e-05,
         gamma=10.421706507320557),
    
        MLPRegressor(
                    alpha=9.577699892831528e-05,
                    hidden_layer_sizes=13,
                    activation='relu',
                    solver='lbfgs',
                    random_state=90),
       
       RandomForestRegressor(
                            n_estimators=20,
                            max_depth=6,
                            max_features=4,
                            min_samples_leaf=1,
                            min_samples_split=2,
                            random_state=90)
]



# 使用交叉验证得到基础模型的预测结果
predictions = []
for model in models:
    preds = cross_val_predict(model, x_train, y_train, cv=5)
    predictions.append(preds)
print("predictions:", predictions)



# 定义均方误差损失函数
def mse_loss(weights):
    ensemble_preds = np.dot(weights, predictions)
    return mean_squared_error(y_train, ensemble_preds)
print("mse_loss:", mse_loss)




# 初始权重
initial_weights = np.ones(len(models)) / len(models)

# 最小化均方误差损失函数，得到优化后的权重
result = minimize(mse_loss, initial_weights, method='L-BFGS-B')

#svr、mlp优化后的权重: [0.78657457 0.21440629]
#svr、mlp、rf优化后的权重:[0.62223953 0.12769325 0.27890609]
optimized_weights = result.x
print("优化后的权重:", optimized_weights)
