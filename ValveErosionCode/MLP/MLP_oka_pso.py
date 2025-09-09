import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 
#导入粒子群算法
from sko.PSO import PSO
import datetime

# 记录开始时间
start_time = datetime.datetime.now()



# 定义激活函数的候选列表
activation_functions = ['tanh','relu']


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




# 使用MinMaxScaler进行归一化
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



'''
param_grid={
        'alpha':[0.01,0.05,0.1],
        'hidden_layer_sizes':np.arange(4,14,1),
        'max_iter':np.arange(1,501,50),
        
}
'''

def train_MLP(x):
    alpha, hidden_layer_sizes, activation_index = x
    activation = activation_functions[int(activation_index)]
    clf = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=int(hidden_layer_sizes),
                        activation=activation,
                        solver='lbfgs',
                        random_state=90)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test, y_pred)
    return -score

'''


def train_MLP(x):
    #alpha,hidden_layer_sizes,max_iter = x
    alpha,hidden_layer_sizes= x
    clf = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=int(hidden_layer_sizes),
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''




'''
def train_MLP(x):
    alpha,hidden_layer_sizes,max_iter = x
    clf = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=int(hidden_layer_sizes),
                        max_iter=int(max_iter),
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score
'''



pso=PSO(func=train_MLP,dim=3,pop=200,max_iter=400,lb=[0, 5, 0], ub=[1, 14, len(activation_functions) - 1],w=1,c1=2,c2=2)
#pso=PSO(func=train_MLP,dim=2,pop=200,max_iter=400,lb=[0,4], ub=[1,13],w=1,c1=2,c2=2)
#pso=PSO(func=train_MLP,dim=2,pop=200,max_iter=400,lb=[0,4], ub=[1,13],w=1,c1=2,c2=2)


# 运行优化算法
pso.run()
# 获取优化结果
print('best_x:', pso.gbest_x, '\n', 'best_y:', pso.gbest_y)

alpha=pso.gbest_x[0]
hidden_layer_sizes=int(pso.gbest_x[1])
activation_index = int(pso.gbest_x[2])
activation = activation_functions[activation_index]

print('best_x[0]:', alpha)
print('best_x[1]:', hidden_layer_sizes)
print('best_x[2]_activation:', activation)

'''
alpha=9.5777
hidden_layer_sizes=13
'''

regressor = MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver='lbfgs',
                        random_state=90)

'''
regressor=MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
'''
                    
'''
regressor=MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
'''
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)


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



# 记录结束时间
end_time = datetime.datetime.now()
 
# 计算运行时长
duration = end_time - start_time
 
# 打印运行时长
print("Duration is ",duration)
