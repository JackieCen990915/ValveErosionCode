import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 



from sko.GA import GA #导入遗传算法的包





######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/弯头冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand_oka.xlsx',sheet_name='Sheet1_er_kg')





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


##### 将10%(6个)异常值添加到训练集#####################################
random_selected_train_index=[4, 23, 7, 64, 33, 73]
random_multiple=[0.24511851,0.55879534,0.61794635,1.87577546,1.48501098,1.60618524]
y_train[random_selected_train_index]= y_train[random_selected_train_index]*random_multiple



# 使用MinMaxScaler进行归一化,对结果有影响------------------
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


#按照这个格式，改成和网格搜索一样的参数？
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

#ga = GA(func=train_MLP, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.01,lb=[0,4,1], ub=[1,13,500], precision=1e-5)
'''
ga = GA(func=train_MLP, n_dim=2, size_pop=200, max_iter=400, prob_mut=0.01,lb=[0,4], ub=[1,13], precision=1e-5)


best_x, best_y = ga.run()





#拿到最佳预测值，方便之后做拟合

print('best_x:', best_x, '\n','best_y:',-best_y[0])

alpha=best_x[0]
hidden_layer_sizes=int(best_x[1])
#max_iter=best_x[2]

print('best_x[0]:', alpha)
print('best_x[1]:', hidden_layer_sizes)
#print('best_x[2]_epsilon:', max_iter)

'''



regressor=MLPRegressor(
                        alpha=0.0008,
                        hidden_layer_sizes=10,
                        activation='relu',
                        solver='lbfgs',
                        random_state=90)
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
MAE_train: 0.026500028263470577
MSE_train: 0.001965783372801057
RMSE_train: 0.04433715566881864
MAPE_train: 3858.39179238051
r2_score_train: 0.939202482467039


MAE_test: 0.025555662297039976
MSE_test: 0.0013826069318184082
RMSE_test: 0.03718342280934352
MAPE_test: 3622.4168687356228
r2_score_test: 0.9165220180594134
EV_test: 0.9167791128114773

'''


'''
best_x: [8.08721990e-04 1.01209699e+01]

 best_y: 0.9577401996727505
best_x[0]: 0.0008087219903716306
best_x[1]: 10



MAE_train: 0.02015946800997666
MSE_train: 0.0012177227148489366
RMSE_train: 0.034895883924166995
MAPE_train: 1149.1932219353541
r2_score_train: 0.96325389789967


MAE_test: 0.01852661135773115
MSE_test: 0.0006999293887015883
RMSE_test: 0.02645617864888254
MAPE_test: 997.6530001197189
r2_score_test: 0.9577401996727505
EV_test: 0.9665519547501301

'''



'''
ga跑出来，给de用吧----de太慢了
best_x: [2.41090707e-03 5.59626350e+00] 
 best_y: 0.9547402647804266
best_x[0]: 0.0024109070656361817
best_x[1]: 5

MAE_train: 0.013366820534293706
MSE_train: 0.0007489706787205963
RMSE_train: 0.027367328673449227
MAPE_train: 888.3778589630825
r2_score_train: 0.9773989983969095
MAE_test: 0.014603444525327285
MSE_test: 0.0007496159129887115
RMSE_test: 0.027379114539895395
MAPE_test: 212.68772254038964
r2_score_test: 0.9547402647804266
EV_test: 0.9623725425431697
'''
