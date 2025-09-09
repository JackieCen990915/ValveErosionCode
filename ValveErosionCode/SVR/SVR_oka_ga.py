
import pandas as pd
from sklearn.model_selection import train_test_split
#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
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






# 使用MinMaxScaler进行归一化
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



# 首先需要把 SVM 的训练封装为函数
#按照这个格式，改成和网格搜索一样的参数
def train_svr(x):
    c,gamma,epsilon = x
    clf = SVR(kernel='rbf',C=c,gamma=gamma,epsilon=epsilon)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score




# 运行遗传算法寻得最优的 C 参数与 gama 参数
#prob_mut变异率0.001-0.3，交叉率0.8,precision=1e-5

#ga = GA(func=train_svr, n_dim=3, size_pop=50, max_iter=50, prob_mut=0.01,lb=[0.1,0.1,0.1], ub=[100,1000,100], precision=1e-5)
ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.01,lb=[0,0,0], ub=[100,1000,100], precision=1e-5)





#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.3, lb=[0.01,0.01,0.01], ub=[100,1,1], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.05,lb=[0.01,0.01,0.01], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.001,lb=[0,0,0], ub=[1000,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.005,lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.01,lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.05,lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.1,lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.2,lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.3,lb=[0,0,0], ub=[100,1000,100], precision=1e-5)


#prob_mut变异率0.001-0.3，交叉率0.8,precision=1e-7
'''
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.001,lb=[0,0,0], ub=[100,1000,100], precision=1e-7)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.005,lb=[0,0,0], ub=[100,1000,100], precision=1e-7)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.01,lb=[0,0,0], ub=[100,1000,100], precision=1e-7)
'''

#prob_mut变异率0.001-0.3，交叉率0.9,precision=1e-5
'''
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.001, lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.005, lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.01, lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.05, lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.1, lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.3, lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
#ga = GA(func=train_svr, n_dim=3, size_pop=200, max_iter=400, prob_mut=0.2, lb=[0,0,0], ub=[100,1000,100], precision=1e-5)
ga.cr=0.9
'''

best_x, best_y = ga.run()





#拿到最佳预测值，方便之后做拟合
print('best_x:', best_x, '\n','best_y:',-best_y[0])

c=best_x[0]
gamma=best_x[1]
epsilon=best_x[2]

print('best_x[0]_C:', c)
print('best_x[1]_gamma:', gamma)
print('best_x[2]_epsilon:', epsilon)


'''
best_x: [9.80940579e+01 8.78239430e+00 2.81929987e-03] 
 best_y: 0.991424415016994
best_x[0]_C: 98.09405792320119
best_x[1]_gamma: 8.782394295799689
best_x[2]_epsilon: 0.0028192998659193435

'''


#regressor=SVR(kernel='rbf',C=0.8, epsilon=0.01, gamma=1)
regressor=SVR(kernel='rbf',C=c,gamma=gamma,epsilon=epsilon)
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
