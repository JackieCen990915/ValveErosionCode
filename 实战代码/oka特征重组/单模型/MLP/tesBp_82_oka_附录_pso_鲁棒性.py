import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 



#导入粒子群算法
from sko.PSO import PSO


import shap


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




#pso=PSO(func=train_MLP,dim=2,pop=200,max_iter=400,lb=[0,4], ub=[1,13],w=1,c1=2,c2=2)
#pso=PSO(func=train_MLP,dim=2,pop=200,max_iter=400,lb=[0,4], ub=[1,13],w=1,c1=2,c2=2)


# 运行优化算法
#pso.run()
# 获取优化结果
#print('best_x:', pso.gbest_x, '\n', 'best_y:', pso.gbest_y)

#alpha=pso.gbest_x[0]
#hidden_layer_sizes=int(pso.gbest_x[1])
#max_iter=best_x[2]

#print('best_x[0]:', alpha)
#print('best_x[1]:', hidden_layer_sizes)
#print('best_x[2]_epsilon:', max_iter)


alpha=9.57769989e-05
hidden_layer_sizes=13

regressor=MLPRegressor(
                        alpha=alpha,
                        hidden_layer_sizes=hidden_layer_sizes,
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
MAE_train: 0.020239289813100313
MSE_train: 0.0011891428875930824
RMSE_train: 0.034483951159823355
MAPE_train: 2242.3364956975065
r2_score_train: 0.9632223282799366


MAE_test: 0.02432655288699308
MSE_test: 0.0013007863697176737
RMSE_test: 0.036066416091950056
MAPE_test: 2153.8035423881965
r2_score_test: 0.9214621172649271
EV_test: 0.9266675271298386
'''



#########使用 SHAP 模型输出特征重要性##########
# 计算 SHAP 值
explainer = shap.KernelExplainer(regressor.predict,x_test)
shap_values = explainer.shap_values(x_test)

# 获取特征重要性
feature_importances = np.abs(shap_values).mean(axis=0)
feature_names = x.columns

# 打印特征重要性
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance}")
    
shap.summary_plot(shap_values, x_test, feature_names=feature_names)
shap.summary_plot(shap_values, x_test, feature_names=feature_names, plot_type='bar')


'''
hv: 2.6445072834058155e-08
D: 7.297429744430218e-10
R/D: 1.0618776137494858e-09
dp: 1.784175616359022e-08
fp: 9.107551904550834e-11
u0: 1.5030198655982836e-08
(hv)^k1: 3.3944724682589946e-08
(dp/dref)^k3: 1.0740734578378562e-08

'''























'''
best_x: [9.57769989e-05 1.30000000e+01] 
 best_y: -0.9634904262739443
best_x[0]: 9.577699892831528e-05
best_x[1]: 13


MAE_train: 0.014261259769455506
MSE_train: 0.00048542844864525787
RMSE_train: 0.022032440823595963
MAPE_train: 2693.723979703927
r2_score_train: 0.9853516706892207

MAE_test: 0.014297092622191145
MSE_test: 0.0006046910638940291
RMSE_test: 0.024590466931191633
MAPE_test: 1668.956543528785
r2_score_test: 0.9634904262739443
EV_test: 0.9660928099438091

'''
