
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#网格搜索
from sklearn.model_selection import GridSearchCV
import numpy as np


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR

from openpyxl import Workbook

#导入差分进化算法
from sko.DE import DE

from openpyxl import Workbook

import shap


######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/弯头冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand_oka_.xlsx',sheet_name='Sheet1_er_kg')





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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)


print("---------------x_train------------------")
print(x_train)
print(type(x_train))
print("---------------y_train------------------")
print(y_train)
print(type(y_train))


'''
---------------x_train------------------
        hv      D    R/D   dp        mp      u0       hv'       dp'
26  1.4097   52.5  2.875  350  0.189500  141.00  0.959632  1.013588
65  1.8795   40.0  1.500   80  0.200000   20.00  0.927075  0.765730
67  1.8795   40.0  1.500  150  0.200000   20.00  0.927075  0.862871
1   1.1937   52.5  1.500  350  0.067580   47.00  0.978977  1.013588
74  1.8795   40.0  1.500  200  0.300000   20.00  0.927075  0.911348
62  1.8795   40.0  1.500  200  0.200000   40.00  0.927075  0.911348
58  1.8795   40.0  8.000  200  0.200000   20.00  0.927075  0.911348
31  1.1937   52.5  1.500  350  0.106000  100.00  0.978977  1.013588
14  1.1937   52.5  1.500  350  0.302100  222.00  0.978977  1.013588
53  1.8795   40.0  2.500  200  0.200000   20.00  0.927075  0.911348
16  1.1937   52.5  1.500  350  0.092490  109.00  0.978977  1.013588
19  1.1937   52.5  1.500  350  0.171190  108.00  0.978977  1.013588
54  1.8795   40.0  3.000  200  0.200000   20.00  0.927075  0.911348
6   1.1937   52.5  1.500  350  0.140980  103.00  0.978977  1.013588
60  1.8795   40.0  1.500  200  0.200000   15.00  0.927075  0.911348
44  1.6257   50.8  1.500  150  0.000521   18.90  0.943355  0.862871
29  1.4097   52.5  2.875  350  0.320700  107.00  0.959632  1.013588
59  1.8795   40.0  1.500  200  0.200000   10.00  0.927075  0.911348
15  1.1937   52.5  1.500  350  0.050090  108.00  0.978977  1.013588
18  1.1937   52.5  1.500  350  0.153170  104.00  0.978977  1.013588
13  1.1937   52.5  1.500  350  0.381600  205.00  0.978977  1.013588
22  1.1937   52.5  1.500  350  0.384250  111.00  0.978977  1.013588
10  1.1937   52.5  1.500  350  0.291500  177.00  0.978977  1.013588
69  1.8795   40.0  1.500  300  0.200000   20.00  0.927075  0.984332
11  1.1937   52.5  1.500  350  0.288900  178.00  0.978977  1.013588
0   1.1937   41.0  3.250  100  0.028600   25.24  0.978977  0.798893
5   1.1937   52.5  1.500  350  0.139390   98.00  0.978977  1.013588
3   1.1937   52.5  1.500  350  0.130600   93.00  0.978977  1.013588
40  1.6257  101.6  5.000  150  0.000148   23.00  0.943355  0.862871
12  1.1937   52.5  1.500  350  0.296800  203.00  0.978977  1.013588
50  1.8795  400.0  1.500  200  0.200000   20.00  0.927075  0.911348
34  1.1937   52.5  1.500  350  0.636000  100.00  0.978977  1.013588
27  1.4097   52.5  2.875  350  0.408100  107.00  0.959632  1.013588
30  1.4097   52.5  4.500  350  0.402800  111.00  0.959632  1.013588
21  1.1937   52.5  1.500  350  0.296800  107.00  0.978977  1.013588
33  1.1937   52.5  1.500  350  0.305000  100.00  0.978977  1.013588
38  1.1937  254.0  3.500  350  2.650000  100.00  0.978977  1.013588
52  1.8795   40.0  2.000  200  0.200000   20.00  0.927075  0.911348
28  1.4097   52.5  2.875  350  0.173600  141.00  0.959632  1.013588
35  1.1937   52.5  1.500  350  0.768000  100.00  0.978977  1.013588
41  1.6257  101.6  5.000  150  0.000188   35.00  0.943355  0.862871
7   1.1937   52.5  1.500  350  0.204580  167.00  0.978977  1.013588
48  1.8795  200.0  1.500  200  0.200000   20.00  0.927075  0.911348
56  1.8795   40.0  5.000  200  0.200000   20.00  0.927075  0.911348
71  1.8795   40.0  1.500  200  0.100000   20.00  0.927075  0.911348
63  1.8795   40.0  1.500   50  0.200000   20.00  0.927075  0.700314
64  1.8795   40.0  1.500   60  0.200000   20.00  0.927075  0.724999
39  1.1937  304.0  3.500  350  2.650000  100.00  0.978977  1.013588
20  1.1937   52.5  1.500  350  0.207760  108.00  0.978977  1.013588
2   1.1937   52.5  1.500  350  0.118200   72.00  0.978977  1.013588
55  1.8795   40.0  4.000  200  0.200000   20.00  0.927075  0.911348
49  1.8795  300.0  1.500  200  0.200000   20.00  0.927075  0.911348
36  1.1937  152.0  3.500  350  2.650000  100.00  0.978977  1.013588
25  1.1937   52.5  1.500  350  0.747300  103.00  0.978977  1.013588
46  1.8795   40.0  1.500  200  0.200000   20.00  0.927075  0.911348
32  1.1937   52.5  1.500  350  0.186000  100.00  0.978977  1.013588
47  1.8795  100.0  1.500  200  0.200000   20.00  0.927075  0.911348
57  1.8795   40.0  6.000  200  0.200000   20.00  0.927075  0.911348
17  1.1937   52.5  1.500  350  0.096195  108.00  0.978977  1.013588
66  1.8795   40.0  1.500  100  0.200000   20.00  0.927075  0.798893
<class 'pandas.core.frame.DataFrame'>
---------------y_train------------------
26    2.589600e-01
65    1.959984e-03
67    2.109978e-03
1     2.589600e-03
74    3.250026e-03
62    9.909978e-03
58    1.319994e-03
31    3.120000e-02
14    5.467800e-01
53    2.040012e-03
16    4.399200e-02
19    1.076400e-01
54    1.969968e-03
6     4.126200e-02
60    1.250028e-03
44    1.419600e-06
29    3.393000e-01
59    8.440380e-04
15    2.776800e-02
18    7.706400e-02
13    6.208800e-01
22    2.043600e-01
10    5.756400e-01
69    2.209974e-03
11    5.085600e-01
0     2.230800e-04
5     3.853200e-02
3     2.886000e-02
40    1.341600e-07
12    6.052800e-01
50    3.143400e-05
34    2.223000e-01
27    3.853200e-01
30    1.653600e-01
21    1.115400e-01
33    1.014000e-01
38    2.808000e-02
52    2.080026e-03
28    2.355600e-01
35    2.184000e-01
41    7.051200e-07
7     2.909400e-01
48    1.250340e-04
56    1.890018e-03
71    1.050036e-03
63    1.650012e-03
64    1.780038e-03
39    1.872000e-02
20    1.068600e-01
2     1.287000e-02
55    1.940016e-03
49    5.678400e-05
36    7.410000e-02
25    2.308800e-01
46    2.159976e-03
32    6.240000e-02
47    4.269720e-04
57    1.719978e-03
17    4.126200e-02
66    2.030028e-03
'''


# 使用MinMaxScaler进行归一化,对结果有影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)



# 首先需要把 SVM 的训练封装为函数
# 按照这个格式，改成和网格搜索一样的参数？
def train_svr(x):
    c,gamma,epsilon = x
    clf = SVR(kernel='rbf',C=c,gamma=gamma,epsilon=epsilon)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = metrics.r2_score(y_test,y_pred)
    return -score





'''
pso=PSO(func=train_svr,dim=3,pop=400,max_iter=200,lb=[0.01,0.01,0.01],ub=[100,1,1],w=1,c1=2,c2=2)
ga = GA(func=train_svr, n_dim=3, size_pop=100, max_iter=800, prob_mut=0.001, lb=[0.01,0.01,0.01], ub=[100,1,1], precision=1e-5)
'''

#种群规模、最大迭代次数保持一致吧----虽然意思可能有点不一样--大点吧，慢就慢点--100、800、0.01
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.3,lb=[0.01,0.01,0.01],ub=[100,1,1])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.3,lb=[0,0,0], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.4,lb=[0,0,0], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.5,lb=[0,0,0], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.6,lb=[0,0,0], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.7,lb=[0,0,0], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.8,lb=[0,0,0], ub=[100,1000,100])


'''
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.001,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.005,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.01,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.05,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.1,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.2,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.3,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.4,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.5,lb=[0.01,0.01,0.01], ub=[100,1000,100])
#A_DE = DE(func=train_svr, n_dim=3, size_pop=200, max_iter=400,prob_mut=0.6,lb=[0.01,0.01,0.01], ub=[100,1000,100])
'''
'''
best_x, best_y = A_DE.run()#运行算法
print('best_x:', best_x, '\n', 'best_y:', best_y)


c=best_x[0]
gamma=best_x[1]
epsilon=best_x[2]

print('best_x[0]_C:', c)
print('best_x[1]_gamma:', gamma)
print('best_x[2]_epsilon:', epsilon)

'''


c=14.063456167204542
gamma=10.421706507320557
epsilon=3.363135491649024e-05

#regressor=SVR(kernel='rbf',C=0.8, epsilon=0.01, gamma=1)
regressor=SVR(kernel='rbf',C=c,gamma=gamma,epsilon=epsilon)
regressor.fit(x_train,y_train)
y_train_pred=regressor.predict(x_train)
y_test_pred=regressor.predict(x_test)



print(y_train_pred[22])
#0.5053876504762647




######################6.评估模型(训练集)######################
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
print('----------------------------')
'''
MAE_train: 0.004017495732222952
MSE_train: 0.00014552316550422864
RMSE_train: 0.012063298284641255
MAPE_train: 21.671754244718965
r2_score_train: 0.9949193183334681
EV_train: 0.9949390876950508
----------------------------
'''



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
MAE_test: 0.013393932056593092
MSE_test: 0.0006603022451379345
RMSE_test: 0.025696346922041945
MAPE_test: 330.8154337487086
r2_score_test: 0.9811607503571157
EV_test: 0.9813021678269813
'''



#########使用 SHAP 模型输出特征重要性##########
#测试集
'''
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

'''
hv: 0.022577643666605182
D: 0.005649217382919037
R/D: 0.003533875133959972
dp: 0.019612679703474748
fp: 0.023703740922410083
u0: 0.03724001550561763
(hv)^k1: 0.02236234100363176
(dp/dref)^k3: 0.014812750293407003


'''
#训练集
# 计算 SHAP 值

explainer = shap.KernelExplainer(regressor.predict, x_train)
shap_values = explainer.shap_values(x_train)

# 获取特征重要性
feature_importances = np.abs(shap_values).mean(axis=0)
feature_names = x.columns

# 打印特征重要性
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance}")

shap.summary_plot(shap_values, x_train, feature_names=feature_names)
shap.summary_plot(shap_values, x_train, feature_names=feature_names, plot_type='bar')









# 可视化单样本的 SHAP 值瀑布图-------咋不行----终于可以了。除了x_test的数值是归一化的，别的都可以显示出来了，包括特征名
'''
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


