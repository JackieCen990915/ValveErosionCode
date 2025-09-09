
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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=90)

print("---------------x_test------------------")
print(x_test)
print(type(x_test))
print("---------------y_test------------------")
print(y_test)
print(type(y_test))

'''
---------------y_test------------------
35    2.184000e-01
41    7.051200e-07
59    8.440380e-04
21    1.115400e-01
45    5.506800e-06
5     3.853200e-02
3     2.886000e-02
42    5.304000e-06
20    1.068600e-01
52    2.080026e-03
8     3.720600e-01
27    3.853200e-01
46    2.159976e-03
50    3.143400e-05
17    4.126200e-02
Name: er, dtype: float64
<class 'pandas.core.series.Series'>
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
y_test_pred=regressor.predict(x_test)



print("---------------y_test_pred------------------")
print(y_test_pred)
print(type(y_test_pred))

'''
[
0.21405195--0
0.00143576--1
0.00098543--2
0.1363231--3
0.00191829--4
0.04114563--5
0.0293114--6
0.01545097--7
0.09404084--8
0.00120071--9
0.39363942--10
0.37407862--11
0.00176222--12
0.0006715--13
0.0438261--14
]
<class 'numpy.ndarray'>
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





#########使用 SHAP 模型输出特征重要性##########
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

#可视化所有样本的SHAP值摘要 点图默认
#shap.summary_plot(shap_values, x_test, feature_names=feature_names)
#可视化所有样本的SHAP特征重要性 条形图
#shap.summary_plot(shap_values, x_test, feature_names=feature_names, plot_type='bar')
'''


'''
hv: 0.01528911584303858
D: 0.0017965339671321414
R/D: 0.004469292032194472
dp: 0.013499887775665523
fp: 0.024570183801681935
u0: 0.024418953826853235
hv': 0.015657623691383193
dp': 0.01085278849213794
'''





# 可视化单样本的 SHAP 值瀑布图-------咋不行----终于可以了。除了x_test的数值是归一化的，别的都可以显示出来了，包括特征名

'''
x_test_columns=pd.DataFrame(x_test, columns=['hv', 'D', 'R/D','dp','fp','u0',"hv'","dp'"])
print("x_test_colums:",x_test_columns)
'''

'''
x_test_colums:
        hv         D       R/D  ...        u0       hv'       dp'
0   0.000000  0.027174  0.000000  ...  0.418494  1.000000  1.000000
1   0.629921  0.133913  0.538462  ...  0.108675  0.313665  0.518897
2   1.000000  0.000000  0.000000  ... -0.010486  0.000000  0.673640
3   0.000000  0.027174  0.000000  ...  0.451859  1.000000  1.000000
4   0.629921  0.023478  0.000000  ...  0.075310  0.313665  0.518897
5   0.000000  0.027174  0.000000  ...  0.408961  1.000000  1.000000
6   0.000000  0.027174  0.000000  ...  0.385129  1.000000  1.000000
7   0.629921  0.133913  0.538462  ...  0.242135  0.313665  0.518897
8   0.000000  0.027174  0.000000  ...  0.456625  1.000000  1.000000
9   1.000000  0.000000  0.076923  ...  0.037178  0.000000  0.673640
10  0.000000  0.027174  0.000000  ...  0.747378  1.000000  1.000000
11  0.314961  0.027174  0.211538  ...  0.451859  0.627278  1.000000
12  1.000000  0.000000  0.000000  ...  0.037178  0.000000  0.673640
13  1.000000  0.782609  0.000000  ...  0.037178  0.000000  0.673640
14  0.000000  0.027174  0.000000  ...  0.456625  1.000000  1.000000

[15 rows x 8 columns]
'''
'''
explainer2 = shap.KernelExplainer(regressor.predict,x_test_columns)
shap_values2 = explainer2(x_test_columns)
# 选择一个测试样本进行可视化
sample_index = 2 
# 获取特定样本的 SHAP 值和基准线值
specific_sample_shap_value = shap_values2[sample_index]
base_value2 = explainer2.expected_value
print("所有测试样本的基准线为",base_value2)

# 输出特定样本各个特征的 SHAP 值和基准线值
for feature_name, value in zip(feature_names, specific_sample_shap_value.values):
    print(f"测试样本 {sample_index} 的特征 '{feature_name}' 的 SHAP 值为：{value}")
#所有测试样本的基准线为 0.08998946337309963


shap.plots.waterfall(specific_sample_shap_value)

'''

'''
高[0]
测试样本 0 的特征 'hv' 的 SHAP 值为：0.0147469302008119
测试样本 0 的特征 'D' 的 SHAP 值为：0.0023518541203330426
测试样本 0 的特征 'R/D' 的 SHAP 值为：0.002251306026985027
测试样本 0 的特征 'dp' 的 SHAP 值为：0.017681695457416838
测试样本 0 的特征 'fp' 的 SHAP 值为：0.09425726527794168
测试样本 0 的特征 'u0' 的 SHAP 值为：-0.03444763388132333
测试样本 0 的特征 'hv'' 的 SHAP 值为：0.0131649497331165
测试样本 0 的特征 'dp'' 的 SHAP 值为：0.014056114704068062
'''
'''
低[2]
所有测试样本的基准线为 0.08998946337309963
测试样本 2 的特征 'hv' 的 SHAP 值为：-0.018457907361063175
测试样本 2 的特征 'D' 的 SHAP 值为：0.00016231740567036745
测试样本 2 的特征 'R/D' 的 SHAP 值为：-0.00128148912372468
测试样本 2 的特征 'dp' 的 SHAP 值为：-0.012776682476702987
测试样本 2 的特征 'fp' 的 SHAP 值为：-0.0041453429772941724
测试样本 2 的特征 'u0' 的 SHAP 值为：-0.024949108015105054
测试样本 2 的特征 'hv'' 的 SHAP 值为：-0.018246362367813462
测试样本 2 的特征 'dp'' 的 SHAP 值为：-0.009309457031135254
'''














'''
# 创建一个新的Excel工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 写入数据
for col1,col2 in zip(y_test,y_test_pred):
    sheet.append([col1,col2])

# 指定Excel文件路径
excel_file_path = 'DE-SVR结果.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')
'''




'''
0.7
best_x: [1.40634562e+01 1.04217065e+01 3.36313549e-05] 
 best_y: [-0.99336359]
best_x[0]_C: 14.063456167204542
best_x[1]_gamma: 10.421706507320557
best_x[2]_epsilon: 3.363135491649024e-05
MAE_test: 0.006750151403826745
MSE_test: 0.00010991571077012003
RMSE_test: 0.010484069380260702
MAPE_test: 354.4231492536652
r2_score_test: 0.9933635934353467
EV_test: 0.9938342715050354

'''

'''
0.3
best_x: [3.33530610e+01 3.57227182e+01 3.25516897e-03] 
 best_y: [-0.9534819]
best_x[0]_C: 33.35306096447831
best_x[1]_gamma: 35.72271824237987
best_x[2]_epsilon: 0.003255168974038547
MAE_test: 0.02123502375369062
MSE_test: 0.0007704576265857885
RMSE_test: 0.027757118484918214
MAPE_test: 1851.562256278025
r2_score_test: 0.9534818997663156
EV_test: 0.9606608409958574

'''
'''
0.5
best_x: [1.44671970e+01 1.02891556e+01 1.02152162e-05] 
 best_y: [-0.99314905]
best_x[0]_C: 14.467196960464232
best_x[1]_gamma: 10.289155581250423
best_x[2]_epsilon: 1.0215216157886996e-05
MAE_test: 0.006800360122304953
MSE_test: 0.00011346915706968971
RMSE_test: 0.010652190247535467
MAPE_test: 350.5486702543286
r2_score_test: 0.9931490461774126
EV_test: 0.9937313778583885

'''
