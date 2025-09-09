
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
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer

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
[0.21405195 0.00143576 0.00098543 0.1363231  0.00191829 0.04114563
 0.0293114  0.01545097 0.09404084 0.00120071 0.39363942 0.37407862
 0.00176222 0.0006715  0.0438261 ]
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








#########使用 LIME 模型解释##########

feature_names = x.columns

# Initialize a LimeTabularExplainer
explainer = LimeTabularExplainer(training_data=x_train, mode="regression",verbose=True, feature_names=feature_names)
 
# Select a sample instance for explanation
sample_instance = x_test[0]
 
# Explain the prediction for the sample instance
explanation = explainer.explain_instance(sample_instance, regressor.predict,num_features=8)
 
# Print the explanation
explanation.show_in_notebook(show_table=True, show_all=True)

explanation.as_pyplot_figure()
explanation.as_list()












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
