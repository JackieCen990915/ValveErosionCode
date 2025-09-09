import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

#网格搜索
from sklearn.model_selection import GridSearchCV
#随机搜索
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook



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

'''
---------------x_train------------------
      Hv    ρw    Vo   dp  ...       dp'       ρw'   ρw'_Hv'  ρw'_Hv'_dp'
62  2.15  7980  0.50  400  ...  1.039633  0.000519  0.000473     0.000492
30  1.77  8030  0.50  400  ...  1.039633  0.000522  0.000487     0.000507
58  2.15  7980  0.25  400  ...  1.039633  0.000519  0.000473     0.000492
35  1.77  8030  0.75  400  ...  1.039633  0.000522  0.000487     0.000507
57  2.15  7980  0.25  300  ...  0.984332  0.000519  0.000473     0.000466
47  2.06  7930  0.50  500  ...  1.084658  0.000515  0.000473     0.000513
16  1.77  8030  0.75  600  ...  1.122890  0.000522  0.000487     0.000547
34  1.77  8030  0.75  400  ...  1.039633  0.000522  0.000487     0.000507
42  2.06  7930  0.25  400  ...  1.039633  0.000515  0.000473     0.000491
28  1.77  8030  0.50  400  ...  1.039633  0.000522  0.000487     0.000507
7   1.77  8030  0.50  400  ...  1.039633  0.000522  0.000487     0.000507
53  2.06  7930  1.00  300  ...  0.984332  0.000515  0.000473     0.000465
40  2.06  7930  0.25  200  ...  0.911348  0.000515  0.000473     0.000431
44  2.06  7930  0.50  200  ...  0.911348  0.000515  0.000473     0.000431
46  2.06  7930  0.50  400  ...  1.039633  0.000515  0.000473     0.000491
19  1.77  8030  1.00  300  ...  0.984332  0.000522  0.000487     0.000480
56  2.15  7980  0.25  200  ...  0.911348  0.000519  0.000473     0.000431
39  1.77  8030  1.00  400  ...  1.039633  0.000522  0.000487     0.000507
25  1.77  8030  0.25  400  ...  1.039633  0.000522  0.000487     0.000507
38  1.77  8030  1.00  400  ...  1.039633  0.000522  0.000487     0.000507
13  1.77  8030  0.75  300  ...  0.984332  0.000522  0.000487     0.000480
50  2.06  7930  0.75  400  ...  1.039633  0.000515  0.000473     0.000491
3   1.77  8030  0.25  500  ...  1.084658  0.000522  0.000487     0.000529
17  1.77  8030  0.75  700  ...  1.156265  0.000522  0.000487     0.000564
8   1.77  8030  0.50  500  ...  1.084658  0.000522  0.000487     0.000529
55  2.06  7930  1.00  500  ...  1.084658  0.000515  0.000473     0.000513
6   1.77  8030  0.50  300  ...  0.984332  0.000522  0.000487     0.000480
36  1.77  8030  1.00  400  ...  1.039633  0.000522  0.000487     0.000507
64  2.15  7980  0.75  200  ...  0.911348  0.000519  0.000473     0.000431
69  2.15  7980  1.00  300  ...  0.984332  0.000519  0.000473     0.000466
68  2.15  7980  1.00  200  ...  0.911348  0.000519  0.000473     0.000431
15  1.77  8030  0.75  500  ...  1.084658  0.000522  0.000487     0.000529
27  1.77  8030  0.25  400  ...  1.039633  0.000522  0.000487     0.000507
41  2.06  7930  0.25  300  ...  0.984332  0.000515  0.000473     0.000465
26  1.77  8030  0.25  400  ...  1.039633  0.000522  0.000487     0.000507
48  2.06  7930  0.75  200  ...  0.911348  0.000515  0.000473     0.000431
24  1.77  8030  0.25  400  ...  1.039633  0.000522  0.000487     0.000507
59  2.15  7980  0.25  500  ...  1.084658  0.000519  0.000473     0.000513
63  2.15  7980  0.50  500  ...  1.084658  0.000519  0.000473     0.000513
11  1.77  8030  0.75  100  ...  0.798893  0.000522  0.000487     0.000389
32  1.77  8030  0.75  400  ...  1.039633  0.000522  0.000487     0.000507
66  2.15  7980  0.75  400  ...  1.039633  0.000519  0.000473     0.000492
61  2.15  7980  0.50  300  ...  0.984332  0.000519  0.000473     0.000466
37  1.77  8030  1.00  400  ...  1.039633  0.000522  0.000487     0.000507
29  1.77  8030  0.50  400  ...  1.039633  0.000522  0.000487     0.000507
43  2.06  7930  0.25  500  ...  1.084658  0.000515  0.000473     0.000513
65  2.15  7980  0.75  300  ...  0.984332  0.000519  0.000473     0.000466
1   1.77  8030  0.25  300  ...  0.984332  0.000522  0.000487     0.000480
52  2.06  7930  1.00  200  ...  0.911348  0.000515  0.000473     0.000431
21  1.77  8030  1.00  500  ...  1.084658  0.000522  0.000487     0.000529
2   1.77  8030  0.25  400  ...  1.039633  0.000522  0.000487     0.000507
23  1.77  8030  1.00  700  ...  1.156265  0.000522  0.000487     0.000564
20  1.77  8030  1.00  400  ...  1.039633  0.000522  0.000487     0.000507
60  2.15  7980  0.50  200  ...  0.911348  0.000519  0.000473     0.000431
14  1.77  8030  0.75  400  ...  1.039633  0.000522  0.000487     0.000507
51  2.06  7930  0.75  500  ...  1.084658  0.000515  0.000473     0.000513

[56 rows x 11 columns]
<class 'pandas.core.frame.DataFrame'>
---------------x_test------------------
      Hv    ρw    Vo   dp  ...       dp'       ρw'   ρw'_Hv'  ρw'_Hv'_dp'
22  1.77  8030  1.00  600  ...  1.122890  0.000522  0.000487     0.000547
0   1.77  8030  0.25  200  ...  0.911348  0.000522  0.000487     0.000444
49  2.06  7930  0.75  300  ...  0.984332  0.000515  0.000473     0.000465
4   1.77  8030  0.25  600  ...  1.122890  0.000522  0.000487     0.000547
54  2.06  7930  1.00  400  ...  1.039633  0.000515  0.000473     0.000491
18  1.77  8030  1.00  200  ...  0.911348  0.000522  0.000487     0.000444
10  1.77  8030  0.50  700  ...  1.156265  0.000522  0.000487     0.000564
33  1.77  8030  0.75  400  ...  1.039633  0.000522  0.000487     0.000507
45  2.06  7930  0.50  300  ...  0.984332  0.000515  0.000473     0.000465
12  1.77  8030  0.75  200  ...  0.911348  0.000522  0.000487     0.000444
31  1.77  8030  0.50  400  ...  1.039633  0.000522  0.000487     0.000507
9   1.77  8030  0.50  600  ...  1.122890  0.000522  0.000487     0.000547
67  2.15  7980  0.75  500  ...  1.084658  0.000519  0.000473     0.000513
5   1.77  8030  0.50  200  ...  0.911348  0.000522  0.000487     0.000444

[14 rows x 11 columns]
<class 'pandas.core.frame.DataFrame'>
---------------y_train------------------
62    0.000503
30    0.014860
58    0.001075
35    0.020600
57    0.000620
47    0.000398
16    0.014330
34    0.018210
42    0.001307
28    0.004750
7     0.009180
53    0.000411
40    0.000090
44    0.002030
46    0.000496
19    0.008620
56    0.000096
39    0.020700
25    0.007970
38    0.015310
13    0.009000
50    0.000047
3     0.016033
17    0.026500
8     0.012390
55    0.000038
6     0.008550
36    0.004280
64    0.007390
69    0.000465
68    0.007140
15    0.013980
27    0.025000
41    0.001921
26    0.021300
48    0.001128
24    0.004490
59    0.001907
63    0.000362
11    0.005870
32    0.004300
66    0.000049
61    0.000166
37    0.007140
29    0.006870
43    0.001675
65    0.003850
1     0.010670
52    0.007560
21    0.011920
2     0.013670
23    0.020600
20    0.010540
60    0.002170
14    0.010620
51    0.000073
Name: er, dtype: float64
<class 'pandas.core.series.Series'>
---------------y_test------------------
22    0.015090
0     0.008190
49    0.003840
4     0.020600
54    0.000491
18    0.007440
10    0.023600
33    0.006770
45    0.000186
12    0.007080
31    0.019250
9     0.014050
67    0.000063
5     0.007710
Name: er, dtype: float64
<class 'pandas.core.series.Series'>
'''


# 使用MinMaxScaler进行归一化,对结果影响------------------
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
print('-------------------')



######################评估模型(测试集)######################
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
print('-------------------')






######################使用 SHAP 模型解释######################




# 生成shap算法多特征交互图
explainer = shap.TreeExplainer(regressor)
'''
shap_values = explainer.shap_values(x_test)
shap_interaction_values = explainer.shap_interaction_values(x_test)
shap.summary_plot(shap_interaction_values, x_test, plot_type="interaction")
'''
shap_values = explainer.shap_values(x_train)
shap_interaction_values = explainer.shap_interaction_values(x_train)
shap.summary_plot(shap_interaction_values, x_train, plot_type="interaction")



'''
# 测试集？

# 计算 SHAP 值
explainer = shap.TreeExplainer(regressor)
#输出numpy.array数组
#shap_values = explainer.shap_values(x_test)
#shap_values = explainer.shap_values(x)
shap_values = explainer.shap_values(x_train)



# 获取特征重要性
feature_importances = np.abs(shap_values).mean(axis=0)
feature_names = x.columns
# 打印特征重要性
for feature, importance in zip(feature_names, feature_importances):
    print(f"{feature}: {importance}")



# 1）可视化所有样本的SHAP值摘要 点图默认
#shap.summary_plot(shap_values, x_test, feature_names=feature_names)
#shap.summary_plot(shap_values, x, feature_names=feature_names)
shap.summary_plot(shap_values, x_train, feature_names=feature_names)




# 2）可视化所有样本的SHAP特征重要性 条形图
#shap.summary_plot(shap_values, x_test, feature_names=feature_names, plot_type='bar')
#shap.summary_plot(shap_values, x, feature_names=feature_names, plot_type='bar')
shap.summary_plot(shap_values, x_train, feature_names=feature_names, plot_type='bar')
'''

'''
Hv: 0.005203914260846203
ρw: 0.0
Vo: 0.00062044350330581
dp: 0.0006440236998828418
u0: 0.0019510430723761517
mp: 8.12516665364508e-05
Hv': 0.0
dp': 0.000530462425711419
ρw': 0.0
ρw'_Hv': 0.0
ρw'_Hv'_dp': 0.0006248773930809742

'''




'''

# 可视化单样本的 SHAP 值瀑布图-------咋不行----终于可以了。除了x_test的数值是归一化的，别的都可以显示出来了，包括特征名
feature_names = x.columns
#x_train_columns=pd.DataFrame(x_train, columns=['hv', 'D', 'R/D','dp','mp','u0',"hv'","dp'"])
x_train_columns=pd.DataFrame(x_train, columns=['Hv','ρw','Vo','dp','u0','mp',"Hv'","dp'","ρw'","ρw'_Hv'","ρw'_Hv'_dp'"])
print("x_train_colums:",x_train_columns)


#训练集
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
    print(f"训练样本 {sample_index} 的特征 '{feature_name}' 的 SHAP 值为：{value}")


shap.plots.waterfall(specific_sample_shap_value)
'''



'''
所有训练样本的基准线为 0.007521765535714285
训练样本 3 的特征 'Hv' 的 SHAP 值为：0.005269221755952375
训练样本 3 的特征 'ρw' 的 SHAP 值为：0.0
训练样本 3 的特征 'Vo' 的 SHAP 值为：-0.0004333531324404727
训练样本 3 的特征 'dp' 的 SHAP 值为：0.001354031684027779
训练样本 3 的特征 'u0' 的 SHAP 值为：0.004566789350198428
训练样本 3 的特征 'mp' 的 SHAP 值为：0.00010595535714285113
训练样本 3 的特征 'Hv'' 的 SHAP 值为：0.0
训练样本 3 的特征 'dp'' 的 SHAP 值为：-0.00022531756448413527
训练样本 3 的特征 'ρw'' 的 SHAP 值为：0.0
训练样本 3 的特征 'ρw'_Hv'' 的 SHAP 值为：0.0
训练样本 3 的特征 'ρw'_Hv'_dp'' 的 SHAP 值为：-0.00026909298611112187
'''













'''
# 3）可视化单样本的SHAP值力图
# 基础值（通常是模型的平均预测值）
base_value = explainer.expected_value
# 选择一个测试样本进行可视化
single_sample_index = 1 
single_shap_value = shap_values[single_sample_index]

print("shap_values:",shap_values)
print("single_shap_value:",single_shap_value)
print("x_test[single_sample_index]:",x_test[single_sample_index])

'''

'''
#matplotlib=True
#shap.force_plot(base_value, single_shap_value, x_test[single_sample_index], feature_names=feature_names, matplotlib=True)
#shap.force_plot(base_value, single_shap_value, x.iloc[single_sample_index], feature_names=feature_names, matplotlib=True)




#可视化单样本的SHAP值的决策图
#可视化单样本的SHAP值的局部条形图---和力图、瀑布图、决策图相通
#输出shap.Explanation对象
#shap_values2 = explainer(x_test) 
#shap.plots.bar(shap_values2[single_sample_index],show_data=True)

'''




# 可视化单样本的 SHAP 值瀑布图-------咋不行----终于可以了。除了x_test的数值是归一化的，别的都可以显示出来了，包括特征名
#explainer2 = shap.TreeExplainer(regressor,x_test)的x_test要传入！！！！！！
'''
explainer2 = shap.TreeExplainer(regressor,x)
shap_values2 = explainer2(x) 
shap.plots.waterfall(shap_values2[single_sample_index])
'''

'''
#x_test_columns=pd.DataFrame(x_test, columns=['hv', 'D', 'R/D','dp','fp','u0',"hv'","dp'"])
x_test_columns=pd.DataFrame(x_test, columns=['Hv','ρw','Vo','dp','u0','mp',"Hv'","dp'","ρw'","ρw'_Hv'","ρw'_Hv'_dp'"])
print("x_test_colums:",x_test_columns)

#x_test=x_test_columns.values
explainer2 = shap.TreeExplainer(regressor,x_test_columns)
shap_values2 = explainer2(x_test_columns) 
shap.plots.waterfall(shap_values2[single_sample_index])

'''

'''
# 可视化单样本的 SHAP 值决策图
#shap.decision_plot(base_value,single_shap_value, feature_names)


# 可视化所有样本的单输入特征和输出特征ER的依赖图
#print("x_test.columns:",x_test.columns)
#shap.dependence_plot('hv',shap_values,x_test,interaction_index=None,show=False)报错无'hv'列
#shap.dependence_plot('u0',shap_values,x,interaction_index=None,show=False)

'''










#######将y_test和y_pred写入xlsx，后续画实际和预测的图用###############
'''
# 创建一个新的Excel工作簿
workbook = Workbook()

# 获取默认的工作表
sheet = workbook.active

# 写入数据
for col1,col2 in zip(y_test,y_pred):
    sheet.append([col1,col2])

# 指定Excel文件路径
excel_file_path = '0306_RF_yTest_yPred_网格搜索.xlsx'

# 保存工作簿到文件
workbook.save(excel_file_path)

print(f'Data has been written to {excel_file_path}')
'''






