import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from sklearn import tree
'''
import pydotplus
from IPython.display import Image

import graphviz
from sklearn.tree import export_graphviz
'''
# 导入 permutation_importance 函数
from sklearn.inspection import permutation_importance

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


'''
        Feature  Importance
0            Hv    0.606298
4            u0    0.242951
10  ρw'_Hv'_dp'    0.096359
3            dp    0.023833
2            Vo    0.015310
7           dp'    0.013547
5            mp    0.001702
1            ρw    0.000000
6           Hv'    0.000000
8           ρw'    0.000000
9       ρw'_Hv'    0.000000
'''



# 可视化决策树

#第一种方法graphviz
'''
# 导出决策树为DOT格式
dot_data = export_graphviz(regressor,
                           out_file=None,
                           feature_names=df.columns[:-1],  # 特征名称，假设最后一列是目标列
                           filled=True,
                           rounded=True,
                           special_characters=True)

# 使用Graphviz显示决策树
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # 保存为PDF文件
graph.view()  # 打开图形查看器
'''



'''
# 导出决策树为DOT格式
dot_data = export_graphviz(
    regressor,
    out_file=None,
    feature_names=x_train.columns,
    filled=True,
    rounded=True,
    special_characters=True
)



# 手动添加字体大小设置
dot_data = dot_data.replace("]", " fontsize=54]")

# 使用Graphviz显示决策树
graph = graphviz.Source(dot_data)

# 保存为图片文件
graph.render("decision_tree", format="png")  # 保存为PNG文件
graph.view()  # 打开图形查看器
'''



##第二种方法
#plt.figure(figsize=(20, 10))
'''
tree.plot_tree(regressor, feature_names=x.columns, filled=True)
plt.show()
'''
'''
tree.plot_tree(regressor, 
               feature_names=x.columns,  # 特征名称
               filled=True,  # 是否填充颜色
               rounded=True,  # 是否圆角
               fontsize=18)  # 字体大小

# 调整布局以防止重叠
plt.tight_layout()

plt.show()


# 查看树的结构信息
n_nodes = regressor.tree_.node_count
children_left = regressor.tree_.children_left
children_right = regressor.tree_.children_right
feature = regressor.tree_.feature
threshold = regressor.tree_.threshold


print("Decision Tree Structure:")
for i in range(n_nodes):
    if children_left[i]!= children_right[i]:  # 非叶子节点
        feature_name = x.columns[feature[i]]
        print(f"Node {i}: Feature {feature_name} <= {threshold[i]}")
'''

'''
# 查看特征重要性（训练集）
feature_importances = regressor.feature_importances_
print("Feature Importances:")
for i, feature_name in enumerate(x.columns):
    print(f"{feature_name}: {feature_importances[i]}")
'''
'''
Decision Tree Structure:
Node 0: Feature Hv <= 0.3815789520740509
Node 1: Feature ρw'_Hv'_dp' <= 0.9533059298992157
Node 2: Feature dp <= 0.4166666716337204
Node 4: Feature Vo <= 0.1666666716337204
Node 5: Feature u0 <= 0.5833333432674408
Node 8: Feature u0 <= 0.5833333432674408
Node 9: Feature u0 <= 0.4166666716337204
Node 12: Feature ρw'_Hv'_dp' <= 0.73663529753685
Node 13: Feature u0 <= 0.75
Node 15: Feature Vo <= 0.8333333432674408
Node 20: Feature dp' <= 0.41678497195243835
Node 21: Feature u0 <= 0.5000000149011612
Node 24: Feature u0 <= 0.8333333432674408
Node 25: Feature Vo <= 0.1666666716337204
Node 27: Feature ρw'_Hv'_dp' <= 0.5120910257101059
Node 29: Feature dp <= 0.5833333432674408
Node 32: Feature mp <= 0.3333333432674408

Feature Importances:
Hv: 0.6062976335552297
ρw: 0.0
Vo: 0.01531034252533661
dp: 0.02383306006126348
u0: 0.24295118032040763
mp: 0.0017023309269909527
Hv': 0.0
dp': 0.01354680381852083
ρw': 0.0
ρw'_Hv': 0.0
ρw'_Hv'_dp': 0.09635864879225083

'''

# 查看特征重要性（测试集）

# 计算测试集上的特征重要性
'''
result = permutation_importance(regressor, x_test, y_test, n_repeats=10, random_state=42)
feature_importances = result.importances_mean

# 输出特征重要性得分
feature_names = x.columns
print("\nFeature importances on test set:")
for name, importance in zip(feature_names, feature_importances):
    print(f"{name}: {importance}")
'''
'''
Feature importances on test set:
Hv: 0.7968247077410192
ρw: 0.0
Vo: 0.05104803258181859
dp: 0.33489059226044937
u0: 0.29542681506049384
mp: 0.0
Hv': 0.0
dp': 0.0014629140342971691
ρw': 0.0
ρw'_Hv': 0.0
ρw'_Hv'_dp': 0.3686404818542145
'''

