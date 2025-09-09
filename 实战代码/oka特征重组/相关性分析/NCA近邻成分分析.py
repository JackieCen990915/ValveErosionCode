import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis






######################1.导入数据######################
df=pd.read_excel('D:/A研2项目/课题论文/2机理数据融合模型/冲蚀/周报&开会讨论_研二下/实战所用数据/弯头气固/0229_冲蚀数据统计_Num_GA_oka.xlsx',sheet_name='Sheet1')





######################2.提取特征变量######################
x=df.drop(columns='erosion_rate')
y=df['erosion_rate']


print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))




######################3.划分训练集和测试集######################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)



######################4.NCA  特征重要性######################

# 创建NCA模型
nca = NeighborhoodComponentsAnalysis(random_state=123)

# 构建pipeline，先进行特征的标准化，然后应用NCA
pipeline = make_pipeline(StandardScaler(), nca)

# 拟合pipeline
pipeline.fit(x_train, y_train)

# 获取特征的重要性
feature_importances = np.abs(nca.components_).sum(axis=1)

# 输出特征重要性
print("特征重要性：", feature_importances)
