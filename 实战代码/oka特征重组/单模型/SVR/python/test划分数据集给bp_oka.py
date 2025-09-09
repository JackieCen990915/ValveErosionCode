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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import csv


from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt

from openpyxl import Workbook









######################1.导入数据######################
df=pd.read_excel('../../0229_冲蚀数据统计_Num_GA.xlsx',sheet_name='Sheet1')








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










print("---------------x_train------------------")
print(x_train,type(x_train))
print("---------------y_train------------------")
print(y_train,type(y_train))
print("---------------x_test------------------")
print(x_test,type(x_test))
print("---------------y_test------------------")
print(y_test,type(y_test))







#######写入xlsx###############
x_train.to_excel('x_train.xlsx',index=False)
y_train.to_excel('y_train.xlsx',index=False)
x_test.to_excel('x_test.xlsx',index=False)
y_test.to_excel('y_test.xlsx',index=False)
