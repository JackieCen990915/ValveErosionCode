from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler 


import pandas as pd

from sklearn.svm import SVR


######################1.导入数据######################
df=pd.read_excel('../../../../0114_cx_整理数据_17_最终.xlsx',sheet_name='Sheet1')








######################2.提取特征变量######################
x=df.drop(columns='ROP ')
y=df['ROP ']


print("---------------x------------------")
print(x)
print(type(x))
print("---------------y------------------")
print(y)
print(type(y))






######################3.划分训练集和测试集######################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)




#7378*0.8=5,902.4----5900吧，5900*0.5=2,950
##### 将异常值添加到训练集#####################################
##### 将异常值添加到前半部分的训练集---0，2949
random_numbers1=[1547,14,1857,1755,661,660,782,2481,2332,1258,548,1923,1667,2759,843,619,208,211,1218,17,2788,2165,885,2698,70,2247,903,1357,2440,512,1258,29,2445,70,1603,87,1895,2807,298,840,1781,2363,2546,2010,1767,2574,1269,217,2007,2702,1770,246,1531,1288,1387,1484,1129,1097,567,358,964,2379,2446,1301,2243,721,442,1104,2153,689,2036,891,2833,579,1856,2585,942,34]
#print("前半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于25%。
# 生成形状为 (78, 1) 的[0, 0.7)范围内的随机浮点数数组
random_numbers2=[0.28691412, 0.59691485, 0.2829274 , 0.28639757, 0.19436294, 0.51393646, 0.20823995, 0.03068926, 0.19193054, 0.64850154, 0.29846472, 0.31995556, 0.20953405, 0.5606966 , 0.37849613, 0.59756751, 0.08706218, 0.57665398, 0.38908464, 0.50954296, 0.6966508 , 0.50254307, 0.10289169, 0.61864009, 0.16942152, 0.37073167, 0.14924525, 0.34016078, 0.59811729, 0.41640596, 0.58120718, 0.68744579, 0.45462405, 0.49901976, 0.38612244, 0.22573166, 0.15608352, 0.22362077, 0.5273381 , 0.02672154, 0.1484423 , 0.08956092, 0.52015675, 0.23831257, 0.37586648, 0.66343232, 0.49777506, 0.64182169, 0.33424689, 0.26936921, 0.28976906, 0.36964762, 0.43064792, 0.09771016, 0.58499023, 0.47812157, 0.58449193, 0.12073031, 0.55881853, 0.66896319, 0.05353452, 0.68245482, 0.31207958, 0.66654959, 0.64448537, 0.60652446, 0.13798288, 0.4489062 , 0.55659062, 0.42570907, 0.59038404, 0.63362994, 0.20522486, 0.28058752, 0.24074398, 0.63696107, 0.25501987, 0.25743283]
y_train[random_numbers1]=random_numbers2*y_train[random_numbers1] 
#print("前半部分加噪声：",y_train[random_numbers1])




##### 将异常值添加到后半部分的训练集---2950，5900
random_numbers1=[5001,3839,5499,5670,3060,5706,4220,5442,4856,3088,4676,5348,4152,3190,3855,2978,5143,5463,4677,3992,3540,4746,3111,5638,4032,4459,4495,3997,3014,3219,5472,3595,3173,5579,5530,4463,4160,3193,4439,4797,4410,3584,3702,2984,4240,3689,3108,4314,4628,4851,5465,3669,3727,4814,5677,4002,3967,4389,3367,4101,3902,3350,3083,3787,3193,3258,4634,3600,4019,4173,3716,4396,5032,4788,5378,3453,4235,4647]
#print("后半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于25%。
# 生成形状为 (78, 1) 的[1.3, 2)范围内的随机浮点数数组
random_numbers2= [1.95431493, 1.59613174, 1.86028138, 1.75498948, 1.99917973, 1.55631169, 1.62278518, 1.46229252, 1.6217385 , 1.64394409, 1.4548752 , 1.93585559, 1.98453474, 1.93376602, 1.83443676, 1.84991975, 1.95075516, 1.4918816 , 1.70418637, 1.78280969, 1.76633115, 1.32757086, 1.8650841 , 1.76238862, 1.44247587, 1.45657194, 1.62846556, 1.80757273, 1.98560973, 1.34714407, 1.83612744, 1.67496349, 1.40223992, 1.5421093 , 1.78686405, 1.65364691, 1.38107973, 1.85776349, 1.61854925, 1.30842964, 1.4880365 , 1.57207155, 1.6215919 , 1.93562333, 1.56104591, 1.84106942, 1.71752031, 1.86213108, 1.94975049, 1.81172995, 1.69288353, 1.83573642, 1.53291883, 1.57050409, 1.8533868 , 1.40618689, 1.94829694, 1.81623776, 1.58953638, 1.77953812, 1.36005302, 1.31929714, 1.36059954, 1.60637366, 1.51549   , 1.53635907, 1.31000773, 1.84442958, 1.43096203, 1.6467071 , 1.83515166, 1.84947841, 1.8437436 , 1.46802665, 1.98139352, 1.88883128, 1.50493733, 1.86633857]
y_train[random_numbers1]=random_numbers2*y_train[random_numbers1] 
#print("后半部分未加噪声：",y_train[random_numbers1])




# 使用MinMaxScaler进行归一化,对结果没影响------------------
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

 
 


clf1=RandomForestRegressor(n_estimators=181,
                           max_depth=20,
                           max_features=4,
                           min_samples_leaf=1,
                           min_samples_split=2,
                           random_state=90)


clf2=SVR(kernel='rbf',C=100,epsilon=0.8,gamma=1)



clf3=xgb.XGBRegressor(
                  learning_rate=0.1,
                  max_depth=18,
                  min_child_weight=5,
                  gamma=0,
                  subsample=0.6,
                  colsample_bytree=0.25,
                  alpha=0,
                  reg_lambda=0.1,
                  random_state=90)
          


estimators=[ ('rf',clf1),( 'svr',clf2),( 'xgb',clf3)]
final_estimator=LinearRegression()

eclf=StackingRegressor(estimators=estimators,
                       final_estimator=final_estimator)



eclf.fit(x_train,y_train)
y_pred = eclf.predict(x_test)




######################6.评估模型######################

MAE=metrics.mean_absolute_error(y_test, y_pred)
MSE=metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(MSE)
MAPE=metrics.mean_absolute_percentage_error(y_test, y_pred)
R2=metrics.r2_score(y_test, y_pred)




print('MAE:', MAE)
print('MSE:', MSE)
print('RMSE:', RMSE)
print('MAPE:', MAPE)
print('r2_score:', R2)


'''
MAE: 0.7855659989974344
MSE: 1.7876948134665216
RMSE: 1.337047049832773
MAPE: 0.03368850533401385
r2_score: 0.9762214902953772
'''
