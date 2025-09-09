from sklearn.datasets import make_blobs
from sklearn import datasets
import numpy as np


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler 


import pandas as pd




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
random_numbers1=[2586,2705,676,737,2673,1991,2122,39,575,927,1751,2797,351,1942,1266,2252,454,2142,1923,2798,1392,2783,2395,2376,1723,1618,468,2876,1827,1992,1814,2915,1301,1737,2899,2858,2159,1891,674,756,975,2400,390,950,1995,1980,1497,1686,1903,159,2941,1451]
#print("前半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于25%。
# 生成形状为 (52, 1) 的[0, 0.7)范围内的随机浮点数数组
random_numbers2=[1.01856810e-01,3.07743841e-01,4.64781840e-01,5.42833335e-02,1.33609373e-01,2.56455440e-01,1.48471008e-01,4.29799311e-01,3.54211604e-01,4.33062590e-01,1.38690907e-01,1.85543617e-02,9.23872532e-02,1.76667134e-01,5.43535070e-01,4.55348670e-01,6.19992229e-01,9.03036758e-02,6.46513113e-01,4.22874651e-01,5.66269548e-01,3.77662386e-01,5.16194149e-01,3.10418914e-01,3.35087615e-01,1.75828565e-01,4.41672034e-01,2.06300013e-01,3.85710297e-01,5.34978685e-01,5.67409635e-01,4.61249904e-01,3.35175096e-01,5.12532222e-01,5.36597998e-01,2.73545337e-01,2.69902097e-01,4.99628439e-01,3.51799926e-01,4.04386613e-01,1.72424568e-01,6.04389683e-01,6.28734282e-01,6.13691189e-01,4.42032631e-01,1.15802551e-01,6.72905615e-01,7.89853702e-02,3.00965823e-01,2.94514375e-01,4.30902285e-01,4.96740314e-01]
y_train[random_numbers1]=random_numbers2*y_train[random_numbers1] 
#print("前半部分加噪声：",y_train[random_numbers1])




##### 将异常值添加到后半部分的训练集---2950，5900
random_numbers1=[3155,4823,2996,5263,5458,3273,2952,4873,5431,3068,4017,5299,3800,4561,3533,3666,5591,5652,3021,5644,5432,3929,4974,4109,5818,4505,5682,3101,3722,4841,4977,4731,4125,5801,3204,3310,5700,4145,3752,3917,5829,5001,3374,4948,5889,5888,3998,3087,5100,5273,4227,4377]
#print("后半部分未加噪声：",y_train[random_numbers1])
#随机倍数----相对误差大于25%。
# 生成形状为 (52, 1) 的[0, 0.7)范围内的随机浮点数数组
random_numbers2= [ 1.51473227, 1.53023626, 1.72639175, 1.62000778, 1.34123776, 1.84847664, 1.91001952, 1.9375036 , 1.8646492 , 1.8719561 , 1.72283157, 1.55783949, 1.44123603, 1.89144653, 1.94597092, 1.36630285, 1.59121322, 1.74290838, 1.99591493, 1.76573934, 1.82311904, 1.98772126, 1.6812243 , 1.55806218, 1.71309165, 1.59269718, 1.42917829, 1.8067828 , 1.99931802, 1.46178011, 1.9450033 , 1.38395181, 1.8751847 , 1.46345825, 1.43835529, 1.74149792, 1.38593215, 1.79337244, 1.90787809, 1.99543535, 1.44207112, 1.65592863, 1.80011069, 1.36656167, 1.80095207, 1.82227447, 1.31165131, 1.4223794 , 1.35985004, 1.73335289, 1.71407402, 1.53085092]
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


clf2=LGBMRegressor(n_estimators=201,
                   max_depth=18,
                   num_leaves=91,
                   min_data_in_leaf=1,
                   max_bin=90,
                   feature_fraction=0.5,
                   bagging_fraction=0.1,
                   bagging_freq=0,
                   reg_alpha=0.25,
                   reg_lambda=0.0,
                   min_split_gain=0,  
                   random_state=90)

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

clf4=SVR(kernel='rbf',C=100,epsilon=0.8,gamma=1)

# 软投票
estimators=[ ('rf',clf1),( 'lgbm',clf2),( 'xgb',clf3),( 'svr',clf4)]
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
MAE: 1.1059199254164647
MSE: 2.7186035707490555
RMSE: 1.6488188410947564
MAPE: 0.04669119123755068
r2_score: 0.9638392745209532
'''
