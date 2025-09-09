

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#from scipy.stats.stats import pearsonr
import seaborn as sns




#相关性系数

'''
MaterialDensity VickersHardness	ExperimentalTime ParticleSize
ErosionAngle SandDischargeRate	GasFlowRate ParticleVelocity AirVelocity

ErosionRate
'''

file_path='../0114_cx_整理数据_17_最终.csv'
df=pd.read_csv(file_path)
print(df)
print(df.head())
print(df.info())


#corr_=df.corr(method='pearson')
corr_=df.corr(method='spearman')
print(corr_)
print(type(corr_))

# 将DataFrame写入Excel文件
corr_.to_excel('斯皮尔曼相关性系数.xlsx', index=False)
print("斯皮尔曼相关性系数.xlsx已写入")


'''

                WOH       WOB       SPP  ...     Angle      Azim       ROP
WOH        1.000000  0.143025 -0.655688  ...  0.959748 -0.959748 -0.521722
WOB        0.143025  1.000000 -0.381929  ...  0.229787 -0.229787 -0.396172
SPP       -0.655688 -0.381929  1.000000  ... -0.607047  0.607047  0.621791
RPM       -0.827697 -0.215908  0.575258  ... -0.869517  0.869517  0.687211
Torque     0.319620  0.339060 -0.215568  ...  0.336430 -0.336430  0.075019
Flow Rate -0.831256 -0.442814  0.761268  ... -0.854540  0.854540  0.538012
Bit Run    0.012154 -0.118851  0.398665  ...  0.052624 -0.052624  0.139282
Bit Time   0.467967 -0.073597  0.083977  ...  0.523793 -0.523793 -0.171064
GR         0.060152 -0.014885  0.073345  ...  0.100252 -0.100252 -0.174979
DT        -0.091046 -0.276022  0.255438  ... -0.104417  0.104417  0.116856
RMSL       0.566635  0.152770 -0.413742  ...  0.592464 -0.592464 -0.399277
RD         0.412375  0.117420 -0.317292  ...  0.446263 -0.446263 -0.360260
Depth      0.959748  0.229787 -0.607047  ...  1.000000 -1.000000 -0.602691
Angle      0.959748  0.229787 -0.607047  ...  1.000000 -1.000000 -0.602691
Azim      -0.959748 -0.229787  0.607047  ... -1.000000  1.000000  0.602691
ROP       -0.521722 -0.396172  0.621791  ... -0.602691  0.602691  1.000000

[16 rows x 16 columns]

'''


'''
plt.rc('font',family='SimHei',size=8)
plt.subplots(figsize=(15,15))
#在上面显示数字大小
sns.heatmap(corr_,annot=True)
plt.show()
'''
