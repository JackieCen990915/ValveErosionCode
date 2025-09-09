

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

#file_path=('D:/A研2项目/课题论文/2机理数据融合模型/弯头冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand_oka_相关性.xlsx')

file_path=('D:/A研2项目/课题论文/2机理数据融合模型/弯头冲蚀/周报&开会讨论_研二下/弯管实战/实战所用数据Sand/弯头气固/0409_冲蚀数据统计_Sand_相关性.xlsx')


df=pd.read_excel(file_path)
print(df)
print(df.head())
print(df.info())


#corr_=df.corr(method='pearson')
corr_=df.corr(method='spearman')
print(corr_)
print(type(corr_))

'''
# 将DataFrame写入Excel文件
corr_.to_excel('斯皮尔曼相关性系数.xlsx', index=False)
print("斯皮尔曼相关性系数.xlsx已写入")

'''
'''

           hv         D       R/D  ...       hv'       dp'        er
hv   1.000000 -0.572673  0.126498  ... -1.000000 -0.812693 -0.703140
D   -0.572673  1.000000  0.101750  ...  0.572673  0.493022  0.249155
R/D  0.126498  0.101750  1.000000  ... -0.126498 -0.125363 -0.161193
dp  -0.812693  0.493022 -0.125363  ...  0.812693  1.000000  0.856431
fp  -0.204281  0.170760  0.049335  ...  0.204281  0.412578  0.588901
u0  -0.819373  0.507237 -0.044037  ...  0.819373  0.818894  0.874954
hv' -1.000000  0.572673 -0.126498  ...  1.000000  0.812693  0.703140
dp' -0.812693  0.493022 -0.125363  ...  0.812693  1.000000  0.856431
er  -0.703140  0.249155 -0.161193  ...  0.703140  0.856431  1.000000


           hv         D       R/D        dp        fp        u0        er
hv   1.000000 -0.572673  0.126498 -0.812693 -0.204281 -0.819373 -0.703140
D   -0.572673  1.000000  0.101750  0.493022  0.170760  0.507237  0.249155
R/D  0.126498  0.101750  1.000000 -0.125363  0.049335 -0.044037 -0.161193
dp  -0.812693  0.493022 -0.125363  1.000000  0.412578  0.818894  0.856431
fp  -0.204281  0.170760  0.049335  0.412578  1.000000  0.345139  0.588901
u0  -0.819373  0.507237 -0.044037  0.818894  0.345139  1.000000  0.874954
er  -0.703140  0.249155 -0.161193  0.856431  0.588901  0.874954  1.000000
<class 'pandas.core.frame.DataFrame'>

'''



plt.rc('font',family='SimHei',size=8)
plt.subplots(figsize=(15,15))
#在上面显示数字大小
sns.heatmap(corr_,annot=True)
plt.show()

