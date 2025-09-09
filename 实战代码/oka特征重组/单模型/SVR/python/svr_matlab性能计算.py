import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,explained_variance_score
import numpy as np
from sklearn import metrics


# 读取包含预测值和实际值的Excel文件
df = pd.read_excel("svr_matlab.xlsx")  # 替换为你的文件路径

# 提取预测值和实际值列
y_test = df['y_test']
y_pred = df['y_pred']

# 计算MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')

# 计算MSE
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# 计算RMSE
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# 计算MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f'MAPE: {mape}%')

# 计算R2
r2 = r2_score(y_test, y_pred)
print(f'R2: {r2}')


EV=metrics.explained_variance_score(y_test, y_pred)
print('EV:', EV)

'''
实战\新\oka特征重组\单模型\SVR\svr_matlab性能计算.py 
MAE: 0.03069352953336251
MSE: 0.004095740632355003
RMSE: 0.06399797365819486
MAPE: 11912.574790362374%
R2: 0.6552891711002156
EV: 0.6554299930700114
'''
