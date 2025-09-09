from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 假设有一个回归数据集 X, y
# X 是特征，y 是目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练基本模型
base_model_lr = LinearRegression()
base_model_dt = DecisionTreeRegressor()

base_model_lr.fit(X_train, y_train)
base_model_dt.fit(X_train, y_train)

# 生成基本模型的预测结果
predictions_lr = base_model_lr.predict(X_test)
predictions_dt = base_model_dt.predict(X_test)

# 使用基本模型的预测结果和元模型进行训练
meta_model = LinearRegression()
meta_model.fit(np.column_stack((predictions_lr, predictions_dt)), y_test)

# 预测
blend_predictions = meta_model.predict(np.column_stack((base_model_lr.predict(X_test), base_model_dt.predict(X_test))))

# 评估性能
mse_blend = mean_squared_error(y_test, blend_predictions)
print(f'Blending Mean Squared Error: {mse_blend}')
