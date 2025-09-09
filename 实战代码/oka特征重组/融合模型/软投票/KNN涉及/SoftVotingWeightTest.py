from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# 生成示例数据
X, y = ...  # 你的特征和目标值

# 定义基础模型
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    SVR()
]

# 使用交叉验证得到基础模型的预测结果
predictions = []
for model in models:
    preds = cross_val_predict(model, X, y, cv=5, method='predict')
    predictions.append(preds)

# 定义均方误差损失函数
def mse_loss(weights):
    ensemble_preds = np.dot(weights, predictions)
    return mean_squared_error(y, ensemble_preds)

# 初始权重
initial_weights = np.ones(len(models)) / len(models)

# 最小化均方误差损失函数，得到优化后的权重
result = minimize(mse_loss, initial_weights, method='L-BFGS-B')

# 优化后的权重
optimized_weights = result.x
print("优化后的权重:", optimized_weights)
