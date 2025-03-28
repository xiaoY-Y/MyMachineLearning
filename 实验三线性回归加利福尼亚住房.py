import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 100
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载加利福尼亚住房数据集
housing = fetch_california_housing()
X = housing.data
y = housing.target
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 初始化线性回归模型
model = LinearRegression()
# 训练模型
model.fit(X_train, y_train)
# 预测
y_pred = model.predict(X_test)
# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")
# 5折交叉验证
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"5折交叉验证的均方根误差: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# 绘制预测值与真实值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', label='预测值与真实值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='理想预测线')
plt.xlabel('真实房价中位数')
plt.ylabel('预测房价中位数')
plt.title('线性回归预测加利福尼亚房价结果')
plt.legend()
plt.show()

# 绘制残差图
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, c='red', marker='o')
plt.axhline(y=0, color='k', lw=2)
plt.xlabel('预测房价中位数')
plt.ylabel('残差')
plt.title('线性回归残差图')
plt.show()