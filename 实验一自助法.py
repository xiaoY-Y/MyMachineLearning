import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def bootstrap_split(X, y, num_samples=None, random_state=None):
    """
    使用自助法拆分数据集
    参数:
    X (array-like): 特征矩阵，形状为 (样本数量, 特征数量)
    y (array-like): 标签向量，形状为 (样本数量,)
    num_samples (int, 可选): 自助抽样的样本数量，默认为原始数据集的样本数量
    random_state (int, 可选): 随机种子，用于确保结果的可重复性
    返回:
    X_train (array-like): 训练集的特征矩阵
    y_train (array-like): 训练集的标签向量
    X_test (array-like): 测试集的特征矩阵
    y_test (array-like): 测试集的标签向量
    """
    if num_samples is None:
        num_samples = len(X)
    if random_state is not None:
        np.random.seed(random_state)
    # 生成自助抽样的索引（有放回抽样）
    bootstrap_indices = np.random.choice(len(X), size=num_samples, replace=True)
    # 生成训练集
    X_train = X[bootstrap_indices]
    y_train = y[bootstrap_indices]
    # 生成测试集的索引（未被抽到的样本索引）
    test_indices = np.setdiff1d(np.arange(len(X)), bootstrap_indices)
    # 生成测试集
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test


# 读取文件
excel_file = pd.ExcelFile('data-EXP1--Real-vs-Pred-(rand)-5-6-7.xlsx')
# 选取数据集 data-Label-feature(0.9)
df = excel_file.parse('data-Label-feature(0.9)')

# 准备特征矩阵 X 和标签向量 y
X = df.drop(columns=['RealLabel']).values
y = df['RealLabel'].values

num_splits = 10  # 增加拆分次数以更明显观察波动
total_samples = len(X)
all_results = []

for split in range(num_splits):
    X_train, y_train, X_test, y_test = bootstrap_split(X, y, random_state=split)

    # 测试集样本数占总样本量的比例
    test_sample_count = len(y_test)
    test_proportion = test_sample_count / total_samples

    # 测试集中不同类别的样本数量及比例
    test_class_count = pd.Series(y_test).value_counts()
    test_class_proportion = test_class_count / test_sample_count

    result = {
        '拆分次数': split + 1,
        '测试集样本数': test_sample_count,
        '测试集样本数占比': test_proportion,
        '测试集类别 0 数量': test_class_count.get(0, 0),
        '测试集类别 0 比例': test_class_proportion.get(0, 0),
        '测试集类别 1 数量': test_class_count.get(1, 0),
        '测试集类别 1 比例': test_class_proportion.get(1, 0)
    }
    all_results.append(result)

# 调整 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 输出表格结果
results_df = pd.DataFrame(all_results)
print(results_df)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 调整字体大小
plt.rcParams['font.size'] = 5

# 绘制测试集样本数占总样本量的比例变化图
plt.figure(figsize=(6, 3))
plt.plot(results_df['拆分次数'], results_df['测试集样本数占比'], marker='o')
plt.xlabel('拆分次数')
plt.ylabel('测试集样本数占总样本量的比例')
plt.title('测试集样本数占比随拆分次数的变化')
plt.grid(True)

# 绘制测试集中不同类别的样本比例变化图
plt.figure(figsize=(6, 3))
plt.plot(results_df['拆分次数'], results_df['测试集类别 0 比例'], marker='o', label='类别 0 比例')
plt.plot(results_df['拆分次数'], results_df['测试集类别 1 比例'], marker='s', label='类别 1 比例')
plt.xlabel('拆分次数')
plt.ylabel('测试集中不同类别的样本比例')
plt.title('测试集中不同类别的样本比例随拆分次数的变化')
plt.legend()
plt.grid(True)

# 绘制测试集中不同类别的样本数量变化图
plt.figure(figsize=(6, 3))
plt.plot(results_df['拆分次数'], results_df['测试集类别 0 数量'], marker='o', label='类别 0 数量')
plt.plot(results_df['拆分次数'], results_df['测试集类别 1 数量'], marker='s', label='类别 1 数量')
plt.xlabel('拆分次数')
plt.ylabel('测试集中不同类别的样本数量')
plt.title('测试集中不同类别的样本数量随拆分次数的变化')
plt.legend()
plt.grid(True)

plt.show()
