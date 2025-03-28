import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def hold_out_split(X, y, test_size=0.2, random_state=None):
    """
    实现留出法拆分数据集
    参数:
    X (array - like): 特征矩阵，形状为 (样本数量, 特征数量)。其中每一行代表一个样本，每一列代表一个特征。
    y (array - like): 标签向量，形状为 (样本数量,)。对应特征矩阵中每个样本的真实标签。
    test_size (float): 测试集所占的比例，范围为 (0, 1)。例如设置为 0.2 表示测试集占总样本数的 20%。
    random_state (int): 随机种子，用于保证结果可重复。当传入一个整数时，每次调用该函数在相同参数下将得到相同的拆分结果。
    返回:
    X_train (array - like): 训练集的特征矩阵，形状为 (训练集样本数量, 特征数量)。
    X_test (array - like): 测试集的特征矩阵，形状为 (测试集样本数量, 特征数量)。
    y_train (array - like): 训练集的标签向量，形状为 (训练集样本数量,)。
    y_test (array - like): 测试集的标签向量，形状为 (测试集样本数量,)。
    """
    # 如果传入了随机种子，则设置 numpy 的随机数生成器的种子，保证结果可重复
    if random_state is not None:
        np.random.seed(random_state)
    # 获取特征矩阵 X 的样本数量
    num_samples = len(X)
    # 生成从 0 到 num_samples - 1 的整数索引数组
    indices = np.arange(num_samples)
    # 对索引数组进行随机打乱顺序操作
    np.random.shuffle(indices)
    # 计算测试集的样本数量，通过总样本数乘以测试集比例得到，并转换为整数
    test_num = int(num_samples * test_size)
    # 从打乱后的索引数组中选取前 test_num 个索引作为测试集的索引
    test_indices = indices[:test_num]
    # 从打乱后的索引数组中选取 test_num 之后的索引作为训练集的索引
    train_indices = indices[test_num:]
    # 根据训练集的索引从特征矩阵 X 中选取对应的样本，得到训练集的特征矩阵
    X_train = X[train_indices]
    # 根据测试集的索引从特征矩阵 X 中选取对应的样本，得到测试集的特征矩阵
    X_test = X[test_indices]
    # 根据训练集的索引从标签向量 y 中选取对应的标签，得到训练集的标签向量
    y_train = y[train_indices]
    # 根据测试集的索引从标签向量 y 中选取对应的标签，得到测试集的标签向量
    y_test = y[test_indices]
    # 返回训练集和测试集的特征矩阵以及标签向量
    return X_train, X_test, y_train, y_test

# 读取文件
excel_file = pd.ExcelFile('data-EXP1--Real-vs-Pred-(rand)-5-6-7.xlsx')
# 选取数据集 data-Label-feature(0.5)
df = excel_file.parse('data-Label-feature(0.5)')

# 准备特征矩阵 X 和标签向量 y
X = df.drop(columns=['RealLabel']).values
y = df['RealLabel'].values

test_size = 0.1
num_splits = 5

results = []
for i in range(num_splits):
    X_train, X_test, y_train, y_test = hold_out_split(X, y, test_size=test_size, random_state=i)

    # 训练集和测试集的样本数量及比例
    train_count = len(y_train)
    test_count = len(y_test)
    train_proportion = train_count / (train_count + test_count)
    test_proportion = test_count / (train_count + test_count)

    # 训练集中不同类别的样本数量及比例
    train_class_count = pd.Series(y_train).value_counts()
    train_class_proportion = train_class_count / train_count

    # 测试集中不同类别的样本数量及比例
    test_class_count = pd.Series(y_test).value_counts()
    test_class_proportion = test_class_count / test_count

    result = {
        '拆分次数': i + 1,
        '训练集样本数量': train_count,
        '测试集样本数量': test_count,
        '训练集样本比例': train_proportion,
        '测试集样本比例': test_proportion,
        '训练集类别 0 数量': train_class_count.get(0, 0),
        '训练集类别 0 比例': train_class_proportion.get(0, 0),
        '训练集类别 1 数量': train_class_count.get(1, 0),
        '训练集类别 1 比例': train_class_proportion.get(1, 0),
        '测试集类别 0 数量': test_class_count.get(0, 0),
        '测试集类别 0 比例': test_class_proportion.get(0, 0),
        '测试集类别 1 数量': test_class_count.get(1, 0),
        '测试集类别 1 比例': test_class_proportion.get(1, 0)
    }
    results.append(result)

# 调整pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 输出表格结果
results_df = pd.DataFrame(results)
print(results_df)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
# 调整字体大小
plt.rcParams['font.size'] = 5

# 绘制训练集和测试集样本数量变化图
plt.figure(figsize=(4, 2))
plt.subplot(1, 2, 1)
plt.plot(results_df['拆分次数'], results_df['测试集类别 0 数量'], marker='o', label='测试集类别 0 数量')
plt.plot(results_df['拆分次数'], results_df['测试集类别 1 数量'], marker='s', label='测试集类别 1 数量')
plt.xlabel('拆分次数')
plt.ylabel('样本数量')
plt.title('测试集不同类别样本数量变化')
# 调整图例位置
plt.legend(loc='upper right', fontsize=6)

# 绘制训练集和测试集中不同类别的样本比例变化图
plt.subplot(1, 2, 2)
plt.plot(results_df['拆分次数'], results_df['测试集类别 0 比例'], marker='o', label='测试集类别 0 比例')
plt.plot(results_df['拆分次数'], results_df['测试集类别 1 比例'], marker='s', label='测试集类别 1 比例')
plt.xlabel('拆分次数')
plt.ylabel('样本比例')
plt.title('测试集不同类别样本比例变化')
# 调整图例位置
plt.legend(loc='upper right', fontsize=6)

# 手动调整子图间距
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.show()