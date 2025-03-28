import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def custom_split(X, y, num_folds, random_state=None):
    """
    将数据集拆分成指定份数的子集合
    参数:
    X (array-like): 特征矩阵，形状为 (样本数量, 特征数量)，每一行代表一个样本，每一列代表一个特征
    y (array-like): 标签向量，形状为 (样本数量,)，对应特征矩阵中每个样本的真实标签
    num_folds (int): 要将数据集拆分成的份数，例如 3 或 5
    random_state (int, 可选): 随机种子，用于保证结果可重复。当传入一个整数时，每次调用该函数在相同参数下将得到相同的拆分结果。默认值为 None
    返回:
    subsets (list): 包含多个元组的列表，每个元组中第一个元素为子集合的特征矩阵 (array-like)，
                    第二个元素为子集合的标签向量 (array-like)
    """
    # 如果传入了随机种子，则设置 numpy 的随机数生成器的种子，保证结果可重复
    if random_state is not None:
        np.random.seed(random_state)
    # 获取特征矩阵 X 的样本数量
    num_samples = len(X)
    # 生成从 0 到 num_samples - 1 的整数索引数组，用于表示数据集中每个样本的索引
    indices = np.arange(num_samples)
    # 对索引数组进行随机打乱顺序操作，使得每个样本都有相同的概率被分配到不同的子集合中
    np.random.shuffle(indices)
    # 计算每个子集合的大致样本数量，通过总样本数除以拆分份数得到（向下取整）
    fold_size = num_samples // num_folds
    # 初始化一个空列表，用于存储拆分后的各个子集合
    subsets = []
    for i in range(num_folds):
        # 计算当前子集合在索引数组中的起始位置
        start = i * fold_size
        # 计算当前子集合在索引数组中的结束位置。如果不是最后一个子集合，
        # 则结束位置为 (i + 1) * fold_size；如果是最后一个子集合，则结束位置为总样本数量
        end = (i + 1) * fold_size if i < num_folds - 1 else num_samples
        # 从打乱后的索引数组中选取当前子集合对应的索引
        subset_indices = indices[start:end]
        # 根据选取的索引，从特征矩阵 X 中提取出当前子集合的特征矩阵
        X_subset = X[subset_indices]
        # 根据选取的索引，从标签向量 y 中提取出当前子集合的标签向量
        y_subset = y[subset_indices]
        # 将当前子集合的特征矩阵和标签向量组成一个元组，添加到 subsets 列表中
        subsets.append((X_subset, y_subset))
    # 返回包含所有子集合的列表
    return subsets

# 读取文件
excel_file = pd.ExcelFile('data-EXP1--Real-vs-Pred-(rand)-5-6-7.xlsx')
# 选取数据集 data-Label-feature(0.5)
df = excel_file.parse('data-Label-feature(0.9)')

# 准备特征矩阵 X 和标签向量 y
X = df.drop(columns=['RealLabel']).values
y = df['RealLabel'].values

num_splits = 5
num_folds = 3

all_results = []
for split in range(num_splits):
    subsets = custom_split(X, y, num_folds, random_state=split)
    for fold, (_, y_subset) in enumerate(subsets):
        subset_count = len(y_subset)
        class_count = pd.Series(y_subset).value_counts()
        class_proportion = class_count / subset_count

        result = {
            '拆分次数': split + 1,
            '子集合编号': fold + 1,
            '子集合样本数量': subset_count,
            '类别 0 数量': class_count.get(0, 0),
            '类别 0 比例': class_proportion.get(0, 0),
            '类别 1 数量': class_count.get(1, 0),
            '类别 1 比例': class_proportion.get(1, 0)
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

# 绘制每个子集合中不同类别的样本数量变化图
plt.figure(figsize=(4, 2))
for fold in range(1, num_folds + 1):
    fold_df = results_df[results_df['子集合编号'] == fold]
    plt.plot(fold_df['拆分次数'], fold_df['类别 0 数量'], marker='o', label=f'子集合 {fold} 类别 0 数量')
    plt.plot(fold_df['拆分次数'], fold_df['类别 1 数量'], marker='s', label=f'子集合 {fold} 类别 1 数量')
plt.xlabel('拆分次数')
plt.ylabel('样本数量')
plt.title('每个子集合中不同类别的样本数量变化')
# 调整图例位置
plt.legend(loc='upper right', fontsize=6)

plt.show()