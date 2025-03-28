import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 150
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 计算多个评估指标的函数
def evaluate_metrics(y_true, y_pred):
    """
    计算一系列分类评估指标。
    参数:
    y_true (array-like): 真实的标签数组，其中元素通常为 0 或 1，表示负类和正类。
    y_pred (array-like): 预测的标签数组，元素同样通常为 0 或 1。
    返回:
    error_rate (float): 错误率，即分类错误的样本数占总样本数的比例。
    accuracy (float): 准确率，即分类正确的样本数占总样本数的比例。
    precision (float): 精确度，在所有被预测为正类的样本中，实际为正类的比例。
    recall (float): 召回率，在所有实际为正类的样本中，被正确预测为正类的比例。
    f1 (float): F1 值，综合考虑精确度和召回率的指标，是二者的调和平均值。
    tpr (float): 真阳性率，等同于召回率，即实际为正类的样本中被正确预测为正类的比例。
    fpr (float): 假阳性率，在所有实际为负类的样本中，被错误预测为正类的比例。
    """
    # 计算真正例（True Positive）的数量：真实标签为 1 且预测标签也为 1 的样本数量
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # 计算真负例（True Negative）的数量：真实标签为 0 且预测标签也为 0 的样本数量
    tn = np.sum((y_true == 0) & (y_pred == 0))
    # 计算假正例（False Positive）的数量：真实标签为 0 但预测标签为 1 的样本数量
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # 计算假负例（False Negative）的数量：真实标签为 1 但预测标签为 0 的样本数量
    fn = np.sum((y_true == 1) & (y_pred == 0))
    # 计算错误率，如果分母不为 0，则用（假正例数量 + 假负例数量）除以总样本数，否则为 0
    error_rate = (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    # 计算准确率，如果分母不为 0，则用（真正例数量 + 真负例数量）除以总样本数，否则为 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    # 计算精确度，如果分母不为 0，则用真正例数量除以（真正例数量 + 假正例数量），否则为 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    # 计算召回率，如果分母不为 0，则用真正例数量除以（真正例数量 + 假负例数量），否则为 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    # 计算 F1 值，如果分母不为 0，则用 2 乘以（精确度 * 召回率）除以（精确度 + 召回率），否则为 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    # 真阳性率等于召回率
    tpr = recall
    # 计算假阳性率，如果分母不为 0，则用假正例数量除以（假正例数量 + 真负例数量），否则为 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    return error_rate, accuracy, precision, recall, f1, tpr, fpr


# 计算 ROC 曲线及 AUPROC 的函数
def roc_metrics(y_true, y_score):
    """
    计算 ROC（Receiver Operating Characteristic）曲线及其曲线下面积（AUPROC，Area Under the ROC Curve）。
    参数:
    y_true (array-like): 真实的标签数组，其中元素通常为 0 或 1，表示负类和正类。
    y_score (array-like): 预测的得分数组，通常是模型输出的概率值或得分值，用于确定预测标签。
    返回:
    fpr_list (numpy.ndarray): 假阳性率数组，对应 ROC 曲线上的横坐标。
    tpr_list (numpy.ndarray): 真阳性率数组，对应 ROC 曲线上的纵坐标。
    roc_auc (float): ROC 曲线下的面积。
    """
    # 获取预测得分中的唯一值，并按降序排序，这些唯一值将作为阈值来计算不同情况下的指标
    thresholds = np.sort(np.unique(y_score))[::-1]
    fpr_list = []
    tpr_list = []
    # 遍历每个阈值
    for thresh in thresholds:
        # 根据当前阈值将预测得分转换为预测标签，大于等于阈值的为 1（正类），小于阈值的为 0（负类）
        y_pred = (y_score >= thresh).astype(int)
        # 使用 evaluate_metrics 函数计算当前阈值下的评估指标，这里只关注真阳性率和假阳性率
        _, _, _, _, _, tpr, fpr = evaluate_metrics(y_true, y_pred)
        # 将当前阈值下的假阳性率添加到 fpr_list 中
        fpr_list.append(fpr)
        # 将当前阈值下的真阳性率添加到 tpr_list 中
        tpr_list.append(tpr)
    fpr_list = np.array(fpr_list)
    tpr_list = np.array(tpr_list)
    roc_auc = 0
    # 使用梯形法则计算 ROC 曲线下的面积。遍历假阳性率数组（除最后一个元素）
    for i in range(len(fpr_list) - 1):
        # 计算梯形的宽度，即相邻两个假阳性率的差值
        width = fpr_list[i + 1] - fpr_list[i]
        # 计算梯形的高度，即相邻两个真阳性率的平均值
        height = (tpr_list[i] + tpr_list[i + 1]) / 2
        # 将当前梯形的面积累加到 roc_auc 中
        roc_auc += width * height
    return fpr_list, tpr_list, roc_auc


# 计算 PR 曲线及 AUPRC 的函数
def pr_metrics(y_true, y_score):
    """
    计算 Precision-Recall（PR）曲线及其曲线下面积（AUPRC，Area Under the PR Curve）。
    参数:
    y_true (array-like): 真实的标签数组，其中元素通常为 0 或 1，表示负类和正类。
    y_score (array-like): 预测的得分数组，通常是模型输出的概率值或得分值，用于确定预测标签。
    返回:
    precision_list (numpy.ndarray): 精确度数组，对应 PR 曲线上的纵坐标。
    recall_list (numpy.ndarray): 召回率数组，对应 PR 曲线上的横坐标。
    pr_auc (float): PR 曲线下的面积。
    """
    # 获取预测得分中的唯一值，并按降序排序，这些唯一值将作为阈值来计算不同情况下的指标
    thresholds = np.sort(np.unique(y_score))[::-1]
    precision_list = []
    recall_list = []
    # 遍历每个阈值
    for thresh in thresholds:
        # 根据当前阈值将预测得分转换为预测标签，大于等于阈值的为 1（正类），小于阈值的为 0（负类）
        y_pred = (y_score >= thresh).astype(int)
        # 使用 evaluate_metrics 函数计算当前阈值下的评估指标，这里只关注精确度和召回率
        _, _, precision, recall, _, _, _ = evaluate_metrics(y_true, y_pred)
        # 将当前阈值下的精确度添加到 precision_list 中
        precision_list.append(precision)
        # 将当前阈值下的召回率添加到 recall_list 中
        recall_list.append(recall)
    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    pr_auc = 0
    # 使用梯形法则计算 PR 曲线下的面积。遍历召回率数组（除最后一个元素）
    for i in range(len(recall_list) - 1):
        # 计算梯形的宽度，即相邻两个召回率的差值
        width = recall_list[i + 1] - recall_list[i]
        # 计算梯形的高度，即相邻两个精确度的平均值
        height = (precision_list[i] + precision_list[i + 1]) / 2
        # 将当前梯形的面积累加到 pr_auc 中
        pr_auc += width * height
    return precision_list, recall_list, pr_auc


# 读取文件
excel_file = pd.ExcelFile('data-EXP1--Real-vs-Pred-(rand)-5-6-7.xlsx')

# 获取所有表名
sheet_names = excel_file.sheet_names

# 定义存储结果的列表
results = []

# 遍历不同工作表
for sheet_name in sheet_names:
    # 获取当前工作表的数据
    df = excel_file.parse(sheet_name)

    # 提取真实标签和预测标签
    y_true = df['RealLabel']
    y_pred = df['PredictedLabel']
    y_score = df['PredictedScore']

    # 自定义函数计算指标
    error_rate, accuracy, precision, recall, f1, tpr, fpr = evaluate_metrics(y_true, y_pred)
    fpr_custom, tpr_custom, roc_auc_custom = roc_metrics(y_true, y_score)
    precision_custom, recall_custom, pr_auc_custom = pr_metrics(y_true, y_score)

    # 库函数计算指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    error_rate_sys = (fp + fn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    accuracy_sys = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
    precision_sys = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall_sys = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_sys = 2 * (precision_sys * recall_sys) / (precision_sys + recall_sys) if (precision_sys + recall_sys) != 0 else 0
    tpr_sys = recall_sys
    fpr_sys = fp / (fp + tn) if (fp + tn) != 0 else 0

    fpr_sys_curve, tpr_sys_curve, _ = roc_curve(y_true, y_score)
    roc_auc_sys = auc(fpr_sys_curve, tpr_sys_curve)

    precision_sys_pr, recall_sys_pr, _ = precision_recall_curve(y_true, y_score)
    pr_auc_sys = auc(recall_sys_pr, precision_sys_pr)

    # 汇总自定义函数结果
    results_custom = [sheet_name + '_自定义函数', error_rate, accuracy, precision, recall, f1, tpr, fpr, roc_auc_custom,
                      pr_auc_custom]
    # 汇总库函数结果
    results_sys = [sheet_name + '_系统自带函数', error_rate_sys, accuracy_sys, precision_sys, recall_sys, f1_sys, tpr_sys,
                   fpr_sys, roc_auc_sys, pr_auc_sys]

    # 将结果添加到总结果列表中
    results.extend([results_custom, results_sys])

    # 绘制 ROC 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_custom, tpr_custom, label=f'自定义函数 (AUC = {roc_auc_custom:.2f})')
    plt.plot(fpr_sys_curve, tpr_sys_curve, label=f'系统自带函数 (AUC = {roc_auc_sys:.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{sheet_name} 的 ROC 曲线')
    plt.legend(loc="lower right")

    # 绘制 PR 曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall_custom, precision_custom, label=f'自定义函数 (AUC = {pr_auc_custom:.2f})')
    plt.plot(recall_sys_pr, precision_sys_pr, label=f'系统自带函数 (AUC = {pr_auc_sys:.2f})', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确度')
    plt.title(f'{sheet_name} 的 PR 曲线')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()

# 调整 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# 创建 DataFrame 展示结果
results_df = pd.DataFrame(results,
                          columns=['测试', '错误率', '准确率', '精确度', '召回率', 'F1', '真阳性率', '假阳性率', 'AUPROC',
                                   'AUPRC'])
print(results_df)