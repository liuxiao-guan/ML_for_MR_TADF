import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import log, sqrt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

# 从Excel文件读取数据
file_path = '/root/autodl-tmp/guangdian/machine learning_x1.xlsx'  # 修改为你的Excel文件路径
sheet_name = 'Sheet1'  # 修改为你的Sheet名称

df = pd.read_excel(file_path, sheet_name=sheet_name)

# 确认数据列名与实际一致
X_origin = df[['LUMO', 'HOMO','Dihedral_Angle']].values
y_origin = (df['FWHM-TOL'] > 40).astype(int).values  # 将y转换为二进制变量，y > 30 为1，否则为0
split_index = 27
X = X_origin[:split_index]
y = y_origin[:split_index]
X_test = X_origin[split_index:31]
y_test = y_origin[split_index:31]


# 创建包含多项式特征的逻辑回归模型
degree = 2 # 多项式特征的度数，可以调整
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)  # 测试集应用训练集的转换

# 拟合逻辑回归模型
logistic_model = LogisticRegression()
logistic_model.fit(X_poly, y)

# 使用模型预测测试集结果
y_pred = logistic_model.predict(X_test_poly)
y_pred_proba = logistic_model.predict_proba(X_test_poly)[:, 1]  # 预测概率
# accuracy = logistic_model.score(X_poly, y)
# print("Model Accuracy:", accuracy)


# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)

# 输出结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC-ROC: {roc_auc:.4f}')
print(f'Log-Loss: {logloss:.4f}')

# 回归评价指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmsle = sqrt(np.mean((np.log1p(y_test) - np.log1p(y_pred)) ** 2))
r2 = r2_score(y_test, y_pred)

# 计算 Adjusted R-squared
n = len(y_test)  # 样本数量
p = X_test_poly.shape[1] - 1  # 自变量的数量
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# 输出回归评价指标
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'Root Mean Squared Logarithmic Error (RMSLE): {rmsle:.4f}')
print(f'R-squared (R²): {r2:.4f}')
print(f'Adjusted R-squared: {r2_adj:.4f}')



# 计算 Silhouette Coefficient
silhouette = silhouette_score(X_test_poly, y_pred)
print(f'Silhouette Coefficient: {silhouette:.4f}')

# 计算 Dunn Index
# 计算每一对聚类的距离
def dunn_index(X, labels):
    distances = cdist(X, X, 'euclidean')  # 计算欧几里得距离
    min_inter_cluster_dist = float('inf')
    max_intra_cluster_dist = float('-inf')

    # 遍历每个类别
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        # 计算每个聚类内部的最大距离
        intra_cluster_dist = np.max(cdist(cluster_points, cluster_points, 'euclidean'))
        max_intra_cluster_dist = max(max_intra_cluster_dist, intra_cluster_dist)
        
        # 计算每个聚类与其他聚类之间的最小距离
        for other_label in np.unique(labels):
            if label != other_label:
                other_cluster_points = X[labels == other_label]
                inter_cluster_dist = np.min(cdist(cluster_points, other_cluster_points, 'euclidean'))
                min_inter_cluster_dist = min(min_inter_cluster_dist, inter_cluster_dist)

    return min_inter_cluster_dist / max_intra_cluster_dist

# 计算 Dunn Index
dunn = dunn_index(X_test_poly, y_pred)
print(f'Dunn Index: {dunn:.4f}')



