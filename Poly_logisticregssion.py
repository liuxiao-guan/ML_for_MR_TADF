import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 从Excel文件读取数据
file_path = 'machine_learning.xlsx'  # 修改为你的Excel文件路径
sheet_name = 'Sheet1'  # 修改为你的Sheet名称

df = pd.read_excel(file_path, sheet_name=sheet_name)

# 确认数据列名与实际一致
X = df[['LUMO', 'HOMO','Dihedral_Angle']].values
y = (df['FWHM-TOL'] > 40).astype(int).values  # 将y转换为二进制变量，y > 30 为1，否则为0
split_index = 27
X = X[:split_index]
y = y[:split_index]


# 创建包含多项式特征的逻辑回归模型
degree = 2 # 多项式特征的度数，可以调整
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# 拟合逻辑回归模型
logistic_model = LogisticRegression()
logistic_model.fit(X_poly, y)
accuracy = logistic_model.score(X_poly, y)
print("Model Accuracy:", accuracy)

# 获取回归系数和截距
intercept = logistic_model.intercept_[0]
coefficients = logistic_model.coef_[0]
feature_names = poly.get_feature_names_out(['LUMO', 'HOMO','Dihedral_Angle'])

# 输出逻辑回归方程
equation = f"logit(p) = {intercept:.4f}"
for coef, name in zip(coefficients, feature_names):
    equation += f" + ({coef:.4f})*{name}"

print("Logistic Regression Equation with Polynomial Features:")
print(equation)

# 绘制决策边界的三维图像
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成网格点
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1

xx1, xx2, xx3 = np.meshgrid(np.linspace(x1_min, x1_max, 40),
                            np.linspace(x2_min, x2_max, 40),
                            np.linspace(x3_min, x3_max, 40))

XX = np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()]
XX_poly = poly.transform(XX)

# 预测网格点的值
Z = logistic_model.predict(XX_poly).reshape(xx1.shape)

# 创建一个新的 3D 图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 定义自定义颜色列表（根据 Z 的值进行映射）
from matplotlib.colors import ListedColormap
#custom_colors = [(0.213,0.500,0.597), (0.341,0.075,0.114)]  # 自定义颜色
#custom_colors =['green', 'orange']
# custom_colors = [
#     (0, 1, 0, 0.3), # 绿色 (RGBA, 0.5 为透明度)
#     (1, 0.65, 0,0.3) # 橙色 (RGB)
    
# ]
custom_colors = [
    
    (1, 0.65, 0,0.2), # 橙色 (RGB)
    (0, 0, 0.5, 0.7), # 绿色 (RGBA, 0.5 为透明度)
    
]
cmap = ListedColormap(custom_colors)



# 创建散点图
scatter = ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c=Z.ravel(), cmap=cmap, s=20, edgecolors=(1, 0, 0,0), label='Predicted Points')

# 绘制边框
# 示例数据：四个点的坐标
A = np.array([-2.86, -8.26, 88.3])  # 点A
B = np.array([0.02, -8.26, 88.3])  # 点B
C = np.array([0.02, -3.06, 88.3])  # 点C
D = np.array([0.02, -3.06, 14.7])  # 点D
E = np.array([0.02, -8.26, 14.7])
F = np.array([-2.86, -8.26, 14.7])  # 点F
h1 = np.array([-2.86, -7.2, 88.3]) 

# 连接点：A到B，B到C，C到D，简化为一次性传入所有点的坐标
points = np.array([A, B, C])  # 将所有点放入一个数组
ax.plot(points[:, 0], points[:, 1], points[:, 2], color= (0.1, 0.1, 0.1,1), linewidth=4)  # 直接连接所有点

points3 = np.array([ C, D, E, F, A])  # 将所有点放入一个数组
ax.plot(points3[:, 0], points3[:, 1], points3[:, 2], color= (0.5, 0.5, 0.5,0.5), linewidth=4)  # 直接连接所有点

points1 = np.array([B, E])  # 将所有点放入一个数组
ax.plot(points1[:, 0], points1[:, 1], points1[:, 2], color= (0.1, 0.1, 0.1,1), linewidth=4)  # 直接连接所有点

h2 = np.array([-0.8, -3.06, 88.3])
points2 = np.array([C, h2,h1,A])  # 将所有点放入一个数组
ax.plot(points2[:, 0], points2[:, 1], points2[:, 2], color= (0.5, 0.5, 0.5,0.5), linewidth=4)  # 直接连接所有点


# h1 = np.array([-2.86, -7.2, 88.3]) 
# h2 = np.array([-0.8, -3.06, 88.3])
h3 = np.array([-2.86, -6.0, 14.7])
h4 = np.array([-1.5, -3.06, 14.7])
points4 = np.array([h1, h3,h4,h2])  # 将所有点放入一个数组
ax.plot(points4[:, 0], points4[:, 1], points4[:, 2], color= (0, 0, 0.5, 0.3), linewidth=4)  # 直接连接所有点

# 绘制原始数据点
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k', marker='^', label='Original Data Points')
# 设置刻度标签的字体
for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
    label.set_fontname('Times New Roman')
    label.set_fontsize(18)
ax.set_xlabel('LUMO',fontname='Times New Roman', fontsize=26,fontweight='bold',labelpad=15)
ax.set_ylabel('HOMO',fontname='Times New Roman', fontsize=26,fontweight='bold',labelpad=24)
ax.set_zlabel('Dihedral Angle(°)',fontname='Times New Roman', fontsize=26,fontweight='bold',labelpad=10)
ax.set_title('Plot',fontname='Times New Roman', fontsize=20)
# 设置坐标轴线宽
ax.w_xaxis.line.set_linewidth(3)
ax.w_yaxis.line.set_linewidth(3)
ax.w_zaxis.line.set_linewidth(3)
# 设置y轴显示浮点数格式
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
# 设置图例字体
# legend = ax.legend(prop={'family': 'Times New Roman', 'size': 50})
cbar = plt.colorbar(scatter, ax=ax,ticks=[0, 1],pad=0.07)
cbar.ax.set_yticklabels(['0', '1'], fontname='Times New Roman', fontsize=18)
# 设置颜色条与图之间的距离
#cbar.ax.margins(x=50.0, y=50.0)  # 调整x和y的值以增加或减少颜色条与图之间的距离
plt.rcParams.update({'font.size': 18})     #设置图例字体大小
plt.legend(loc='upper right',frameon=False)   
# plt.legend()
plt.show()

plt.savefig("Poly_Decision_Boundary_xyz_b.png")

# 计算y > 30 的概率
def predict_probability(model, LUMO, HOMO,Dihedral_Angle):
    x_poly = poly.transform(np.array([[LUMO, HOMO,Dihedral_Angle]]))
    return model.predict_proba(x_poly)[0, 1]

output  = predict_probability(logistic_model,-1.08,-4.53,67.4)
print(output < 0.5)
output  = predict_probability(logistic_model,-1.08,-4.64,70.0)
print(output < 0.5)
output  = predict_probability(logistic_model,-1.08,-4.68,87.2)
print(output < 0.5)
output  = predict_probability(logistic_model,-1.24,-4.68,85.5)
print(output > 0.5)
output  = predict_probability(logistic_model,-1.41,-4.68,85.5)
print(output > 0.5)
output  = predict_probability(logistic_model,-1.57,-4.48,85.5)
print(output > 0.5)


