# import pandas as pd
# import matplotlib.pyplot as plt

# # 输入你的 Excel 文件路径
# file_path = '/home/jinghao/projects/dental_plague_detection/dataset/dataset_children_statistic.xlsx'  # 例如：C:/Users/你的用户名/桌面/data.xlsx

# # 读取 sheet
# df = pd.read_excel(file_path, sheet_name="basic information")

# # 获取第二列（按位置）
# age_months = df.iloc[:, 1]

# # 按月份统计人数
# age_counts = age_months.value_counts().sort_index()

# # 补全所有月份（避免断点），假设月龄从最小到最大
# all_months = range(int(age_months.min()), int(age_months.max()) + 1)
# age_counts_full = age_counts.reindex(all_months, fill_value=0)

# # 绘图
# plt.figure(figsize=(10, 5))
# plt.fill_between(age_counts_full.index, age_counts_full.values, step='mid', alpha=0.7)
# plt.xlabel('Month')
# plt.ylabel('Number of patients')
# plt.title('Age Distribution by Month')
# plt.tight_layout()

# # 保存图片
# plt.savefig("age_distribution.png")
# plt.close()
# print("图片已保存为 age_distribution.png")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams
# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

# 读取 Excel 文件
file_path = '/home/jinghao/projects/dental_plague_detection/dataset/dataset_children_statistic.xlsx'  # 替换为你的文件路径
sheet_name = "basic information"   # 目标 sheet 名称

# 读取数据
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 假设第二列是年龄统计，提取数据
month_data = df.iloc[:, 1].dropna()  # 第二列并移除空值

# 转换为数字类型（假设月份为整数）
months = pd.to_numeric(month_data, errors="coerce").dropna()
months = months[months <= 100]  # 过滤月份大于 100

# 找到月份的最大值和最小值
min_month, max_month = int(months.min()), int(months.max())

# 按月份统计数量
month_counts = months.value_counts().sort_index()  # 按月份排序
x = month_counts.index  # 月份
y = month_counts.values  # 每个月的数量

# # 创建横轴（月数）
# months = np.arange(1, len(ages_by_month) + 1)

# 使用插值让图表平滑
# x_smooth = np.linspace(min_month, max_month, 300)
# spl = make_interp_spline(x, y, k=5)  # Cubic spline interpolation
# y_smooth = spl(x_smooth)

degree = 25  # 可调整为更高或更低的值
polynomial_coefficients = np.polyfit(x, y, degree)  # 拟合多项式系数
polynomial_function = np.poly1d(polynomial_coefficients)  # 生成多项式函数

# 创建平滑的横轴
x_smooth = np.linspace(min_month, max_month, 300)  # 生成平滑的点
y_smooth = polynomial_function(x_smooth)  # 使用拟合函数计算平滑的 Y 值
y_smooth_clipped = np.clip(y_smooth, 0, None)  # 小于0的值设为0，大于0的保持不变

# print(y_smooth)

# 绘制平滑的柱状图
plt.figure(figsize=(10, 6))
# plt.bar(x, y, alpha=0.5, label="raw", color="skyblue", edgecolor="black")
# plt.plot(x_smooth, y_smooth, label="smooth", color="red", linewidth=2)
plt.fill_between(x_smooth, y_smooth_clipped, color="#2b95c7", alpha=1)  # 填充颜色

# 设置 X 轴间隔为 5
plt.xticks(np.arange(min_month+1, max_month+1, 5), fontsize=12)  # X 轴刻度间隔为 5 ; fontweight="bold"
plt.yticks(fontsize=12)  # 加粗 Y 轴刻度

# 添加标题和标签
# plt.title("年龄统计（按月份）", fontsize=16)
plt.xlabel('Month', fontweight='bold', fontsize=12)
plt.ylabel('Number of patients', fontweight='bold', fontsize=12)
# plt.legend()
plt.grid(alpha=0.3)

# 保存图表
output_file = "age_distribution.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"图表已保存为 {output_file}")