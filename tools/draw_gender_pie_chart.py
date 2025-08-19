import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['352', '296']  # 标签
sizes = [352, 296]       # 对应的比例数据
colors = ['#f0788c', '#65d2e6'] # 每部分的颜色
explode = [0.04, 0.04]     # 控制每部分的分离程度

# 创建饼图
fig, ax = plt.subplots()
wedges, texts = ax.pie(
    sizes,
    explode=explode,       # 设置缝隙
    # labels=labels,         # 显示数据
    colors=colors,         # 设置颜色
    autopct=None,          # 不显示默认百分比
    startangle=90,         # 设置起始角度
    wedgeprops={'edgecolor': 'white'}  # 设置边框颜色
)

# 添加中心数字标签
for i, wedge in enumerate(wedges):
    # 获取每个 wedge 的中心角度
    theta = (wedge.theta1 + wedge.theta2) / 2
    x = 0.7 * np.cos(np.radians(theta))  # 使用 numpy 的 cos 计算 x 坐标
    y = 0.7 * np.sin(np.radians(theta))  # 使用 numpy 的 sin 计算 y 坐标
    # ax.text(
    #     x, y,
    #     labels[i],          # 显示对应数字
    #     ha='center',
    #     va='center',
    #     fontsize=16,
    #     color='white',
    #     fontweight='bold'
    # )

# 设置图形为正圆形
ax.axis('equal')

# 保存图片到文件
plt.savefig('fig_gender_pie_chart_with_gap.png', dpi=300, bbox_inches='tight', transparent=True)

# 显示图像
# plt.show()
