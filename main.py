import pandas as pd
import numpy as np

# 1. 读取数据
df = pd.read_csv('ICData.csv')

# --- 关键步骤：去除列名首尾的空格 ---
# 这一步很重要，防止 ' 交易时间' 这种隐藏空格导致报错
df.columns = df.columns.str.strip()

# --- 调试：打印列名 ---
print("检测到的列名：", df.columns.tolist())

# 2. 转换时间格式并提取小时
# 对应中文列名：'交易时间'
df['交易时间'] = pd.to_datetime(df['交易时间'])
# 提取小时数存入新列 hour
df['hour'] = df['交易时间'].dt.hour

# 3. 计算搭乘站点数 (ride_stops)
# 对应中文列名：'下车站点' - '上车站点'
df['ride_stops'] = df['下车站点'] - df['上车站点']

# 4. 数据清洗
# 删除 ride_stops 小于等于 0 的异常行
df = df[df['ride_stops'] > 0]

# 5. 检查缺失值
print("各列缺失值统计：")
print(df.isnull().sum())

# 打印处理后的数据总行数
print(f"处理后的数据总行数: {len(df)}")

# 显示前5行数据
print(df.head())

# ==================== 任务 2：时间分布分析 ====================

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# (a) 早晚时段刷卡量统计 (必须使用 numpy)

# 1. 定义筛选条件：仅统计上车刷卡 (刷卡类型=0)
# 注意：这里假设清洗后的数据中已经包含了'刷卡类型'列，且0代表上车
up_boarding = df['刷卡类型'] == 0

# 2. 使用 numpy 布尔索引统计早峰前时段 (< 7点) 的刷卡量
# np.sum 配合布尔条件，True 计为1，False 计为0
early_morning_count = np.sum((df['hour'] < 7) & up_boarding)

# 3. 使用 numpy 布尔索引统计深夜时段 (>= 22点) 的刷卡量
late_night_count = np.sum((df['hour'] >= 22) & up_boarding)

# 4. 计算全天总上车刷卡量 (用于计算百分比)
total_count = np.sum(up_boarding)

# 5. 计算百分比
early_percent = (early_morning_count / total_count) * 100
late_percent = (late_night_count / total_count) * 100

# 打印结果
print(f"\n--- 任务 2 (a) 结果 ---")
print(f"早峰前时段 (<07:00) 刷卡量: {early_morning_count} 次, 占比: {early_percent:.2f}%")
print(f"深夜时段 (>=22:00) 刷卡量: {late_night_count} 次, 占比: {late_percent:.2f}%")

# (b) 24小时刷卡量分布可视化

# 1. 准备数据：统计每个小时的刷卡量
# 使用 value_counts 并按小时排序 (0-23)
hourly_counts = df[df['刷卡类型'] == 0]['hour'].value_counts().sort_index()

# 2. 创建画布和坐标轴
plt.figure(figsize=(12, 6))

# 3. 绘制柱状图
# 我们需要将早7点前和晚22点后的柱子标成不同颜色
bars = plt.bar(hourly_counts.index, hourly_counts.values,
               color=['skyblue' if (h >= 7 and h < 22) else 'salmon' for h in hourly_counts.index])

# 4. 设置图表样式
plt.title('24小时公交刷卡量分布图', fontsize=16, fontweight='bold')
plt.xlabel('小时 (Hour)', fontsize=12)
plt.ylabel('刷卡量 (次)', fontsize=12)

# 5. 设置 x 轴刻度：步长为 2 (0, 2, 4...)
plt.xticks(range(0, 24, 2))

# 6. 添加图例 (说明颜色含义)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='skyblue', label='正常运营时段 (07:00-21:59)'),
                   Patch(facecolor='salmon', label='早峰前/深夜时段 (00:00-06:59, 22:00-23:59)')]
plt.legend(handles=legend_elements)

# 7. 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 8. 优化布局并保存
plt.tight_layout()

# 保存图像 (dpi=150)
plt.savefig('hour_distribution.png', dpi=150)
print("\n--- 任务 2 (b) 完成 ---")
print("已生成图表: hour_distribution.png")