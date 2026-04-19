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

# ==================== 任务 3：站点客流统计 ====================

# 1. 筛选上车数据
# 注意：确保'刷卡类型'列是数值型，0代表上车
up_boarding_data = df[df['刷卡类型'] == 0]

# 2. 统计各站点上车人数
# 使用 groupby 按'上车站点'分组，size() 统计每组数量，sort_values 降序排列，head(10) 取前10
top_10_stops = up_boarding_data['上车站点'].value_counts().head(10)

# 打印结果用于检查
print("\n--- 任务 3(a) 结果 ---")
print("客流量前 10 的站点：")
print(top_10_stops)

# 3. 绘制水平条形图
plt.figure(figsize=(10, 6))

# 使用 Matplotlib 绘制水平条形图
# top_10_stops.values 是人数，top_10_stops.index 是站点名
plt.barh(top_10_stops.index, top_10_stops.values, color='skyblue')

# 4. 设置图表样式
plt.xlabel('客流量 (人次)', fontsize=12)
plt.ylabel('上车站点', fontsize=12)
plt.title('客流量前 10 的上车站点统计', fontsize=14, fontweight='bold')

# 5. 在条形图上添加数值标签（可选，但能让图表更专业）
# 遍历数据，在条形末端写上具体数字
for i, v in enumerate(top_10_stops.values):
    plt.text(v + 50, i, str(v), color='black', va='center', fontsize=10)

# 6. 优化布局并保存图片
plt.tight_layout()
plt.savefig('top_stops.png', dpi=150)
print("\n--- 任务 3(b) 完成 ---")
print("已生成图表: top_stops.png")

# ==================== 任务 4：高峰小时系数 (PHF) 计算 ====================

print("\n--- 任务 4：高峰小时系数计算 ---")

# 确保交易时间是 datetime 类型
df['交易时间'] = pd.to_datetime(df['交易时间'])

# 1. 找出高峰小时 (Peak Hour)
df['小时'] = df['交易时间'].dt.hour
hourly_counts = df.groupby('小时').size()

peak_hour = hourly_counts.idxmax()
peak_hour_volume = hourly_counts.max()

print(f"高峰小时为: {peak_hour}:00 - {peak_hour+1}:00, 刷卡量: {peak_hour_volume} 次")

# 2. 筛选高峰小时内的数据
mask = (df['交易时间'].dt.hour == peak_hour)
df_peak = df[mask].copy()

# 设置索引以便重采样
df_peak.set_index('交易时间', inplace=True)

# 3. 5分钟粒度统计 (修改了这里：用 '5min' 代替 '5T')
counts_5min = df_peak.resample('5min').size()
max_5min_volume = counts_5min.max()
max_5min_start = counts_5min.idxmax().strftime('%H:%M')
max_5min_end = (counts_5min.idxmax() + pd.Timedelta(minutes=5)).strftime('%H:%M')

# 计算 PHF5
phf5 = peak_hour_volume / (12 * max_5min_volume)

# 4. 15分钟粒度统计 (修改了这里：用 '15min' 代替 '15T')
counts_15min = df_peak.resample('15min').size()
max_15min_volume = counts_15min.max()
max_15min_start = counts_15min.idxmax().strftime('%H:%M')
max_15min_end = (counts_15min.idxmax() + pd.Timedelta(minutes=15)).strftime('%H:%M')

# 计算 PHF15
phf15 = peak_hour_volume / (4 * max_15min_volume)

# 5. 格式化输出
print("-" * 30)
print(f"高峰小时: {peak_hour}:00 ~ {peak_hour+1}:00, 刷卡量: {peak_hour_volume} 次")
print(f"最大 5 分钟刷卡量 ({max_5min_start}~{max_5min_end}): {max_5min_volume} 次")
print(f"PHF5 = {peak_hour_volume} / (12 × {max_5min_volume}) = {phf5:.4f}")
print(f"最大 15 分钟刷卡量 ({max_15min_start}~{max_15min_end}): {max_15min_volume} 次")
print(f"PHF15 = {peak_hour_volume} / (4 × {max_15min_volume}) = {phf15:.4f}")
print("-" * 30)

# ==================== 任务 5：线路驾驶员信息批量导出 ====================

import os  # 用于操作文件夹和文件路径

print("\n--- 任务 5：线路驾驶员信息批量导出 ---")

# 1. 筛选线路号在 1101 至 1120 之间的记录
# 确保线路号是整数类型（防止它是字符串导致比较出错）
df['线路号'] = df['线路号'].astype(int)

# 筛选范围
target_lines = range(1101, 1121)  # 1101 到 1120
df_filtered = df[df['线路号'].isin(target_lines)].copy()

print(f"共筛选出 {len(df_filtered)} 条属于 1101-1120 线路的记录。")

# 2. 创建文件夹
output_dir = "线路驾驶员信息"
# exist_ok=True 表示如果文件夹已存在则不报错
os.makedirs(output_dir, exist_ok=True)

print(f"正在生成文件到文件夹: {output_dir} ...")

# 3. 遍历每一条线路，生成对应的 txt 文件
for line_id in target_lines:
    # 筛选当前线路的数据
    df_line = df_filtered[df_filtered['线路号'] == line_id]

    # 提取 '车辆编号' 和 '驾驶员编号' 两列，并去重
    # drop_duplicates() 会保留唯一的 (车辆, 驾驶员) 组合
    df_pairs = df_line[['车辆编号', '驾驶员编号']].drop_duplicates()

    # 如果该线路有数据才写入文件
    if not df_pairs.empty:
        # 构造文件路径，例如：线路驾驶员信息/1101.txt
        file_path = os.path.join(output_dir, f"{line_id}.txt")

        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入第一行表头（根据题目要求的格式）
            f.write(f"线路号: {line_id} 车辆编号 驾驶员编号")

            # 遍历每一行数据写入
            # itertuples 比 iterrows 效率更高
            for row in df_pairs.itertuples(index=False):
                # row[0]是车辆编号, row[1]是驾驶员编号
                # 写入格式：换行 + 车辆号 + 空格 + 驾驶员号
                # 题目示例看起来像是有对齐，这里简单用空格分隔，或者用制表符 \t
                f.write(f"\n{row[0]} {row[1]}")

        # 4. 打印生成路径
        print(f"已生成: {file_path}")
    else:
        print(f"警告: 线路 {line_id} 没有找到相关记录。")

print("\n--- 任务 5 完成 ---")

# ==================== 任务 6：服务绩效排名与热力图 ====================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("\n--- 任务 6：服务绩效排名与热力图 ---")

# 1. 统计 Top 10
def get_top_10(data, column, name):
    top_10 = data[column].value_counts().head(10)
    print(f"\nTop 10 {name}:")
    print(top_10)
    return top_10

# 执行统计（确保列名与数据一致）
top_drivers = get_top_10(df, '驾驶员编号', '司机')
top_lines = get_top_10(df, '线路号', '线路')
top_stops = get_top_10(df, '上车站点', '上车站')
top_vehicles = get_top_10(df, '车辆编号', '车辆')

# 2. 构造 4x10 热力图数据
# 核心逻辑：将四个 Series 的数值（values）提取出来，组成一个列表，生成 DataFrame
data_matrix = [
    top_drivers.values,
    top_lines.values,
    top_stops.values,
    top_vehicles.values
]

# 创建 DataFrame
heatmap_data = pd.DataFrame(
    data_matrix,
    index=['司机', '线路', '上车站点', '车辆'],  # 题目要求的行标签
    columns=[f'Top{i+1}' for i in range(10)]     # 题目要求的列标签 Top1-Top10
)

# 3. 绘图与保存
plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_data,
    annot=True,      # 显示数值
    fmt="d",         # 数值格式为整数
    cmap="YlOrRd",   # 题目要求的配色
    linewidths=.5    # 格子间的间隔线
)

# 设置标题（题目要求）
plt.title('服务绩效 Top10 热力图\n(基于乘客人次统计)', fontsize=14)
plt.xlabel('排名', fontsize=12)
plt.ylabel('维度', fontsize=12)
plt.xticks(rotation=0)  # X轴标签旋转0度

# 保存图表（题目要求）
plt.savefig('performance_heatmap.png', dpi=150, bbox_inches='tight')
print("\n图表已保存为 performance_heatmap.png")


# 4. 结论说明（题目要求）
print("""
--- 结论说明 ---
从热力图的颜色深浅分布可以看出明显的“二八定律”现象：
1. 线路流量差异巨大：'线路'这一行的颜色最深且数值断层领先（如 Top1 高达 3600+），说明存在极少数核心骨干线路承载了绝大部分客流，是运营的重中之重。
2. 司机与车辆绩效分化：'司机'和'车辆'维度的 Top1 数值（约 3000 左右）远高于 Top10，说明头部司机和车辆的出勤率或运营效率远超平均水平，存在明显的“金牌员工/车辆”。
3. 站点热度集中：'上车站点'的数值相对均匀但仍有个别热点，说明客流来源比较分散，但也存在几个主要的大型枢纽或居住区站点。
""")