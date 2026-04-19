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