import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
abilene_promega_path = '/Users/chengwenhao/Desktop/Project/PyCharm/bsCheng/byCwh/control_plot/abilene_methods_promega_data.csv'
mininet_promega_path = '/Users/chengwenhao/Desktop/Project/PyCharm/bsCheng/byCwh/control_plot/mininet_methods_promega_data.csv'
abilene_pru_path = '/Users/chengwenhao/Desktop/Project/PyCharm/bsCheng/byCwh/control_plot/abilene_methods_pru_data.csv'
mininet_pru_path = '/Users/chengwenhao/Desktop/Project/PyCharm/bsCheng/byCwh/control_plot/mininet_methods_pru_data.csv'

# 读取CSV数据
abilene_promega_data = pd.read_csv(abilene_promega_path, header=None)
mininet_promega_data = pd.read_csv(mininet_promega_path, header=None)
abilene_pru_data = pd.read_csv(abilene_pru_path, header=None)
mininet_pru_data = pd.read_csv(mininet_pru_path, header=None)

# 提取列名并删除第一行
abilene_promega_columns = abilene_promega_data.iloc[0].tolist() if not abilene_promega_data.empty else ["CFR-RL", "Top-k critical", "Top-k", "ECMP"]
mininet_promega_columns = mininet_promega_data.iloc[0].tolist() if not mininet_promega_data.empty else ["CFR-RL", "Top-k critical", "Top-k", "ECMP"]
abilene_pru_columns = abilene_pru_data.iloc[0].tolist() if not abilene_pru_data.empty else ["CFR-RL", "Top-k critical", "Top-k", "ECMP"]
mininet_pru_columns = mininet_pru_data.iloc[0].tolist() if not mininet_pru_data.empty else ["CFR-RL", "Top-k critical", "Top-k", "ECMP"]

abilene_promega_data = abilene_promega_data.iloc[1:].reset_index(drop=True) if len(abilene_promega_data) > 1 else abilene_promega_data
mininet_promega_data = mininet_promega_data.iloc[1:].reset_index(drop=True) if len(mininet_promega_data) > 1 else mininet_promega_data
abilene_pru_data = abilene_pru_data.iloc[1:].reset_index(drop=True) if len(abilene_pru_data) > 1 else abilene_pru_data
mininet_pru_data = mininet_pru_data.iloc[1:].reset_index(drop=True) if len(mininet_pru_data) > 1 else mininet_pru_data

# 将数据转换为浮点数
for df in [abilene_promega_data, mininet_promega_data, abilene_pru_data, mininet_pru_data]:
    for col in df.columns:
        if not df.empty:
            df[col] = pd.to_numeric(df[col], errors='coerce')

# 定义颜色和线型
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
linestyles = ['-', '--', ':', '-.']

# 创建2x2的子图布局
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Promega数据绘制
# 为Abilene Promega数据集绘制CDF
axs[0, 0].set_title('Abilene Network - Promega', fontsize=14)
for i, column in enumerate(abilene_promega_columns):
    if not abilene_promega_data.empty:
        # 计算CDF
        sorted_data = np.sort(abilene_promega_data[i])
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF
        axs[0, 0].plot(sorted_data, y, color=colors[i], linestyle=linestyles[i], linewidth=2, label=column)

axs[0, 0].set_xlabel('链路利用率 (Promega)', fontsize=12)
axs[0, 0].set_ylabel('CDF', fontsize=12)
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].legend()

# 为Mininet Promega数据集绘制CDF
axs[0, 1].set_title('Mininet Network - Promega', fontsize=14)
for i, column in enumerate(mininet_promega_columns):
    if not mininet_promega_data.empty:
        # 计算CDF
        sorted_data = np.sort(mininet_promega_data[i])
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF
        axs[0, 1].plot(sorted_data, y, color=colors[i], linestyle=linestyles[i], linewidth=2, label=column)

axs[0, 1].set_xlabel('链路利用率 (Promega)', fontsize=12)
axs[0, 1].set_ylabel('CDF', fontsize=12)
axs[0, 1].grid(True, linestyle='--', alpha=0.7)
axs[0, 1].legend()

# PRU数据绘制
# 为Abilene PRU数据集绘制CDF
axs[1, 0].set_title('Abilene Network - PRU', fontsize=14)
for i, column in enumerate(abilene_pru_columns):
    if not abilene_pru_data.empty:
        # 计算CDF
        sorted_data = np.sort(abilene_pru_data[i])
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF
        axs[1, 0].plot(sorted_data, y, color=colors[i], linestyle=linestyles[i], linewidth=2, label=column)

axs[1, 0].set_xlabel('路径资源利用率 (PRU)', fontsize=12)
axs[1, 0].set_ylabel('CDF', fontsize=12)
axs[1, 0].grid(True, linestyle='--', alpha=0.7)
axs[1, 0].legend()

# 为Mininet PRU数据集绘制CDF
axs[1, 1].set_title('Mininet Network - PRU', fontsize=14)
for i, column in enumerate(mininet_pru_columns):
    if not mininet_pru_data.empty:
        # 计算CDF
        sorted_data = np.sort(mininet_pru_data[i])
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF
        axs[1, 1].plot(sorted_data, y, color=colors[i], linestyle=linestyles[i], linewidth=2, label=column)

axs[1, 1].set_xlabel('路径资源利用率 (PRU)', fontsize=12)
axs[1, 1].set_ylabel('CDF', fontsize=12)
axs[1, 1].grid(True, linestyle='--', alpha=0.7)
axs[1, 1].legend()

# 调整布局
plt.tight_layout()
plt.savefig('/Users/chengwenhao/Desktop/Project/PyCharm/bsCheng/byCwh/control_plot/combined_cdf_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 创建新的CDF图
fig, ax = plt.subplots(figsize=(8, 6))

# 为Abilene PRU数据集绘制CDF
ax.set_title('Day 2', fontsize=14)
for i, column in enumerate(abilene_pru_columns):
    if not abilene_pru_data.empty:
        # 计算CDF
        sorted_data = np.sort(abilene_pru_data[i])
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF
        ax.plot(sorted_data, y, color=colors[i], linestyle=linestyles[i], linewidth=2, label=column)

ax.set_xlabel('PRU', fontsize=12)
ax.set_ylabel('CDF', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left')

# 保存新的图像
plt.tight_layout()
plt.savefig('/Users/chengwenhao/Desktop/Project/PyCharm/bsCheng/byCwh/control_plot/day2_cdf_plot.png', dpi=300, bbox_inches='tight')
plt.show()
