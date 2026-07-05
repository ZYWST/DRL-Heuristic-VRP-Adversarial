import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 1. 读取数据
# 请替换你的文件名
file_path = 'data/reward.csv' 
df = pd.read_csv(file_path)

# ================= 配置区 =================
# 设置平滑系数 (0~1)，越大越平滑。
# TensorBoard 默认大概在 0.6 - 0.9 之间。
# 如果你想让趋势看起来更稳健，可以调大这个数。
SMOOTHING_WEIGHT = 0.6 

# 颜色设置 (TensorBoard 风格的绿色)
COLOR_RAW = '#A5D6A7'  # 浅色（原始数据）
COLOR_SMOOTH = '#2E7D32' # 深色（平滑数据）
# =========================================

# 定义平滑函数 (类似 TensorBoard 的算法)
def smooth(scalars, weight):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# 计算平滑后的数据
# 注意：这里使用了 .to_numpy() 来避免之前的 ValueError
step = df['Step'].to_numpy()
value = df['Value'].to_numpy()
smoothed_value = smooth(value, SMOOTHING_WEIGHT)

# 开始绘图
plt.figure(figsize=(10, 5), dpi=330) # 宽长一点，让曲线看起来跨度更大
sns.set_style("whitegrid") # 开启网格，方便对比高度差

# 字体设置：英文使用 Times New Roman，中文使用 宋体 (SimSun)
# 注意：在 Windows 上一般字体名为 'SimSun'，若找不到可改为字体文件路径
font_en = FontProperties(family='Times New Roman')
font_ch = FontProperties(family='SimSun')
# 避免负号显示为方块
mpl.rcParams['axes.unicode_minus'] = False

# 1. 画原始数据的阴影 (浅色线)
plt.plot(step, value, color=COLOR_RAW, alpha=0.4, linewidth=1, label='Original')

# 2. 画平滑后的主线 (深色线)
plt.plot(step, smoothed_value, color=COLOR_SMOOTH, linewidth=2, label='Smoothed')

# === 关键步骤：夸张坐标轴 ===
# 策略：不从0开始，而是紧贴着数据的最小值和最大值
y_min = value.min()
y_max = value.max()
y_range = y_max - y_min

# 上下各留 5% 的余地，这样波峰波谷会几乎顶格，视觉冲击力最强
plt.ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
plt.margins(x=0) # X轴左右不留白

# 设置标题和标签（中文部分按要求显示）
plt.title("Evaluation Mean Reward", fontsize=22, fontweight='bold', fontproperties=font_en)
plt.xlabel("步数", fontsize=18, fontproperties=font_ch)
plt.ylabel("平均奖励", fontsize=18, fontproperties=font_ch)

# 刻度优化：使用科学计数法 (如果数值很大)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# 去掉上边框和右边框 (学术风)
sns.despine()

# 优化刻度字体为英文（Times New Roman）以保证英文字体一致
ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_en)
    label.set_fontsize(18)

plt.legend(prop=font_en)

plt.tight_layout()
plt.savefig('mean_reward_exaggerated.svg', bbox_inches='tight')
plt.show()