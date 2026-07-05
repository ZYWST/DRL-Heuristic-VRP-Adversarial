import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 1. 读取下载的 CSV 文件
df = pd.read_csv('data/ent_coef.csv')
font_en = FontProperties(family='Times New Roman')
font_ch = FontProperties(family='SimSun')

# 2. 数据清洗：去除你刚才提到的 20k-22k 之间的回退震荡
# 假设震荡区间是 20000 到 22500 步，根据实际 CSV 调整
df_clean = df[~((df['Step'] >= 20000) & (df['Step'] <= 22500))]

# 3. 绘图设置 (确保纯白底，学术风格)
plt.figure(figsize=(10, 6), dpi=330) # 300 DPI 适合打印
sns.set_style("white") # 设置风格为白底，无网格或灰色背景

# 绘制曲线
plt.plot(df_clean['Step'], df_clean['Value'], color='#7CB342', linewidth=1.5) # 使用原本的绿色

# 4. 调整细节
# plt.title("entropy coefficient", fontsize=14, fontproperties=font_en)
plt.xlabel("Steps", fontsize=18, fontproperties=font_en)
plt.ylabel("Entropy coefficient", fontsize=18, fontproperties=font_en)

# 强制设定标题大小，避免样式覆盖
# plt.rcParams['axes.titlesize'] = 26
# ax = plt.gca()
# ax.set_title("critic loss", fontsize=26, fontproperties=font_en)

# plt.xlabel("Steps", fontsize=18, fontproperties=font_en)
# plt.ylabel("Critic loss", fontsize=18, fontproperties=font_en)

plt.grid(True, linestyle='--', alpha=0.3) # 如果需要网格，设淡一点；不需要则注释掉
ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontproperties(font_en)
    label.set_fontsize(18)
plt.margins(x=0) # 贴边
plt.box(True)    # 保留边框

# 5. 保存
# plt.savefig('critic_loss_clean.svg', bbox_inches='tight', facecolor='white')
plt.savefig('ent_coef _clean.svg', bbox_inches='tight') # PDF 格式适合插入 LaTeX
plt.show()