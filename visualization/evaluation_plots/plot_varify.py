import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ================= 配置区 =================
# 指定需要解析的日志文件路径
LOG_FILENAME = 'naive日志.txt'

# 全局设置为中文宋体，并修复负号显示 (学术期刊常用)
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 全局字体放大比例（如需整体放大/缩小统一调整）
TEXT_SCALE = 1.2

# ================= 算例数据硬编码区 =================
# 1. 节点坐标 (N)
nodes = {
    1: (-240.1, 103.9), 2: (-373.0, 76.9), 3: (-347.9, 18.3),
    4: (-502.6, 68.0), 5: (-382.8, 136.7), 6: (-456.9, 27.8),
    7: (-466.5, 167.2), 8: (-261.3, 36.2), 9: (-289.5, 74.6),
    10: (-365.4, 111.4), 11: (-440.0, 113.0)
}

# 2. 物理路网 (E) - 用于画浅色背景网格
edges_raw = [
    (1,5), (1,8), (1,9), (1,10), (2,3), (2,6), (2,9), (2,10), 
    (3,2), (3,6), (3,8), (4,6), (4,7), (5,1), (5,7), (5,10), (5,11), 
    (6,2), (6,3), (6,4), (6,11), (7,4), (7,5), (8,1), (8,3), (8,9), 
    (9,1), (9,2), (9,8), (10,1), (10,2), (10,5), (10,11), (11,5), (11,6), (11,10)
]
edges = list(set([tuple(sorted(e)) for e in edges_raw]))

# 3. 节点属性分类 (基于算例供需)
supplies = [8, 9, 10]
demands = [3, 4, 6, 11]
neutrals = [1, 2, 5, 7]

# ================= 动态日志解析引擎 =================
try:
    with open(LOG_FILENAME, 'r', encoding='utf-8') as f:
        log_text = f.read()
except FileNotFoundError:
    print(f"找不到日志文件 {LOG_FILENAME}，请确认路径。")
    exit()

# 解析 1: 提取所有车辆路径
paths = []
for match in re.finditer(r'Path:\s*(.+)', log_text):
    path_str = match.group(1)
    path = [int(x.strip()) for x in path_str.split('->')]
    paths.append(path)

# 解析 2: 提取中断边 (增强版：兼容强制中断与事后审计中断)
interrupted_edge = None

# 首先尝试从文件末尾的【审计结果】模块提取
match_audit = re.search(r'识别出的高流量中断边.*?❌\s*\((\d+),\s*(\d+)\)', log_text, re.DOTALL)
if match_audit:
    interrupted_edge = (int(match_audit.group(1)), int(match_audit.group(2)))
else:
    # 如果没有审计结果，则尝试从【中断边设置】模块提取
    match_int = re.search(r'被中断边:\s*\((\d+),\s*(\d+)\)', log_text)
    if match_int:
        interrupted_edge = (int(match_int.group(1)), int(match_int.group(2)))

# 解析 3: 提取路段流量
edge_flows = {}
for match in re.finditer(r'Edge\s*\((\d+),\s*(\d+)\):\s*Flow\s*=\s*([\d\.]+)', log_text):
    u, v, flow = int(match.group(1)), int(match.group(2)), float(match.group(3))
    edge_flows[(u, v)] = flow

# 解析 4: 提取各个节点的装卸货量 (Pickup / Deliver)
node_tasks = {}
for match in re.finditer(r'(Pickup|Deliver):\s*([\d\.]+)\s*of Good \d+ (?:at|to) Node (\d+)', log_text):
    action, amount, node = match.group(1), float(match.group(2)), int(match.group(3))
    if node not in node_tasks:
        node_tasks[node] = {'pickup': 0.0, 'deliver': 0.0}
    
    if action == 'Pickup':
        node_tasks[node]['pickup'] += amount
    else:
        node_tasks[node]['deliver'] += amount

# ================= 开始绘图 =================
fig, ax = plt.subplots(figsize=(10, 8))

# 1. 绘制底层路网 (虚线)
for u, v in edges:
    ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], 
            color='#CCCCCC', linestyle=':', zorder=1, lw=1.2)

# 2. 绘制带流量的路段标注 (绿色字体，吸附在路线上)
drawn_flow_edges = set()
for (u, v), flow in edge_flows.items():
    if flow > 0:
        mid_x = (nodes[u][0] + nodes[v][0]) / 2
        mid_y = (nodes[u][1] + nodes[v][1]) / 2
        
        # 计算文字倾斜角度，使其顺着路段方向
        dx, dy = nodes[v][0] - nodes[u][0], nodes[v][1] - nodes[u][1]
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90 or angle < -90:  # 保证文字不倒置
            angle += 180
            
        # 根据法向量计算一个微小的偏移，防止挡住箭头
        norm = np.sqrt(dx**2 + dy**2)
        nx, ny = -dy/norm * 6, dx/norm * 6
        
        ax.text(mid_x + nx, mid_y + ny, f"流量:{flow:.1f}", 
                ha='center', va='center', rotation=angle, 
                fontsize=8 * TEXT_SCALE, color='#006400', fontweight='bold', zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# 3. 绘制多车辆路径 (基于解析出来的 paths)
# 定义学术图表常用配色组合（蓝实线、红虚线、橙点划线等）
colors = ['#285F9F', '#D73027', '#F4A582', '#92C5DE']
linestyles = ['-', '--', '-.', ':']

for i, path in enumerate(paths):
    c = colors[i % len(colors)]
    ls = linestyles[i % len(linestyles)]
    for j in range(len(path)-1):
        u, v = path[j], path[j+1]
        ax.annotate("", xy=nodes[v], xytext=nodes[u],
                    arrowprops=dict(arrowstyle="-|>", color=c, linestyle=ls, 
                                    lw=1.8, mutation_scale=15, 
                                    connectionstyle="arc3,rad=0.15"), 
                    zorder=2)

# 4. 绘制各类节点
def plot_nodes(n_list, marker, facecolor, s):
    xs = [nodes[n][0] for n in n_list]
    ys = [nodes[n][1] for n in n_list]
    ax.scatter(xs, ys, marker=marker, facecolors=facecolor, edgecolors='black', s=s, zorder=3)

plot_nodes(supplies, 's', '#ADD8E6', 250)
plot_nodes(demands, 'o', '#FFB6C1', 250)
plot_nodes(neutrals, 'D', '#E0E0E0', 150)

# 5. 标节点序号 & 提货/送货量
for n, (x, y) in nodes.items():
    ax.text(x, y, str(n), ha='center', va='center', fontsize=9 * TEXT_SCALE, zorder=4, fontweight='bold')
    
    # 若该节点有提货/送货任务，则在节点正上方绘制小信息框
    if n in node_tasks:
        pickup = node_tasks[n]['pickup']
        deliver = node_tasks[n]['deliver']
        lines = []
        if pickup > 0: lines.append(f"提: {pickup:.1f}")
        if deliver > 0: lines.append(f"送: {deliver:.1f}")
        
        if lines:
            txt = "\n".join(lines)
            ax.text(x, y + 8, txt, ha='center', va='bottom', fontsize=8 * TEXT_SCALE, color='black', 
                    bbox=dict(boxstyle="round,pad=0.2", fc="#F8F9F9", ec="gray", lw=0.8, alpha=0.95), zorder=5)

# 6. 绘制中断边标记 (如果解析到了)
if interrupted_edge:
    u, v = interrupted_edge
    mid_x, mid_y = (nodes[u][0] + nodes[v][0]) / 2, (nodes[u][1] + nodes[v][1]) / 2
    ax.scatter(mid_x, mid_y, marker='x', color='red', s=150, lw=3, zorder=6)

# 7. 学术化自定义图例
legend_elements = [
    Line2D([0], [0], color='#CCCCCC', linestyle=':', lw=1.5, label='物理路网'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ADD8E6', markeredgecolor='black', markersize=10, label='供应节点'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFB6C1', markeredgecolor='black', markersize=10, label='需求节点'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E0E0E0', markeredgecolor='black', markersize=8, label='中转节点'),
]
# 为存在的车辆动态加图例
for i in range(len(paths)):
    c = colors[i % len(colors)]
    ls = linestyles[i % len(linestyles)]
    legend_elements.append(Line2D([0], [0], color=c, linestyle=ls, lw=1.8, label=f'车辆 {i+1} 路径'))
if interrupted_edge:
    legend_elements.append(Line2D([0], [0], marker='x', color='w', markeredgecolor='red', markersize=12, markeredgewidth=2.5, label='路网中断位置'))

ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10 * TEXT_SCALE, edgecolor='black', bbox_to_anchor=(1.02, 1))

# 底部标题
fig.text(0.5, 0.18, '(a)确定性策略下的应急物资配送方案', ha='center', va='bottom', fontsize=12 * TEXT_SCALE, fontname='SimSun', fontweight='bold')

# ================= 渲染与输出 =================
plt.axis('off')
ax.set_aspect('equal') # 保证相对位置比例不错乱

plt.savefig('vrp_routing_dynamic_naive.svg', dpi=330, bbox_inches='tight', pad_inches=0.01)
print(f"日志解析完成！发现 {len(paths)} 辆车路径, {len(edge_flows)} 条发生流量边。")
print("带有提送货和流量标记的高清学术图像已去白边保存！")
plt.show()