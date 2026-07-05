import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ================= 配置区域 =================
# 设置为 None 则自动查找当前目录最新的 .log 文件
SPECIFIC_LOG_FILE = None 
# SPECIFIC_LOG_FILE = "problem_sensitivity_logs/Seed_2025/Param_C/Scale_1.0_+0%/run.log"

def get_latest_log():
    """获取当前目录下最新的 .log 文件"""
    logs = glob.glob("*.log")
    # 也可以递归查找子目录
    if not logs:
        logs = glob.glob("**/*.log", recursive=True)
    
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)

def parse_log_file(file_path):
    """
    解析日志文件
    针对格式: Step [  1/300] ... Act: [Pop= 9, α=0.34, SA_Len= 21]
    """
    steps = []
    pops = []
    alphas = []
    sa_lens = []

    # 核心修正：
    # 1. Step\s*\[\s*(\d+) -> 匹配 Step [  1  (处理多余空格)
    # 2. Pop=\s*(\d+)      -> 匹配 Pop= 9     (处理多余空格)
    # 3. α=([\d\.]+)       -> 匹配 α=0.34
    # 4. SA_Len=\s*(\d+)   -> 匹配 SA_Len= 21 (处理多余空格)
    pattern = re.compile(r"Step\s*\[\s*(\d+).*?Act:\s*\[Pop=\s*(\d+),\s*α=([\d\.]+),\s*SA_Len=\s*(\d+)\]")

    print(f"🕵️ 正在解析文件: {file_path}")
    print(f"   使用正则: {pattern.pattern}")

    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    step = int(match.group(1))
                    pop = int(match.group(2))
                    alpha = float(match.group(3))
                    sa_len = int(match.group(4))

                    steps.append(step)
                    pops.append(pop)
                    alphas.append(alpha)
                    sa_lens.append(sa_len)
                    
                    # 打印前3行提取结果，用于调试
                    if count < 3:
                        print(f"   [Debug Line {count+1}] 提取成功 -> Step:{step}, Pop:{pop}, Alpha:{alpha}, Len:{sa_len}")
                    count += 1
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return None

    print(f"📊 共提取到 {count} 条数据")
    return steps, pops, alphas, sa_lens

def plot_drl_dynamics(steps, pops, alphas, sa_lens, filename):
    if not steps:
        print("❌ 未提取到任何数据！请检查日志文件内容是否与脚本中的正则匹配。")
        return

    # 设置样式 (使用通用样式避免报错)
    plt.style.use('ggplot')
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建 3 个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 图 1: 种群规模 (Population Size) - 使用空心小圆标记每个数据点
    ax1.plot(steps, pops, color='#1f77b4', linewidth=1.5, label='Pop Size',
             marker='o', markersize=5, markerfacecolor='none', markeredgecolor='#1f77b4', markeredgewidth=0.8)
    
    ax1.set_ylabel('N_pop', fontsize=18, fontweight='bold')
    # ax1.set_title(f'DRL 动作参数随迭代变化 (来源: {os.path.basename(filename)})', fontsize=14, pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # 图 2: 降温系数 (Alpha)
    ax2.plot(steps, alphas, color='#ff7f0e', linewidth=1.5, label='Alpha',
             marker='o', markersize=5, markerfacecolor='none', markeredgecolor='#ff7f0e', markeredgewidth=0.8)

    ax2.set_ylabel('α', fontsize=18, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    # 统一刻度标签字体大小为 18 磅
    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis='both', labelsize=18)

    # 图 3: SA 搜索长度 (SA Length)
    ax3.plot(steps, sa_lens, color='#2ca02c', linewidth=1.5, label='SA Length',
             marker='o', markersize=5, markerfacecolor='none', markeredgecolor='#2ca02c', markeredgewidth=0.8)

    ax3.set_ylabel('L_SA', fontsize=18, fontweight='bold')
    ax3.set_xlabel('Step', fontsize=18, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right')

    # 强制 X 轴为整数
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 将 X 轴刻度设置为 5 的倍数
    try:
        xmin = min(steps)
        xmax = max(steps)
        # 以 5 为步长对齐刻度范围（取整到 5 的倍数）
        start = (xmin // 5) * 5
        end = ((xmax + 4) // 5) * 5
        ax3.set_xticks(range(start, end + 1, 5))
    except Exception:
        # 如果 steps 不是可迭代或有其他问题，忽略刻度设置
        pass

    plt.tight_layout()
    
    # 保存
    output_file = "drl_actions_fixed_plot.svg"
    plt.savefig(output_file, dpi=330)
    print(f"✅ 绘图完成！已保存为: {os.path.abspath(output_file)}")
    plt.show()

def main():
    # 1. 查找日志
    log_file = SPECIFIC_LOG_FILE
    if log_file is None:
        log_file = get_latest_log()
    
    if log_file is None:
        print("❌ 当前目录下找不到 .log 文件。请把脚本放在日志目录下，或者修改 SPECIFIC_LOG_FILE 路径。")
        return

    # 2. 解析
    result = parse_log_file(log_file)
    
    # 3. 绘图
    if result:
        steps, pops, alphas, sa_lens = result
        if len(steps) > 0:
            plot_drl_dynamics(steps, pops, alphas, sa_lens, log_file)
        else:
            print("❌ 数据列表为空，无法绘图。")

if __name__ == "__main__":
    main()