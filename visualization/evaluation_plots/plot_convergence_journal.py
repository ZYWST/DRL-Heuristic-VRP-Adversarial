import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import re
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ================= 配置区域 =================
SPECIFIC_LOG_FILE = None 
# SPECIFIC_LOG_FILE = "run.log"  # 如果需要指定文件，请修改这里

def get_latest_log():
    """获取当前目录下最新的 .log 文件"""
    logs = glob.glob("*.log") + glob.glob("**/*.log", recursive=True)
    if not logs: return None
    return max(logs, key=os.path.getmtime)

def parse_convergence(file_path):
    """
    解析日志，提取迭代过程中的最优解变化
    返回: (所有步数, 所有步对应的最优值, 改进发生的步数, 改进发生时的值)
    """
    steps = []
    best_values = []
    
    # 记录发生改进的点 (用于画特殊的标记)
    improv_steps = []
    improv_values = []
    
    current_global_best = -float('inf') # 假设是最大化问题，初始为负无穷
    
    # 正则1: 捕获步数 Step [  1/300]
    re_step = re.compile(r"Step\s*\[\s*(\d+)")
    
    # 正则2: 捕获 NEW BEST 情况下的 Max 值
    # 格式: ... Max=81377347.75 -> 🎯NEW BEST!
    re_new_best = re.compile(r"Max=([\d\.]+).*?NEW BEST")
    
    # 正则3: 捕获非 NEW BEST 情况下的 Best 值 (保持)
    # 格式: ... -> (Best: 89940396.78)
    re_current_best = re.compile(r"\(Best:\s*([\d\.]+)\)")

    print(f"📖 正在解析: {os.path.basename(file_path)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 1. 提取步数
                step_match = re_step.search(line)
                if not step_match: continue
                step = int(step_match.group(1))
                
                val = None
                is_improvement = False
                
                # 2. 检查是否是新最优解
                new_best_match = re_new_best.search(line)
                if new_best_match:
                    val = float(new_best_match.group(1))
                    is_improvement = True
                else:
                    # 3. 检查是否是保持现有最优
                    curr_match = re_current_best.search(line)
                    if curr_match:
                        val = float(curr_match.group(1))
                
                # 数据录入
                if val is not None:
                    # 如果这是第一条数据，或者比记录的全局最优还大(防止日志乱序或回退)
                    if val > current_global_best:
                        current_global_best = val
                        # 只有当明确标记为 NEW BEST 或者数值真的变大时，才记为改进点
                        if is_improvement:
                            improv_steps.append(step)
                            improv_values.append(val)
                    
                    steps.append(step)
                    best_values.append(current_global_best)
                    
    except Exception as e:
        print(f"❌ 解析出错: {e}")
        return None, None, None, None

    return steps, best_values, improv_steps, improv_values

def plot_journal_style(steps, best_values, imp_steps, imp_vals, filename):
    if not steps:
        print("❌ 没有提取到数据")
        return

    # === 设置学术期刊绘图风格 ===
    # 字体配置 (首选 Times New Roman, 备选 Arial)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式字体类似 Times
    plt.rcParams['font.size'] = 18
    
    # 刻度朝内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    fig, ax = plt.subplots(figsize=(8, 6)) # 标准单栏或半页尺寸

    # 1. 绘制收敛曲线 (黑色实线, 线宽适中)
    ax.plot(steps, best_values, color='black', linestyle='-', linewidth=1.5, label='Current Best Objective')
    
    # 2. 绘制改进点 (空心圆或特定标记, 突出显示)
    # zorder设大一点，防止被线遮挡
    ax.scatter(imp_steps, imp_vals, facecolors='white', edgecolors='black', marker='o', s=40, zorder=3, label='New Best Found')

    # 3. 装饰图表
    ax.set_xlabel('Iteration (Generations)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=18, fontweight='bold')
    ax.set_title('Convergence Analysis', fontsize=20, pad=15)
    
    # 网格 (灰色虚线，不抢眼)
    ax.grid(True, linestyle='--', color='gray', alpha=0.4)
    
    # 设置图例 (带边框, 右下角)
    legend = ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    legend.get_frame().set_linewidth(0.8)

    # 4. 添加标注 (Annotation) - 标注初始值和最终值
    start_val = best_values[0]
    end_val = best_values[-1]
    gap = (end_val - start_val) / start_val * 100
    
    # 标注起点
    ax.annotate(f'Start: {start_val:.2e}', 
                xy=(steps[0], start_val), 
                xytext=(steps[0] + max(steps)*0.1, start_val),
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=10)
    
    # 标注终点
    ax.annotate(f'Final: {end_val:.2e}\n(Impv: +{gap:.1f}%)', 
                xy=(steps[-1], end_val), 
                xytext=(steps[-1] - max(steps)*0.3, end_val - (end_val-start_val)*0.1),
                arrowprops=dict(arrowstyle="->", color='black'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                fontsize=10)

    # 科学计数法 (如果数值很大)
    if max(best_values) > 10000:
        ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    
    # 保存图片 (PDF 是矢量图，适合插入论文；svg 用于预览)
    img_name = "convergence_plot_journal.svg"
    # pdf_name = "convergence_plot_journal.pdf"
    
    plt.savefig(img_name, dpi=300, bbox_inches='tight')
    # plt.savefig(pdf_name, bbox_inches='tight') # PDF for LaTeX
    
    print(f"✅ 绘图完成!")
    print(f"   - 预览图: {img_name}")
    # print(f"   - 论文用(矢量): {pdf_name}")
    plt.show()

def main():
    log_file = SPECIFIC_LOG_FILE
    if log_file is None:
        log_file = get_latest_log()
    
    if log_file is None:
        print("❌ 找不到日志文件")
        return

    # 解析
    steps, best_vals, imp_steps, imp_vals = parse_convergence(log_file)
    
    # 绘图
    if steps:
        plot_journal_style(steps, best_vals, imp_steps, imp_vals, log_file)
    else:
        print("❌ 数据解析为空，请检查日志格式")

if __name__ == "__main__":
    main()