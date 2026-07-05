import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
#读取原始日志文件，提取最优值，计算与基准的偏差，并生成 Excel 报告
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 配置区域 =================
LOG_ROOT_DIR = "./sensitivity_logs"
OUTPUT_EXCEL = "sensitivity_analysis_report.xlsx"
OUTPUT_IMG = "sensitivity_analysis_plot.png"

# [关键] 基准数据 (来自您的截图)
# Key: 算例文件名中的唯一标识 (Seedxxxx), Value: 基准 Avg. Obj.
BASELINE_DATA = {
    "Seed81203":  177948.813,
    "Seed80911":  306110.071,
    "Seed71944":  484022.888,
    "Seed130903": 526498.705,
    "Seed81555":  225488.565,
    "Seed71758":  188094.365,
    "Seed71808":  47859.758,
    "Seed81541":  241166.039,
    "Seed81527":  109821.123,
    "Seed81515":  753497.04
}

# 敏感性参数列表
PARAMS = ['SA_ALPHA', 'ACO_EVAPORATION', 'select_temp']

# ================= 核心逻辑 =================

def extract_instance_id(filename):
    """
    从文件名中提取 Seedxxxx 作为实例 ID
    例如: Case_Clustered_Seed81203.dat -> Seed81203
    """
    match = re.search(r'(Seed\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def get_best_fitness(file_path):
    """
    从日志中提取最优值 (兼容多种格式)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # 策略 1: DRL 新版 (1234.5 -> 🎯NEW BEST!)
        matches = re.findall(r"(-?[\d\.]+)\s*->\s*(?:🎯)?NEW BEST!", content)
        if matches: return float(matches[-1])

        # 策略 2: DRL 旧版 ((Best: 1234.5))
        matches = re.findall(r"->\s*\(Best:\s*(-?[\d\.]+)\)", content)
        if matches: return float(matches[-1])
        
        # 策略 3: Baseline (Final Fitness: 1234.5)
        matches = re.findall(r"Final Fitness:\s*(-?[\d\.]+)", content)
        if matches: return float(matches[-1])
            
    except Exception as e:
        print(f"⚠️ 读取失败 {file_path}: {e}")
    return None

def parse_logs():
    """遍历所有日志并提取数据"""
    records = []
    
    # 查找所有子文件夹 (例如 SA_ALPHA_+10%)
    scenario_dirs = glob.glob(os.path.join(LOG_ROOT_DIR, "*"))
    
    print(f"📂 正在扫描 {len(scenario_dirs)} 个场景文件夹...")

    for s_dir in scenario_dirs:
        if not os.path.isdir(s_dir): continue
        scenario_name = os.path.basename(s_dir)
        
        # 解析参数名和变化幅度
        # 假设文件夹名为: PARAM_+10%
        try:
            # 找到最后一个下划线分割
            last_underscore = scenario_name.rfind('_')
            param_name = scenario_name[:last_underscore]
            change_str = scenario_name[last_underscore+1:] # e.g., "+10%"
            
            # 将百分比转为数字用于排序 (+10% -> 10, -30% -> -30)
            change_val = int(change_str.replace('%', '').replace('+', ''))
        except:
            print(f"⚠️ 跳过无法解析的文件夹名: {scenario_name}")
            continue

        log_files = glob.glob(os.path.join(s_dir, "*.log"))
        
        for log_path in log_files:
            fname = os.path.basename(log_path)
            
            # 提取实例 ID (用于匹配基准)
            instance_id = extract_instance_id(fname)
            if not instance_id:
                continue
                
            # 提取 Fitness
            fit = get_best_fitness(log_path)
            if fit is None:
                continue
                
            records.append({
                "Parameter": param_name,
                "ChangeStr": change_str,
                "ChangeVal": change_val,
                "InstanceID": instance_id,
                "Obj": fit,
                "LogFile": fname
            })
            
    return pd.DataFrame(records)

def process_and_plot(df):
    if df.empty:
        print("❌ 没有提取到数据！")
        return

    # 1. 关联基准数据
    # 将基准字典转为 DataFrame 方便合并
    baseline_df = pd.DataFrame(list(BASELINE_DATA.items()), columns=['InstanceID', 'Baseline'])
    
    # 合并数据
    merged_df = pd.merge(df, baseline_df, on='InstanceID', how='left')
    
    # 检查是否有未匹配到的算例
    missing = merged_df[merged_df['Baseline'].isna()]
    if not missing.empty:
        print("⚠️ 警告: 以下算例未在基准表中找到:")
        print(missing['InstanceID'].unique())
        # 填充缺失值为 实验值本身 (即 Gap=0)，防止报错，但在 Excel 中标记
        merged_df['Baseline'].fillna(merged_df['Obj'], inplace=True)

    # 2. 计算 Gap (偏差百分比)
    # Gap > 0 表示优于基准 (假设越大越好)，Gap < 0 表示劣于基准
    merged_df['Gap_Percentage'] = (merged_df['Obj'] - merged_df['Baseline']) / merged_df['Baseline'] * 100

    # 3. 聚合数据 (按 Parameter, ChangeVal, InstanceID 取平均 - 处理多随机种子的情况)
    # 先对同一种子跑多次的情况取平均
    agg_df = merged_df.groupby(['Parameter', 'ChangeVal', 'ChangeStr', 'InstanceID']).agg({
        'Obj': 'mean',
        'Baseline': 'mean',
        'Gap_Percentage': 'mean'
    }).reset_index()

    # 4. 生成 Excel 报告
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        # Sheet 1: 详细数据
        merged_df.to_excel(writer, sheet_name='Raw_Data', index=False)
        
        # Sheet 2: 统计摘要 (按参数和变化幅度)
        summary = agg_df.groupby(['Parameter', 'ChangeVal', 'ChangeStr'])['Gap_Percentage'].mean().reset_index()
        summary.sort_values(by=['Parameter', 'ChangeVal'], inplace=True)
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
    print(f"✅ Excel 报告已生成: {os.path.abspath(OUTPUT_EXCEL)}")

    # 5. 绘制敏感性图
    plot_sensitivity_charts(agg_df)

def plot_sensitivity_charts(df):
    """
    绘制 1x3 的子图，展示三个参数的敏感性
    Y轴: 相对基准的平均偏差 (%)
    X轴: 参数变化幅度 (-30% ~ +30%)
    """
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

    unique_params = df['Parameter'].unique()
    # 确保只画这三个 (如果日志里有别的乱入)
    target_params = [p for p in PARAMS if p in unique_params]
    
    if not target_params:
        print("❌ 未找到指定参数的日志数据，跳过绘图。")
        return

    fig, axes = plt.subplots(1, len(target_params), figsize=(18, 6), sharey=True)
    if len(target_params) == 1: axes = [axes] # 兼容只有一个参数的情况

    # 计算全局 Y 轴范围，保持统一
    y_min = df['Gap_Percentage'].min() - 5
    y_max = df['Gap_Percentage'].max() + 5

    for i, param in enumerate(target_params):
        ax = axes[i]
        subset = df[df['Parameter'] == param]
        
        # 计算每个变化幅度的平均 Gap 和 标准差 (误差棒)
        stats = subset.groupby('ChangeVal')['Gap_Percentage'].agg(['mean', 'std']).reset_index()
        stats.sort_values('ChangeVal', inplace=True)
        
        # 绘制折线图
        ax.errorbar(
            stats['ChangeVal'], 
            stats['mean'], 
            yerr=stats['std'], 
            fmt='-o', 
            linewidth=2, 
            markersize=8, 
            capsize=5, 
            label='Average Gap'
        )
        
        # 绘制基准线 (0%)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Baseline (0%)')
        
        # 装饰图表
        ax.set_title(f"Parameter: {param}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Change (%)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Avg. Gap to Baseline (%)", fontsize=12)
        
        ax.set_xticks([-30, -20, -10, 10, 20, 30])
        ax.set_xticklabels(["-30%", "-20%", "-10%", "+10%", "+20%", "+30%"])
        
        # 添加数值标签
        for _, row in stats.iterrows():
            ax.annotate(
                f"{row['mean']:.1f}%", 
                xy=(row['ChangeVal'], row['mean']), 
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                fontsize=10
            )

    plt.suptitle("Sensitivity Analysis: Impact on Objective Value (Higher is Better)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300)
    print(f"✅ 敏感性分析图已保存: {os.path.abspath(OUTPUT_IMG)}")
    plt.show()

if __name__ == "__main__":
    # 检查日志目录是否存在
    if not os.path.exists(LOG_ROOT_DIR):
        print(f"❌ 找不到目录: {LOG_ROOT_DIR}")
        print("   请先运行 batch_sensitivity.py 生成日志！")
    else:
        print("🚀 开始分析敏感性数据...")
        df_raw = parse_logs()
        process_and_plot(df_raw)