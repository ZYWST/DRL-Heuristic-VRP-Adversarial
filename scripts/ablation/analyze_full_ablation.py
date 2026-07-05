import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import glob
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= 配置区域 =================
LOG_ROOT_DIR = "./ablation_logs"
OUTPUT_EXCEL = "full_ablation_report.xlsx"
OUTPUT_IMG = "full_ablation_plot.png"

# 基准数据 (InstanceID -> Avg. Obj)
# 注意：这是特定算例的基准表现，与算法的随机种子无关
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

# ================= 核心工具函数 =================

def extract_instance_id(filename):
    """从文件名提取算例ID (例如 Case_Seed81203.dat -> Seed81203)"""
    match = re.search(r'(Seed\d+)', filename, re.IGNORECASE)
    return match.group(1) if match else None

def get_best_fitness(file_path):
    """从日志提取最优值 (优先匹配 NEW BEST)"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # 1. DRL New Best
        matches = re.findall(r"(-?[\d\.]+)\s*->\s*(?:🎯)?NEW BEST!", content)
        if matches: return float(matches[-1])
        # 2. DRL Old Best
        matches = re.findall(r"->\s*\(Best:\s*(-?[\d\.]+)\)", content)
        if matches: return float(matches[-1])
        # 3. Final Fitness
        matches = re.findall(r"Final Fitness:\s*(-?[\d\.]+)", content)
        if matches: return float(matches[-1])
    except:
        pass
    return None

def parse_all_logs():
    """遍历所有 Seed 文件夹提取数据"""
    if not os.path.exists(LOG_ROOT_DIR):
        print(f"❌ 找不到日志根目录: {LOG_ROOT_DIR}")
        return pd.DataFrame()

    records = []
    
    # 第一层：算法种子 (Seed_2024, Seed_2025...)
    seed_dirs = glob.glob(os.path.join(LOG_ROOT_DIR, "Seed_*"))
    print(f"📂 发现 {len(seed_dirs)} 个算法种子文件夹，开始全量扫描...")

    for s_dir in seed_dirs:
        algo_seed = os.path.basename(s_dir).split('_')[-1] # Extract 2024 from Seed_2024
        
        # 第二层：参数名 (ALPHA, POPULATION_SIZE...)
        param_dirs = glob.glob(os.path.join(s_dir, "*"))
        
        for p_dir in param_dirs:
            if not os.path.isdir(p_dir): continue
            param_name = os.path.basename(p_dir)
            
            # 第三层：参数值 (Val_0_0.2000...)
            val_dirs = glob.glob(os.path.join(p_dir, "Val_*"))
            
            for v_dir in val_dirs:
                # 解析参数值
                try:
                    folder_name = os.path.basename(v_dir)
                    val_str = folder_name.split('_')[-1]
                    val_float = float(val_str)
                except:
                    continue

                # 第四层：具体算例日志
                log_files = glob.glob(os.path.join(v_dir, "*.log"))
                
                for log in log_files:
                    instance_id = extract_instance_id(os.path.basename(log))
                    if not instance_id: continue
                    
                    fitness = get_best_fitness(log)
                    if fitness is None: continue
                    
                    records.append({
                        "AlgoSeed": algo_seed,     # 算法的随机种子 (2024, 2025...)
                        "Parameter": param_name,   # 消融参数名
                        "Value": val_float,        # 消融参数值
                        "InstanceID": instance_id, # 算例ID (Seed81203...)
                        "Obj": fitness             # 目标函数值
                    })
    
    return pd.DataFrame(records)

def process_data(df):
    if df.empty: return df
    
    # 1. 关联基准值
    baseline_df = pd.DataFrame(list(BASELINE_DATA.items()), columns=['InstanceID', 'Baseline'])
    merged = pd.merge(df, baseline_df, on='InstanceID', how='left')
    
    # 2. 填充无基准的情况 (防止报错)
    missing_mask = merged['Baseline'].isna()
    if missing_mask.any():
        print(f"⚠️ 警告: 有 {missing_mask.sum()} 条记录未找到基准值，Gap 将设为 0。")
        merged.loc[missing_mask, 'Baseline'] = merged.loc[missing_mask, 'Obj']

    # 3. 计算 Gap % (越大约好)
    # Gap = (实验值 - 基准值) / 基准值 * 100
    merged['Gap_Percent'] = (merged['Obj'] - merged['Baseline']) / merged['Baseline'] * 100
    
    return merged

def plot_full_analysis(df):
    """绘制全量分析图 (聚合所有 Seed)"""
    sns.set(style="whitegrid", context="talk")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    params = df['Parameter'].unique()
    if len(params) == 0: return

    # 创建子图
    fig, axes = plt.subplots(1, len(params), figsize=(7 * len(params), 6), sharey=True)
    if len(params) == 1: axes = [axes]

    for i, param in enumerate(sorted(params)):
        ax = axes[i] if len(params) > 1 else axes
        subset = df[df['Parameter'] == param]
        
        # 聚合计算: 对每个 Parameter Value，计算所有 Seed 和所有 Instance 的平均 Gap
        # estimator='mean' 画线, errorbar='sd' 画标准差阴影
        sns.lineplot(
            data=subset, 
            x='Value', 
            y='Gap_Percent',
            marker='o', 
            linewidth=3,
            errorbar='sd', # 显示标准差范围 (Standard Deviation)
            ax=ax,
            label='Mean Gap (All Seeds)'
        )
        
        # 绘制 0% 基准线
        ax.axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Baseline')

        # 装饰
        ax.set_title(f"Sensitivity: {param}", fontweight='bold', fontsize=16)
        ax.set_xlabel("Parameter Value (Physical)", fontsize=14)
        if i == 0:
            ax.set_ylabel("Gap to Baseline (%)", fontsize=14)
        
        # 整数参数刻度优化
        if param in ["POPULATION_SIZE", "SA_METROPOLIS_LEN"]:
            unique_vals = sorted(subset['Value'].unique())
            # 如果点太多，稀疏显示
            if len(unique_vals) > 15:
                ax.set_xticks(unique_vals[::2])
            else:
                ax.set_xticks(unique_vals)

    plt.suptitle("DRL Ablation Study: Full Analysis (Aggregated over 3 Seeds)", fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=330)
    print(f"📈 全量分析图已保存: {os.path.abspath(OUTPUT_IMG)}")

def main():
    print("🚀 开始全量数据分析...")
    
    # 1. 提取
    raw_df = parse_all_logs()
    if raw_df.empty:
        print("❌ 未提取到任何数据，请检查 ablation_logs 目录结构。")
        return
    print(f"✅ 提取完成，共 {len(raw_df)} 条记录。")

    # 2. 处理
    processed_df = process_data(raw_df)

    # 3. 生成 Excel 报告
    #   Sheet1: Raw_Data (所有原始记录)
    #   Sheet2: Aggregated (按参数值聚合，计算 Mean/Std/Max/Min)
    agg_stats = processed_df.groupby(['Parameter', 'Value'])['Gap_Percent'].agg(
        ['mean', 'std', 'min', 'max', 'count']
    ).reset_index()
    
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        processed_df.to_excel(writer, sheet_name='All_Raw_Data', index=False)
        agg_stats.to_excel(writer, sheet_name='Aggregated_Stats', index=False)
    
    print(f"✅ Excel 报告已保存: {os.path.abspath(OUTPUT_EXCEL)}")

    # 4. 绘图
    plot_full_analysis(processed_df)

if __name__ == "__main__":
    main()