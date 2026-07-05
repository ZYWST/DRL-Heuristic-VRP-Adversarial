import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置区域 =================
LOG_ROOT_DIR = "./problem_sensitivity_logs"
OUTPUT_EXCEL = "problem_sensitivity_fina0311.xlsx"
OUTPUT_IMG_CR = "problem_CR_lineplot0311.svg"
OUTPUT_IMG_ES = "problem_ES_heatmap0311.svg"

# 图表字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 参数名称映射 (让图表更好看)
PARAM_MAP = {
    'U': '中断数量 (U)',
    'C': '车辆容量 (C)',
    'H': '时间限制 (H)'
}

def get_best_fitness(file_path):
    """从日志提取最优值"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # 1. 优先匹配 DRL New Best
        matches = re.findall(r"(-?[\d\.]+)\s*->\s*(?:🎯)?NEW BEST!", content)
        if matches: return float(matches[-1])
        # 2. 其次匹配 Final Fitness
        matches = re.findall(r"Final Fitness:\s*(-?[\d\.]+)", content)
        if matches: return float(matches[-1])
    except:
        pass
    return None

def parse_logs():
    """遍历日志目录结构提取数据"""
    if not os.path.exists(LOG_ROOT_DIR):
        print(f"❌ 错误：找不到日志目录 {LOG_ROOT_DIR}")
        return pd.DataFrame()

    records = []
    
    # 1. 第一层：算法种子 (Seed_2024...)
    seed_dirs = glob.glob(os.path.join(LOG_ROOT_DIR, "Seed_*"))
    print(f"📂 正在扫描 {len(seed_dirs)} 个随机种子文件夹...")

    for s_dir in seed_dirs:
        seed_str = os.path.basename(s_dir).split('_')[-1] # 2024
        
        # 2. 第二层：参数种类 (Param_U, Param_C...)
        param_dirs = glob.glob(os.path.join(s_dir, "Param_*"))
        
        for p_dir in param_dirs:
            raw_param = os.path.basename(p_dir).split('_')[-1] # U, C, H
            # 映射为中文名或保持原名
            param_name = PARAM_MAP.get(raw_param, raw_param)
            
            # 3. 第三层：缩放比例 (Scale_0.7_-30%...)
            scale_dirs = glob.glob(os.path.join(p_dir, "Scale_*"))
            
            for sc_dir in scale_dirs:
                # 解析文件夹名: Scale_0.7_-30%
                folder_name = os.path.basename(sc_dir)
                parts = folder_name.split('_')
                
                # change_str 是最后一部分 "-30%"
                change_str = parts[-1] 
                try:
                    change_val = int(change_str.replace('%', '').replace('+', ''))
                except:
                    continue # 解析失败跳过

                log_file = os.path.join(sc_dir, "run.log")
                if not os.path.exists(log_file): continue
                
                obj = get_best_fitness(log_file)
                if obj is not None:
                    records.append({
                        "Seed": seed_str,
                        "Parameter": param_name,
                        "ChangeVal": change_val,
                        "Obj": obj
                    })
    
    return pd.DataFrame(records)

def main():
    print("🚀 开始分析问题参数敏感性...")
    
    # 1. 提取原始数据
    df_raw = parse_logs()
    if df_raw.empty:
        print("❌ 未提取到数据，请检查目录结构。")
        return

    # 2. 计算基准值 (Baseline)
    # 对于每个参数，找到 ChangeVal = 0 的所有记录，计算平均值作为该参数的 Baseline
    baseline_map = {}
    unique_params = df_raw['Parameter'].unique()
    
    for param in unique_params:
        # 取出该参数下 ChangeVal == 0 的所有行
        base_rows = df_raw[(df_raw['Parameter'] == param) & (df_raw['ChangeVal'] == 0)]
        if not base_rows.empty:
            baseline_map[param] = base_rows['Obj'].mean()
        else:
            print(f"⚠️ 警告: 参数 {param} 缺少 0% 基准数据！")
            baseline_map[param] = np.nan

    # 3. 聚合数据 (消除随机种子影响)
    # 按 Parameter 和 ChangeVal 分组，计算 Obj 的平均值
    df_agg = df_raw.groupby(['Parameter', 'ChangeVal'])['Obj'].mean().reset_index()
    
    # 将基准值合并进去
    df_agg['Baseline'] = df_agg['Parameter'].map(baseline_map)
    
    # 按 ChangeVal 排序 (-30 -> +30)
    df_agg.sort_values(by=['Parameter', 'ChangeVal'], inplace=True)

    # 4. 计算 CR (%)
    # CR = (Avg_Obj - Baseline) / Baseline * 100
    df_agg['CR_Percent'] = (df_agg['Obj'] - df_agg['Baseline']) / df_agg['Baseline'] * 100

    # 5. 计算 ES (Sensitivity) - 后向差分
    es_list = []
    
    for param in unique_params:
        subset = df_agg[df_agg['Parameter'] == param].copy().sort_values('ChangeVal')
        
        cr_vals = subset['CR_Percent'].values
        change_vals = subset['ChangeVal'].values
        es_vals = []
        
        for i in range(len(change_vals)):
            # 第一个点 (-30%) 无法计算 ES
            if i == 0:
                es_vals.append(np.nan)
            else:
                # 转换为小数进行计算
                diff_cr = (cr_vals[i] - cr_vals[i-1]) / 100.0
                diff_change = (change_vals[i] - change_vals[i-1]) / 100.0
                
                if abs(diff_change) < 1e-6:
                    es = 0
                else:
                    es = diff_cr / diff_change
                es_vals.append(es)
        
        subset['ES'] = es_vals
        es_list.append(subset)

    df_final = pd.concat(es_list)

    # 6. 保存 Excel
    # Raw_Data Sheet: 包含每个种子的原始数据
    # Summary Sheet: 包含计算出的 CR 和 ES
    output_cols = ['Parameter', 'ChangeVal', 'Obj', 'CR_Percent', 'ES']
    summary_df = df_final[output_cols].copy()
    summary_df.columns = ['参数种类', '调整比例(%)', '平均目标值', '变化率CR(%)', '敏感性ES']

    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        df_raw.to_excel(writer, sheet_name='Raw_Logs', index=False)
        summary_df.to_excel(writer, sheet_name='Summary_Metrics', index=False, float_format="%.4f")
    
    print(f"✅ Excel 已保存: {OUTPUT_EXCEL}")

    # 7. 绘图 (Matplotlib)
    plot_cr_line_matplotlib(df_final)
    plot_es_heatmap_matplotlib(df_final)

def plot_cr_line_matplotlib(df):
    plt.figure(figsize=(10, 6))
    
    params = df['Parameter'].unique()
    # 使用不同标记和颜色
    markers = ['o', 's', '^', 'D'] 
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 蓝、橙、绿

    for i, param in enumerate(params):
        subset = df[df['Parameter'] == param].sort_values('ChangeVal')
        x = subset['ChangeVal'].values
        y = subset['CR_Percent'].values
        
        plt.plot(x, y, 
                 marker=markers[i % len(markers)], 
                 color=colors[i % len(colors)],
                 linewidth=2.5, 
                 label=param)

    # 基准线
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.title('问题参数敏感性: 变化率 (CR)', fontsize=14, fontweight='bold')
    plt.xlabel('参数调整比例 (%)', fontsize=12)
    plt.ylabel('目标值变化率 CR (%)', fontsize=12)
    plt.xticks([-30, -20, -10, 0, 10, 20, 30])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='参数类型')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_CR, dpi=300)
    print(f"📈 CR 折线图已保存: {OUTPUT_IMG_CR}")

def plot_es_heatmap_matplotlib(df):
    # 准备矩阵数据
    # 过滤掉 ES 为空的数据 (即 -30% 那一列)
    df_clean = df.dropna(subset=['ES']).copy()
    
    params = sorted(df_clean['Parameter'].unique())
    changes = sorted(df_clean['ChangeVal'].unique()) # 应该是 [-20, -10, 0, 10, 20, 30]
    
    data_matrix = np.zeros((len(params), len(changes)))
    
    for r, param in enumerate(params):
        for c, chg in enumerate(changes):
            val = df_clean[(df_clean['Parameter'] == param) & (df_clean['ChangeVal'] == chg)]['ES'].values
            if len(val) > 0:
                data_matrix[r, c] = val[0]
            else:
                data_matrix[r, c] = np.nan

    plt.figure(figsize=(10, 5))
    
    # 绘制热力图
    max_abs = np.nanmax(np.abs(data_matrix))
    im = plt.imshow(data_matrix, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, aspect='auto')
    
    cbar = plt.colorbar(im)
    cbar.set_label('敏感性 (ES)', rotation=90)

    # 添加数值标签
    for r in range(len(params)):
        for c in range(len(changes)):
            val = data_matrix[r, c]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.5 * max_abs else "black"
                plt.text(c, r, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    plt.title('问题参数敏感性热力图 (ES 指标)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('参数调整比例 (%)', fontsize=12)
    plt.ylabel('参数类型', fontsize=12)
    
    # 设置刻度
    plt.xticks(range(len(changes)), [f"{int(x)}%" for x in changes])
    plt.yticks(range(len(params)), params)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_ES, dpi=300)
    print(f"📈 ES 热力图已保存: {OUTPUT_IMG_ES}")

if __name__ == "__main__":
    main()