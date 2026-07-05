import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
#读取analyze_sensitivity_results.py生成的Excel表格，计算超参数敏感性指标，并生成最终版图表
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置区域 =================
INPUT_FILE = "sensitivity_analysis_report.xlsx"
OUTPUT_EXCEL = "hyperparam_sensitivity_final.xlsx"
OUTPUT_IMG_CR = "hyperparam_CR_lineplot.png"
OUTPUT_IMG_ES = "hyperparam_ES_heatmap.png"

# 图表字体设置 (防止中文乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到输入文件 {INPUT_FILE}")
        return

    print("🚀 开始读取原始数据并计算超参数敏感性指标...")

    # 1. 读取 Raw_Data
    try:
        # 确保读取的是 Raw_Data sheet
        df_raw = pd.read_excel(INPUT_FILE, sheet_name='Raw_Data')
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 2. 聚合数据：计算 Avg_Obj 和 Avg_Baseline
    # Group By: Parameter, ChangeVal
    # ChangeVal 是整数 (-30, -20...)
    df_agg = df_raw.groupby(['Parameter', 'ChangeVal']).agg({
        'Obj': 'mean',
        'Baseline': 'mean'
    }).reset_index()

    # 3. 插入 0% 基准点
    # 原始 Raw_Data 通常不包含 0% 的实验记录（因为那是 baseline）
    # 我们需要手动构造 0% 的行，其中 Obj = Baseline
    unique_params = df_agg['Parameter'].unique()
    zeros_data = []

    for param in unique_params:
        # 获取该参数对应的 Avg_Baseline (理论上同一算例集下它是常数，取均值即可)
        base_val = df_agg[df_agg['Parameter'] == param]['Baseline'].mean()
        zeros_data.append({
            'Parameter': param,
            'ChangeVal': 0,
            'Obj': base_val,
            'Baseline': base_val
        })
    
    df_zeros = pd.DataFrame(zeros_data)
    
    # 合并数据并按 ChangeVal 排序 (-30 -> +30)
    df_calc = pd.concat([df_agg, df_zeros], ignore_index=True)
    df_calc.sort_values(by=['Parameter', 'ChangeVal'], inplace=True)

    # 4. 计算 CR (Change Rate %)
    # CR = (Avg_Obj - Avg_Base) / Avg_Base * 100
    df_calc['CR_Percent'] = (df_calc['Obj'] - df_calc['Baseline']) / df_calc['Baseline'] * 100

    # 5. 计算 ES (Elasticity / Sensitivity)
    # 要求：从 -20% 开始算，+30% 也要算 -> 使用后向差分
    # ES = (CR_curr - CR_prev) / (Change_curr - Change_prev)
    # 注意：这里的 CR 和 Change 都要换算成小数比例计算斜率，或者统一单位
    # 公式逻辑：(CR%变化) / (Change%变化) = CR_diff / 10
    
    es_list = []
    
    for param in unique_params:
        subset = df_calc[df_calc['Parameter'] == param].copy().sort_values('ChangeVal')
        
        # 获取数组
        cr_vals = subset['CR_Percent'].values
        change_vals = subset['ChangeVal'].values
        
        es_vals = []
        
        for i in range(len(change_vals)):
            curr_change = change_vals[i]
            
            # 第一个点 (-30%) 没有前驱，无法计算后向 ES
            if i == 0:
                es_vals.append(np.nan)
            else:
                # 后向差分
                diff_cr = (cr_vals[i] - cr_vals[i-1]) / 100.0 # 转回小数 (e.g. 1% -> 0.01)
                diff_change = (change_vals[i] - change_vals[i-1]) / 100.0 # 转回小数 (e.g. 10 -> 0.1)
                
                if abs(diff_change) < 1e-6:
                    es = 0
                else:
                    es = diff_cr / diff_change
                
                es_vals.append(es)
        
        subset['ES'] = es_vals
        es_list.append(subset)

    df_final = pd.concat(es_list)

    # 6. 生成 Excel 表格 (格式化)
    # 筛选列
    output_df = df_final[['Parameter', 'ChangeVal', 'Obj', 'CR_Percent', 'ES']].copy()
    output_df.columns = ['参数种类', '调整比例(%)', '时效性平均值', '变化率CR(%)', '敏感性ES']
    
    # 按照您的要求，剔除 -30% 的 ES 显示 (因为是 NaN)，或者保留并在 Excel 中为空
    # 但根据要求 "-20%开始，+30%要算"，-30% 行还是需要的 (展示 CR 和 Obj)
    
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        output_df.to_excel(writer, index=False, float_format="%.4f")
        print(f"✅ 数据表已保存: {OUTPUT_EXCEL}")

    # 7. 绘图 1: CR 折线图
    plot_cr_line(df_final)
    
    # 8. 绘图 2: ES 热力图 (过滤掉 NaN 的行，即 -30%)
    df_heatmap = df_final.dropna(subset=['ES'])
    plot_es_heatmap(df_heatmap)

def plot_cr_line(df):
    """绘制 CR 变化率折线图"""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # 绘制
    sns.lineplot(
        data=df, 
        x='ChangeVal', 
        y='CR_Percent', 
        hue='Parameter', 
        marker='o', 
        linewidth=2.5
    )
    
    # 基准线
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.title('Hyperparameter Sensitivity: Change Rate (CR)', fontsize=14, fontweight='bold')
    plt.xlabel('Parameter Adjustment (%)', fontsize=12)
    plt.ylabel('Change Rate CR (%)', fontsize=12)
    plt.xticks([-30, -20, -10, 0, 10, 20, 30])
    plt.legend(title='Parameter')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_CR, dpi=300)
    print(f"📈 CR 折线图已保存: {OUTPUT_IMG_CR}")

def plot_es_heatmap(df):
    """绘制 ES 敏感性热力图"""
    # 透视表: Index=Parameter, Col=ChangeVal, Value=ES
    pivot = df.pivot(index='Parameter', columns='ChangeVal', values='ES')
    
    plt.figure(figsize=(10, 5))
    
    # 绘制热力图 (RdBu_r: 红正蓝负白零)
    ax = sns.heatmap(
        pivot, 
        annot=True, 
        fmt=".3f", 
        cmap="RdBu_r", 
        center=0,
        linewidths=1, 
        linecolor='white'
    )
    
    plt.title('Hyperparameter Sensitivity Heatmap (ES Index)', fontsize=14, fontweight='bold')
    plt.xlabel('Parameter Adjustment (%)', fontsize=12)
    plt.ylabel('Parameter Type', fontsize=12)
    
    # 调整 X 轴标签格式
    xticklabels = [f"{int(x)}%" for x in pivot.columns]
    ax.set_xticklabels(xticklabels)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_ES, dpi=300)
    print(f"📈 ES 热力图已保存: {OUTPUT_IMG_ES}")

if __name__ == "__main__":
    main()