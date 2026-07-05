import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import sys
import glob
import subprocess
import time
import argparse
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# =================配置区域=================
TARGET_SCRIPT = "run_with_agent.py"  # 你的主程序文件名
PROBLEM_DIR = "./problem_instances"  # 算例文件夹
LOG_ROOT_DIR = "./sensitivity_logs"  # 日志保存根目录

# 固定约束
MAX_TIME = 300
MAX_GENS = 1000

# 种子列表 (外层循环)
SEEDS = [2024, 2025, 2026] 

# 参数基准值
BASE_PARAMS = {
    'SA_ALPHA': 0.9,
    'ACO_EVAPORATION': 0.3,
    'select_temp': 1.0
}

# 变化幅度 (±10%, ±20%, ±30%)
VARIATIONS = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]

# =========================================

def generate_scenarios():
    """生成所有敏感性测试场景"""
    scenarios = []
    
    # 1. 遍历每个参数
    for param_name, base_val in BASE_PARAMS.items():
        # 2. 遍历每个变化幅度
        for change in VARIATIONS:
            new_val = 0.0
            
            # 特殊逻辑：SA_ALPHA 基于 (1-alpha) 缩放
            if param_name == 'SA_ALPHA':
                gap = 1.0 - base_val
                new_gap = gap * (1.0 + change)
                new_val = 1.0 - new_gap
                # 限制范围防止溢出 [0, 1]
                new_val = max(0.01, min(0.99, new_val))
            
            # 常规逻辑：直接缩放
            else:
                new_val = base_val * (1.0 + change)
                if param_name == 'ACO_EVAPORATION':
                    new_val = max(0.01, min(0.99, new_val))
            
            # 格式化幅度字符串 (例如 "+10%", "-30%")
            change_str = f"{int(change*100):+d}%"
            scenario_name = f"{param_name}_{change_str}"
            
            scenarios.append({
                "name": scenario_name,
                "param": param_name,
                "value": round(new_val, 4),
                "change": change_str
            })
            
    return scenarios

def run_sensitivity_analysis():
    # 1. 准备算例列表
    if not os.path.exists(PROBLEM_DIR):
        print(f"❌ 找不到算例目录: {PROBLEM_DIR}")
        return
    
    dats = glob.glob(os.path.join(PROBLEM_DIR, "*.dat"))
    # 按文件名排序保证顺序一致
    dats.sort()
    
    if not dats:
        print("❌ 未找到 .dat 文件")
        return

    scenarios = generate_scenarios()
    total_runs = len(SEEDS) * len(scenarios) * len(dats)
    current_run = 0

    print(f"🚀 开始敏感性分析")
    print(f"   - 种子数: {len(SEEDS)}")
    print(f"   - 算例数: {len(dats)}")
    print(f"   - 场景数: {len(scenarios)} (共 {total_runs} 次运行)")
    print("="*60)

    # 外层循环：种子 (按照你的要求)
    for seed in SEEDS:
        print(f"\n🌱 [Seed: {seed}]")
        
        # 中层循环：敏感性场景 (这是我们要对比的维度)
        for scen in scenarios:
            log_dir = os.path.join(LOG_ROOT_DIR, scen['name'])
            os.makedirs(log_dir, exist_ok=True)
            
            print(f"   📂 场景: {scen['name']} ({scen['param']} = {scen['value']})")
            
            # 内层循环：算例逐一求解
            for problem_path in dats:
                current_run += 1
                case_name = os.path.basename(problem_path)
                log_file = os.path.join(log_dir, f"{case_name}_S{seed}.log")
                
                # 如果日志已存在且包含"Finished"，跳过 (断点续传)
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if "Optimization Finished" in f.read():
                            print(f"      🔹 跳过已完成: {case_name}")
                            continue

                # 构建命令
                cmd = [
                    sys.executable, "-u", TARGET_SCRIPT,
                    "--problem", problem_path,
                    "--seed", str(seed),
                    "--timeout", str(MAX_TIME),
                    "--generations", str(MAX_GENS),
                    # 关键：动态传递参数
                    f"--{scen['param'].lower()}", str(scen['value']) 
                ]
                
                print(f"      ▶️  [{current_run}/{total_runs}] Running {case_name} ...", end="", flush=True)
                
                start_t = time.time()
                try:
                    with open(log_file, "w", encoding="utf-8") as f_log:
                        subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT)
                    elapsed = time.time() - start_t
                    print(f" ✅ ({elapsed:.1f}s)")
                except Exception as e:
                    print(f" ❌ Error: {e}")

    print("\n✅ 所有运行结束，开始汇总数据...")
    summarize_results()

# ================= 结果提取逻辑 (复用之前的通用提取器) =================
def extract_info_from_filename(filename):
    base_name = os.path.basename(filename)
    match = re.search(r'^(.*)_(?:S|seed)(\d+)', base_name, re.IGNORECASE)
    if match:
        problem = match.group(1).replace('.log', '').replace('.dat', '')
        if problem.endswith('_'): problem = problem[:-1]
        seed = int(match.group(2))
        return problem, seed
    return base_name, 0

def get_best_value(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return None
    
    # 兼容两种格式 (Baseline 和 DRL)
    patterns = [
        r"(-?[\d\.]+)\s*->\s*(?:🎯)?NEW BEST!", 
        r"->\s*\(Best:\s*(-?[\d\.]+)\)",
        r"Final Fitness:\s*(-?[\d\.]+)"
    ]
    for p in patterns:
        matches = re.findall(p, content)
        if matches: return float(matches[-1])
    return None

def summarize_results():
    data = []
    
    # 遍历所有场景文件夹
    scenario_dirs = glob.glob(os.path.join(LOG_ROOT_DIR, "*"))
    for s_dir in scenario_dirs:
        if not os.path.isdir(s_dir): continue
        scenario_name = os.path.basename(s_dir)
        
        # 尝试解析参数名和变动幅度
        # 假设格式是 PARAM_NAME_+10%
        try:
            param_part, change_part = scenario_name.rsplit('_', 1)
        except:
            param_part, change_part = scenario_name, "N/A"

        log_files = glob.glob(os.path.join(s_dir, "*.log"))
        
        for log in log_files:
            problem, seed = extract_info_from_filename(log)
            val = get_best_value(log)
            
            if val is not None:
                data.append({
                    "Scenario": scenario_name,
                    "Parameter": param_part,
                    "Change": change_part,
                    "Problem": problem,
                    "Seed": seed,
                    "Fitness": val
                })

    if not data:
        print("❌ 未提取到数据。")
        return

    df = pd.DataFrame(data)
    
    # 创建透视表: 行=[Parameter, Change, Problem], 列=Seed, 值=Fitness
    pivot = df.pivot_table(
        index=['Parameter', 'Change', 'Problem'], 
        columns='Seed', 
        values='Fitness', 
        aggfunc='max'
    )
    
    output_file = "sensitivity_summary.xlsx"
    pivot.to_excel(output_file)
    print(f"📊 敏感性分析报告已生成: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    run_sensitivity_analysis()