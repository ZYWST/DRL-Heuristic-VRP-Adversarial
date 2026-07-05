import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import re
import glob
import pandas as pd

# ================= 配置区域 =================
# 日志文件夹路径
LOG_DIR = "batch_logs"
# 输出文件名
OUTPUT_FILE = "solving_time_report.xlsx"

# Patience (早停) 策略配置
# 最小耐心步数 (防止大算例过早停止)
MIN_PATIENCE = 10
# 动态耐心比例 (针对中小规模算例，取总步数的百分比)
PATIENCE_RATIO = 0.1
# 大算例步数阈值 (如果总步数 <= 该值，则强制使用 MIN_PATIENCE，不使用比例)
LARGE_SCALE_STEP_THRESHOLD = 200
# ===========================================

def get_adaptive_k(total_steps):
    """
    根据算例规模（总步数）动态计算 Patience k
    """
    # 如果总步数很少（说明是大算例，单步时间长），使用固定耐心值
    if total_steps <= LARGE_SCALE_STEP_THRESHOLD:
        return MIN_PATIENCE
    else:
        # 如果总步数很多（说明是中小算例，迭代快），使用比例耐心值
        return max(MIN_PATIENCE, int(total_steps * PATIENCE_RATIO))

def parse_log_file(file_path):
    """
    解析单个日志文件，提取 (Instance, Seed, SolvingTime)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    filename = os.path.basename(file_path)
    
    # --- 1. 提取算例名和种子 ---
    # 假设格式: InstanceName_seed123.log 或 InstanceName_S2025.log
    # 策略: 提取 "_seed" 或 "_S" 之前的部分作为算例名
    instance_name = filename
    seed = "Unknown"
    
    seed_match = re.search(r'(_seed|_S)(\d+)', filename)
    if seed_match:
        instance_name = filename[:seed_match.start()]
        seed = seed_match.group(2)
    else:
        # 如果没有种子标记，去掉扩展名作为算例名
        instance_name = os.path.splitext(filename)[0]
        
    # 清理一下算例名 (防止包含路径斜杠)
    instance_name = os.path.basename(instance_name)

    # --- 2. 解析日志内容 ---
    # 模式 A: DRL/No-DRL 格式 (Step [x/N] ... ⏳ time)
    drl_pattern = re.compile(r"Step\s*\[\s*(\d+)/(\d+)\s*\].*?⏳\s*([\d\.]+)s")
    
    # 模式 B: VNS/Baseline 格式 (Iter: x ... Time: time)
    vns_pattern = re.compile(r"Iter:\s*(\d+).*?Time:\s*([\d\.]+)s")

    steps_data = []
    total_steps = 0
    is_drl_style = False

    for line in lines:
        # 尝试匹配 DRL 格式
        m_drl = drl_pattern.search(line)
        if m_drl:
            is_drl_style = True
            step = int(m_drl.group(1))
            total_steps = int(m_drl.group(2))
            time_val = float(m_drl.group(3))
            is_new_best = "NEW BEST" in line # DRL日志通常用这个标记
            
            steps_data.append({
                'step': step,
                'time': time_val,
                'is_new_best': is_new_best
            })
            continue
        
        # 尝试匹配 VNS 格式 (如果还没确认为 DRL)
        if not is_drl_style and "New Best" in line:
            m_vns = vns_pattern.search(line)
            if m_vns:
                step = int(m_vns.group(1))
                time_val = float(m_vns.group(2))
                steps_data.append({
                    'step': step,
                    'time': time_val,
                    'is_new_best': True # VNS 只打印 New Best 行
                })

    if not steps_data:
        return None

    # 如果是 VNS，steps_data 里只有更新最优解的记录，Total Steps 不明确
    # 我们假设最后一条记录是当前运行的最大步数
    if not is_drl_style:
        total_steps = steps_data[-1]['step'] if steps_data else 0

    # --- 3. 计算 Patience 时间 ---
    # 找到最后一次更新最优解的 Step
    best_step = 0
    for s in steps_data:
        if s['is_new_best']:
            best_step = s['step']
    
    # 计算 k
    k = get_adaptive_k(total_steps)
    target_step = best_step + k
    
    # 在数据中查找 Target Step 的时间
    # 如果实际步数没跑到 Target Step (比如早停了)，取最后一步的时间
    # 如果实际步数超过了 Target Step，取 Target Step 那一行的记录
    
    # 为了查找方便，我们先确定实际 Log 中最大的步数
    max_logged_step = steps_data[-1]['step']
    
    # 确定我们要找的截止步数
    cutoff_step = min(target_step, max_logged_step)
    
    # 查找时间 (找到 >= cutoff_step 的第一条记录)
    solving_time = None
    
    # 如果是 DRL，记录是密集的 (每一步都有)，直接找 step == cutoff_step
    # 如果是 VNS，记录是稀疏的 (只有 New Best)，我们只能取最后一条记录 (Time Limit) 
    # 或者取 > cutoff_step 的第一条记录? 
    # 根据之前的分析，VNS 跑满时间都不收敛，所以取最后一条是最合理的。
    
    if is_drl_style:
        # DRL 逻辑: 精确查找
        # 注意: steps_data 可能因为日志打印频率不是每一步都存
        # 我们找最接近且 >= cutoff_step 的
        for s in steps_data:
            if s['step'] >= cutoff_step:
                solving_time = s['time']
                break
        if solving_time is None:
             solving_time = steps_data[-1]['time']
    else:
        # VNS 逻辑: 通常是 Time Limit 截止，直接取日志最后一行的时间
        solving_time = steps_data[-1]['time']

    # 获取所属文件夹名作为算法标识 (Algorithm)
    folder_name = os.path.basename(os.path.dirname(file_path))
    if folder_name == LOG_DIR: # 如果直接在 batch_logs 下
        folder_name = "Root"

    return {
        'Instance': instance_name,
        'Algorithm': folder_name, # 子文件夹名作为算法名
        'Seed': seed,
        'SolvingTime': solving_time,
        'BestStep': best_step,
        'Patience_K': k
    }

def main():
    if not os.path.exists(LOG_DIR):
        print(f"错误: 找不到文件夹 '{LOG_DIR}'")
        return

    all_data = []
    print(f"正在扫描 '{LOG_DIR}' ...")

    # 递归遍历所有文件
    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith(".log") or file.endswith(".txt"):
                full_path = os.path.join(root, file)
                result = parse_log_file(full_path)
                if result:
                    all_data.append(result)
                    print(f"解析成功: {result['Instance']} ({result['Algorithm']}) -> Time: {result['SolvingTime']:.2f}s")

    if not all_data:
        print("未找到有效日志文件。")
        return

    # --- 生成 Excel ---
    df = pd.DataFrame(all_data)

    # 为了实现 "列为实验"，我们需要对 (Instance, Algorithm) 进行分组，然后生成 Exp_ID
    # 比如同一个算例跑了 5 个 Seed，我们标记为 Exp_1 到 Exp_5
    df.sort_values(by=['Instance', 'Algorithm', 'Seed'], inplace=True)
    df['Exp_ID'] = df.groupby(['Instance', 'Algorithm']).cumcount() + 1
    df['Exp_Label'] = "Exp_" + df['Exp_ID'].astype(str)

    # 1. 生成透视表 (Pivot Table)
    # 行: Instance
    # 列: Algorithm, Exp_Label
    # 值: SolvingTime
    pivot_df = df.pivot_table(
        index='Instance', 
        columns=['Algorithm', 'Exp_Label'], 
        values='SolvingTime'
    )

    # 2. 保存到 Excel
    # 我们生成两个 Sheet：一个是整理好的透视表，一个是原始数据（方便查证）
    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        pivot_df.to_excel(writer, sheet_name='Summary_Matrix')
        df.to_excel(writer, sheet_name='Raw_Data', index=False)

    print(f"\n✅ 提取完成！结果已保存至: {OUTPUT_FILE}")
    print(f"   - Sheet 'Summary_Matrix': 你的目标格式 (行=算例, 列=实验)")
    print(f"   - Sheet 'Raw_Data': 包含种子、Patience K值等详细信息")

if __name__ == "__main__":
    main()