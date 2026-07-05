import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import numpy as np
import os

# ================= 配置区域 =================
LOG_FILE_PATH = "training.log"  # 请将你的日志保存为这个文件
# ===========================================

def parse_log_and_calibrate_percentage(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        print("请将终端打印的训练日志复制并保存为同名文本文件。")
        return

    print(f"🔍 正在分析日志文件 (百分比收益模式): {log_file} ...")
    
    # 正则表达式定义 (保持不变)
    step_pattern = re.compile(r"Step\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]\s*\|\s*⏳\s*([\d\.]+)\s*s")
    new_best_pattern = re.compile(r"Max=([\-\d\.]+)\s*->\s*.*NEW BEST")
    existing_best_pattern = re.compile(r"->\s*\(Best:\s*([\-\d\.]+)\)")
    new_episode_pattern = re.compile(r"New Episode:\s*切换算例")

    # 数据存储
    all_valid_ratios = []       # 所有有效提升的性价比 (百分比/秒)
    late_stage_ratios = []      # 后半程的性价比
    
    # 状态变量
    current_global_best = -1e9
    episode_ratios = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # 1. 检查是否切换了算例
        if new_episode_pattern.search(line):
            # 结算上一个 Episode
            if episode_ratios:
                # 定义后半程：保留后 75% (跳过前 25%)
                mid = len(episode_ratios) // 4
                late_stage_ratios.extend(episode_ratios[mid:])
                all_valid_ratios.extend(episode_ratios)
            
            # 重置状态
            current_global_best = -1e9
            episode_ratios = []
            continue

        # 2. 提取 Step 基本信息
        step_match = step_pattern.search(line)
        if not step_match:
            continue
            
        time_cost = float(step_match.group(3))
        
        # 3. 提取当前 Global Best
        best_match = new_best_pattern.search(line)
        step_best_val = None
        
        if best_match:
            step_best_val = float(best_match.group(1))
        else:
            best_match_2 = existing_best_pattern.search(line)
            if best_match_2:
                step_best_val = float(best_match_2.group(1))
            else:
                continue 
        
        # 4. [核心逻辑修改] 计算百分比性价比
        if current_global_best == -1e9:
            current_global_best = step_best_val
            continue
            
        # 计算基准值 (防止分母为0)
        base_fitness = abs(current_global_best) if abs(current_global_best) > 1e-6 else 1.0
        
        # 计算绝对提升
        improvement_val = step_best_val - current_global_best
        
        # 计算相对提升 (Percentage Improvement, 0.01 代表 1%)
        pct_imp = improvement_val / base_fitness
        
        # 只有产生了正向提升，且耗时正常，才计算汇率
        # 注意：对于百分比，阈值设小一点 (1e-7)，因为后期提升可能只有 0.001%
        if pct_imp > 1e-7 and time_cost > 0.001:
            # Ratio = 每秒提升的百分比份额
            # 例如：0.0001 (0.01%) / 2s = 0.00005/s
            ratio = pct_imp / time_cost
            episode_ratios.append(ratio)
            # 调试打印 (可选)
            # print(f"   提升: {pct_imp*100:.4f}% | 耗时: {time_cost}s | 效率: {ratio:.6f}/s")
        
        # 更新当前最优
        if step_best_val > current_global_best:
            current_global_best = step_best_val

    # 处理最后一个 Episode
    if episode_ratios:
        mid = len(episode_ratios) // 4
        late_stage_ratios.extend(episode_ratios[mid:])
        all_valid_ratios.extend(episode_ratios)

    # ================= 统计分析 =================
    print("-" * 60)
    print(f"📊 日志分析完成 (百分比收益模式)")
    print(f"   - 捕获有效提升次数: {len(all_valid_ratios)}")
    print(f"   - 捕获后期(困难)提升: {len(late_stage_ratios)}")
    
    if not all_valid_ratios:
        print("❌ 警告：日志中未发现任何有效的 Fitness 提升记录。")
        return

    # 计算分位数
    avg_ratio = np.mean(all_valid_ratios)
    p50_total = np.percentile(all_valid_ratios, 50)
    
    target_source = late_stage_ratios if late_stage_ratios else all_valid_ratios
    p50_late = np.percentile(target_source, 50)
    p20_late = np.percentile(target_source, 20) # 保守底线

    print("-" * 60)
    print(f"全阶段平均产出: {avg_ratio:.6f} /s (约 {avg_ratio*100:.4f}%/s)")
    print(f"困难阶段中位产出: {p50_late:.6f} /s (约 {p50_late*100:.4f}%/s)")
    print(f"困难阶段 P20产出: {p20_late:.6f} /s (约 {p20_late*100:.4f}%/s) <--- 锚点")
    
    print("-" * 60)
    print("💡 推荐配置")
    
    # 逻辑：Reward = Pct_Imp * Scale - Time
    # 平衡点：Scale = 1.0 / P20_Ratio
    # 假设 P20 是 0.0001 (0.01%/s)，则 Scale = 10000
    
    recommended_scale = 1.0 / max(p20_late, 1e-9)
    
    print(f"# 请直接修改 hyper_config.py:")
    print(f"REWARD_SCALE_GLOBAL_BEST = {recommended_scale:.1f}")
    print(f"REWARD_SCALE_PROGRESS = {recommended_scale * 0.5:.1f}  # 建议设为 Global 的一半")
    print(f"# 注意：FITNESS_TO_TIME_EXCHANGE_RATE 这个变量在百分比模式下已失效，请直接用上面的 Scale")

if __name__ == "__main__":
    try:
        import numpy
        parse_log_and_calibrate_percentage(LOG_FILE_PATH)
    except ImportError:
        print("请先安装 numpy: pip install numpy")