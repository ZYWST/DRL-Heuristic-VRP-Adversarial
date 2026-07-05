import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import numpy as np
import os

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260116_210603.log"  # 你的日志文件
TARGET_GLOBAL_CONTRIBUTION = 0.75  # 我们希望“全局突破奖励”占总正奖励的 80%
# ===========================================

def calibrate_progress_ratio(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        return

    print(f"🔍 正在分析进步奖励比例 (目标: 全局突破占 {TARGET_GLOBAL_CONTRIBUTION*100:.0f}% 总收益)...")
    
    step_pattern = re.compile(r"Step\s*\[\s*(\d+)/")
    # 匹配 Fit: Avg=... / Max=20523.31
    # 这里的 Max 代表“本代最高分” (Generation Best)
    gen_best_pattern = re.compile(r"Max=([\-\d\.]+)")
    
    # 匹配 -> 🎯NEW BEST! 或者 (Best: 26321.80)
    # 这用于追踪 Global Best
    new_best_pattern = re.compile(r"Max=([\-\d\.]+)\s*->\s*.*NEW BEST")
    
    new_episode_pattern = re.compile(r"New Episode:\s*切换算例")

    # 累加器
    total_global_pct_sum = 0.0  # 全局提升百分比总和
    total_progress_pct_sum = 0.0 # 代际进步百分比总和
    
    # 状态变量
    current_global_best = -1e9
    last_gen_best = -1e9
    
    episode_global_sum = 0.0
    episode_progress_sum = 0.0
    
    valid_episodes = 0

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 切换算例
            if new_episode_pattern.search(line):
                # 结算上一个 Episode
                if current_global_best > -1e9:
                    total_global_pct_sum += episode_global_sum
                    total_progress_pct_sum += episode_progress_sum
                    valid_episodes += 1
                
                # 重置
                current_global_best = -1e9
                last_gen_best = -1e9
                episode_global_sum = 0.0
                episode_progress_sum = 0.0
                continue

            # 2. 解析 Step
            if not step_pattern.search(line):
                continue
            
            # 提取本代最佳 (Gen Best)
            gen_match = gen_best_pattern.search(line)
            if not gen_match: continue
            current_gen_best = float(gen_match.group(1))
            
            # 初始化
            if current_global_best == -1e9:
                current_global_best = current_gen_best
                last_gen_best = current_gen_best
                continue

            # --- A. 计算 Global Improvement ---
            # 检查是否是 New Best
            step_global_imp = 0.0
            if current_gen_best > current_global_best:
                base = abs(current_global_best) if abs(current_global_best) > 1e-6 else 1.0
                step_global_imp = (current_gen_best - current_global_best) / base
                if step_global_imp > 0:
                    episode_global_sum += step_global_imp
                current_global_best = current_gen_best # 更新 Global Best

            # --- B. 计算 Progress Improvement ---
            # 只要本代比上一代好，就算进步 (哪怕没破纪录)
            # 注意：hyper_env 里是 current_gen_best - last_gen_best
            step_progress_imp = 0.0
            if last_gen_best > -1e9:
                base = abs(last_gen_best) if abs(last_gen_best) > 1e-6 else 1.0
                diff = current_gen_best - last_gen_best
                if diff > 0:
                    pct = diff / base
                    step_progress_imp = pct
                    episode_progress_sum += step_progress_imp
            
            # 更新 last_gen_best
            last_gen_best = current_gen_best

    # 结算最后一个
    if current_global_best > -1e9:
        total_global_pct_sum += episode_global_sum
        total_progress_pct_sum += episode_progress_sum
        valid_episodes += 1

    # ================= 统计分析 =================
    print("-" * 60)
    print(f"📊 统计结果 (共 {valid_episodes} 个有效 Episode):")
    
    if total_progress_pct_sum == 0:
        print("❌ 数据不足：未检测到任何代际进步。")
        return

    print(f"1. 累积 Global 提升百分比: {total_global_pct_sum:.4f} (总共提升了这么多个点)")
    print(f"2. 累积 Progress 进步百分比: {total_progress_pct_sum:.4f} (含反复震荡的提升)")
    
    # 原始比例
    raw_ratio = total_global_pct_sum / total_progress_pct_sum
    print(f"3. 原始产出比 (Global / Progress): {raw_ratio:.2f}")
    print(f"   (这意味着：每产生 1% 的 Global 突破，伴随着 {1/raw_ratio:.2f}% 的代际波动提升)")

    print("-" * 60)
    print("💡 推荐配置推导:")
    
    # 公式推导：
    # Total_Reward_Global = Sum(Global) * Scale_Global
    # Total_Reward_Progress = Sum(Progress) * Scale_Progress
    # 我们希望: Total_Reward_Global / Total_Reward_Progress = 80 / 20 = 4.0
    # 即: (Sum(Global) * Scale_Global) / (Sum(Progress) * Scale_Progress) = 4.0
    # => Scale_Progress = (Sum(Global) / Sum(Progress)) * (Scale_Global / 4.0)
    # => Scale_Progress = raw_ratio * Scale_Global * (1-Target)/Target
    
    target_ratio = (1 - TARGET_GLOBAL_CONTRIBUTION) / TARGET_GLOBAL_CONTRIBUTION # 20/80 = 0.25
    
    suggested_factor = raw_ratio * target_ratio
    
    print(f"我们希望 Global 奖励占总奖励的 {TARGET_GLOBAL_CONTRIBUTION*100:.0f}%。")
    print(f"计算得出系数因子: {suggested_factor:.4f}")
    
    print(f"\n✅ 建议配置:")
    print(f"REWARD_SCALE_PROGRESS = REWARD_SCALE_GLOBAL_BEST * {suggested_factor:.3f}")
    
    if suggested_factor > 1.0:
        print("⚠️ 警告：Progress 累积量非常小，系数大于 1.0。")
        print("   这意味着代际提升非常罕见，Agent 每一步都在破纪录？请检查日志是否正常。")
    elif suggested_factor < 0.1:
        print("⚠️ 警告：Progress 累积量巨大 (震荡严重)，系数小于 0.1。")
        print("   必须使用这个小系数，否则 Progress 奖励会彻底淹没 Global 奖励！")

if __name__ == "__main__":
    calibrate_progress_ratio(LOG_FILE_PATH)