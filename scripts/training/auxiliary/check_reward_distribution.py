import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import numpy as np
import os

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260116_210603.log"   # 你的日志文件
TARGET_GLOBAL_SHARE = 0.8          # 目标：希望全局突破奖励占总正向奖励的 80%
TIME_PENALTY_WEIGHT = 1.0          # 时间惩罚权重 (配置中的 TIME_PENALTY_FACTOR)
# ===========================================

def calibrate_all(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        return

    print(f"🔍 正在进行【三维联合校准】...")
    print(f"   目标 1: 确定 Global Scale (基准汇率)")
    print(f"   目标 2: 确定 Progress Scale (80/20 分配)")
    print(f"   目标 3: 确定 Clip Reward (99% 覆盖)")
    print("-" * 60)

    # 正则表达式
    new_episode_re = re.compile(r"New Episode:\s*切换算例")
    step_re = re.compile(r"Step\s*\[\s*(\d+)/.*⏳\s*([\d\.]+)\s*s")
    gen_best_re = re.compile(r"Max=([\-\d\.]+)")
    
    # 数据容器
    # 1. 用于计算 Global Scale 的效率列表 (Ratio/s)
    late_stage_ratios = [] 
    episode_ratios = []
    
    # 2. 用于计算 Progress 比例的累积和
    total_global_imp_sum = 0.0
    total_progress_imp_sum = 0.0
    
    # 3. 用于模拟回测的轨迹数据
    simulation_trace = [] # 存 {'time': t, 'global_imp': g, 'progress_imp': p}

    # 状态变量
    current_global_best = -1e9
    last_gen_best = -1e9
    episode_valid = False

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # --- 1. 切换算例 ---
            if new_episode_re.search(line):
                # 结算上一个 Episode 的 Ratio (取后半程)
                if episode_ratios:
                    mid = len(episode_ratios) // 2
                    late_stage_ratios.extend(episode_ratios[mid:])
                
                # 重置
                current_global_best = -1e9
                last_gen_best = -1e9
                episode_ratios = []
                episode_valid = False
                continue

            # --- 2. 解析 Step ---
            step_match = step_re.search(line)
            if not step_match: continue
            
            time_cost = float(step_match.group(2))
            
            # 提取本代最佳
            gen_match = gen_best_re.search(line)
            if not gen_match: continue
            current_gen_best = float(gen_match.group(1))

            # 初始化
            if current_global_best == -1e9:
                current_global_best = current_gen_best
                last_gen_best = current_gen_best
                episode_valid = True
                continue

            # --- 计算指标 ---
            base_global = abs(current_global_best) if abs(current_global_best) > 1e-6 else 1.0
            base_gen = abs(last_gen_best) if abs(last_gen_best) > 1e-6 else 1.0
            
            step_global_imp_pct = 0.0
            step_progress_imp_pct = 0.0
            
            # A. Global Improvement
            if current_gen_best > current_global_best:
                diff = current_gen_best - current_global_best
                step_global_imp_pct = diff / base_global
                
                # 记录 Ratio (用于 Scale 计算)
                if step_global_imp_pct > 1e-7 and time_cost > 0.001:
                    ratio = step_global_imp_pct / time_cost
                    episode_ratios.append(ratio)
                
                # 累加总和 (用于 Ratio 计算)
                total_global_imp_sum += step_global_imp_pct
                
                # 更新
                current_global_best = current_gen_best

            # B. Progress Improvement
            if current_gen_best > last_gen_best:
                diff = current_gen_best - last_gen_best
                if diff > 0:
                    step_progress_imp_pct = diff / base_gen
                    total_progress_imp_sum += step_progress_imp_pct
            
            last_gen_best = current_gen_best
            
            # C. 记录轨迹 (用于 Clip 计算)
            simulation_trace.append({
                'time': time_cost,
                'g_pct': step_global_imp_pct,
                'p_pct': step_progress_imp_pct
            })

    # 结算最后一段
    if episode_ratios:
        mid = len(episode_ratios) // 2
        late_stage_ratios.extend(episode_ratios[mid:])

    if not late_stage_ratios:
        print("❌ 数据不足，无法计算。")
        return

    # ================= 阶段 1: 计算 Global Scale =================
    print("\n[阶段 1] 计算 Global Scale (基于困难阶段 P20)")
    p20_ratio = np.percentile(late_stage_ratios, 20)
    # Scale = 1.0 / Ratio
    suggested_global_scale = 1.0 / max(p20_ratio, 1e-9)
    print(f"   - 困难阶段产出率 (P20): {p20_ratio:.2e} /s")
    print(f"   - 建议 REWARD_SCALE_GLOBAL_BEST: {suggested_global_scale:.1f}")

    # ================= 阶段 2: 计算 Progress Factor =================
    print("\n[阶段 2] 计算 Progress Scale (基于 80/20 贡献原则)")
    if total_progress_imp_sum == 0:
        raw_ratio = 1.0 # 避免除零
    else:
        raw_ratio = total_global_imp_sum / total_progress_imp_sum
    
    # 公式: Factor = Raw_Ratio * ( (1-Target)/Target )
    # Target=0.8 -> (0.2/0.8) = 0.25
    target_ratio = (1.0 - TARGET_GLOBAL_SHARE) / TARGET_GLOBAL_SHARE
    progress_factor = raw_ratio * target_ratio
    
    suggested_progress_scale = suggested_global_scale * progress_factor
    
    print(f"   - 原始产出总量比 (Global / Progress): {raw_ratio:.2f}")
    print(f"   - 建议 Progress 系数因子: {progress_factor:.3f}")
    print(f"   - 建议 REWARD_SCALE_PROGRESS: {suggested_progress_scale:.1f}")

    # ================= 阶段 3: 模拟回测 & 确定 Clip =================
    print(f"\n[阶段 3] 模拟奖励流 & 确定 Clip (Scale_G={suggested_global_scale:.0f}, Scale_P={suggested_progress_scale:.0f})")
    
    simulated_rewards = []
    
    for step in simulation_trace:
        # 模拟计算 Reward
        # Reward = (Global_Pct * Scale_G) + (Progress_Pct * Scale_P) - (Time * 1.0)
        r_g = step['g_pct'] * suggested_global_scale
        r_p = step['p_pct'] * suggested_progress_scale
        r_t = step['time'] * TIME_PENALTY_WEIGHT
        
        # 只有当总收益为正时，Clip 才有意义（我们不 Clip 惩罚）
        total_r = r_g + r_p - r_t
        simulated_rewards.append(total_r)

    # 统计分布
    rewards_np = np.array(simulated_rewards)
    # 只看正奖励 (因为 Clip 通常是限制最大值)
    positive_rewards = rewards_np[rewards_np > 0]
    
    if len(positive_rewards) == 0:
        print("   ❌ 模拟结果显示没有正向奖励，请检查数据。")
        return

    p90 = np.percentile(positive_rewards, 90)
    p95 = np.percentile(positive_rewards, 95)
    p99 = np.percentile(positive_rewards, 99)
    max_r = np.max(positive_rewards)
    
    print(f"   - 模拟正奖励分布:")
    print(f"     P90: {p90:.2f}")
    print(f"     P95: {p95:.2f}")
    print(f"     P99: {p99:.2f}  <-- 推荐锚点")
    print(f"     Max: {max_r:.2f}")
    
    # 推荐 Clip 值：取 P99 向上取整，且不小于 10.0 (SB3 默认值)
    suggested_clip = max(10.0, np.ceil(p99))
    
    print("-" * 60)
    print("✅ 最终推荐配置 (hyper_config.py & train_hyper.py):")
    print(f"1. REWARD_SCALE_GLOBAL_BEST = {suggested_global_scale:.1f}")
    print(f"2. REWARD_SCALE_PROGRESS    = {suggested_progress_scale:.1f}  (Global * {progress_factor:.3f})")
    print(f"3. clip_reward              = {suggested_clip:.1f}  (VecNormalize 参数)")

if __name__ == "__main__":
    try:
        calibrate_all(LOG_FILE_PATH)
    except ImportError:
        print("需要 numpy: pip install numpy")