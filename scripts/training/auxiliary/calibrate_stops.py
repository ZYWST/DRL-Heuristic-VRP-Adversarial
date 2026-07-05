import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import os
import pandas as pd
import numpy as np

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260116_210603.log"     # 你的训练日志
CURRENT_PATIENCE = 25              # 当前设定的早停步数
CURRENT_TIME_LIMIT = 300           # 当前设定的时间限制 (秒)
TOLERANCE_MISS_RATE = 0.01         # 你能容忍的最大损失率 (例如 1%)
# ===========================================

def check_constraints(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        return

    print(f"🔍 正在进行极限压力测试...")
    print(f"   基准参数: Patience={CURRENT_PATIENCE}, TimeLimit={CURRENT_TIME_LIMIT}s")
    print(f"   目标: 寻找维持 New Best 捕获率的最小参数边界。")
    print("-" * 60)

    # 1. 解析日志，提取轨迹
    episodes = parse_log(log_file)
    if not episodes:
        print("❌ 未提取到有效 Episode 数据。")
        return
    
    total_episodes = len(episodes)
    # 统计在当前基准 (25, 300) 下能捕获到的所有 Hits 总数
    # 注意：如果日志本身是按更宽松参数跑的，我们要先裁切到 (25, 300) 作为分母
    baseline_hits, baseline_time = simulate(episodes, CURRENT_PATIENCE, CURRENT_TIME_LIMIT)
    
    print(f"📊 数据概览:")
    print(f"   - 总 Episode 数: {total_episodes}")
    print(f"   - 在当前设定(25, 300)下的有效产出(Hits): {baseline_hits}")
    print(f"   - 在当前设定下的总耗时: {baseline_time/3600:.2f} 小时")
    print("-" * 60)

    # ================= 测试 1: 早停步数 (Patience) 紧缩测试 =================
    print("\n[测试 1] 早停步数 (Patience) 紧缩测试 (保持时间 300s 不变)")
    print(f"{'Patience':<10} | {'捕获 Hits':<10} | {'损失率':<10} | {'节省时间':<10} | {'评价'}")
    print("-" * 75)
    
    best_patience = CURRENT_PATIENCE
    
    # 从当前值往下探，直到 5
    for p in range(CURRENT_PATIENCE, 4, -1):
        hits, time_cost = simulate(episodes, p, CURRENT_TIME_LIMIT)
        missed = baseline_hits - hits
        miss_rate = missed / baseline_hits if baseline_hits > 0 else 0.0
        time_saved_pct = (baseline_time - time_cost) / baseline_time if baseline_time > 0 else 0.0
        
        status = ""
        if miss_rate <= 0.001: 
            status = "✅ 安全"
            best_patience = p
        elif miss_rate <= TOLERANCE_MISS_RATE:
            status = "⚠️ 可行"
            # 如果能容忍，也可以更新 best
            best_patience = p
        else:
            status = "❌ 损失过大"
        
        print(f"{p:<10} | {hits:<10} | {miss_rate*100:6.2f}%   | {time_saved_pct*100:6.1f}%   | {status}")

    # ================= 测试 2: 时间限制 (Time Limit) 紧缩测试 =================
    print("\n[测试 2] 时间限制 (Time Limit) 紧缩测试 (保持 Patience 25 不变)")
    print(f"{'TimeLimit':<10} | {'捕获 Hits':<10} | {'损失率':<10} | {'节省时间':<10} | {'评价'}")
    print("-" * 75)
    
    best_time = CURRENT_TIME_LIMIT
    
    # 从当前值往下探，步长 30秒
    # 生成测试点，确保包含 best_time
    test_points = sorted(list(set([t for t in range(CURRENT_TIME_LIMIT, 59, -30)] + [CURRENT_TIME_LIMIT])), reverse=True)
    
    for t in test_points:
        hits, time_cost = simulate(episodes, CURRENT_PATIENCE, t)
        missed = baseline_hits - hits
        miss_rate = missed / baseline_hits if baseline_hits > 0 else 0.0
        time_saved_pct = (baseline_time - time_cost) / baseline_time if baseline_time > 0 else 0.0
        
        status = ""
        if miss_rate <= 0.001: 
            status = "✅ 安全"
            best_time = t
        elif miss_rate <= TOLERANCE_MISS_RATE:
            status = "⚠️ 可行"
            best_time = t
        else:
            status = "❌ 损失过大"
            
        print(f"{t:<10} | {hits:<10} | {miss_rate*100:6.2f}%   | {time_saved_pct*100:6.1f}%   | {status}")

    # ================= 结论 =================
    print("-" * 60)
    print("💡 优化建议:")
    
    print(f"1. 早停步数 (Patience):")
    if best_patience < CURRENT_PATIENCE:
        print(f"   👉 可以从 {CURRENT_PATIENCE} 调低到 {best_patience}。")
        print(f"      (这说明大多数 New Best 都出现在连续 {best_patience} 步停滞以内)")
    else:
        print(f"   👉 保持 {CURRENT_PATIENCE}。调低会导致明显的收益损失。")
        
    print(f"\n2. 时间限制 (Time Limit):")
    if best_time < CURRENT_TIME_LIMIT:
        print(f"   👉 可以从 {CURRENT_TIME_LIMIT}s 调低到 {best_time}s。")
        print(f"      (这说明大算例通常在 {best_time}s 内就已经收敛)")
    else:
        print(f"   👉 保持 {CURRENT_TIME_LIMIT}s。你的大算例确实需要这么久。")

# --- 辅助函数 ---
def parse_log(log_file):
    episodes = []
    current_ep = {"steps": []}
    
    new_episode_re = re.compile(r"New Episode:\s*切换算例")
    step_re = re.compile(r"Step\s*\[\s*(\d+)/.*⏳\s*([\d\.]+)\s*s")
    new_best_re = re.compile(r"->\s*🎯NEW BEST")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if new_episode_re.search(line):
                if current_ep["steps"]: episodes.append(current_ep)
                current_ep = {"steps": []}
                continue
            
            step_match = step_re.search(line)
            if step_match:
                is_hit = bool(new_best_re.search(line))
                time_cost = float(step_match.group(2))
                current_ep["steps"].append({"time": time_cost, "hit": is_hit})
                
    if current_ep["steps"]: episodes.append(current_ep)
    return episodes

def simulate(episodes, patience_limit, time_limit_sec):
    total_hits = 0
    total_time = 0.0
    
    for ep in episodes:
        stag_count = 0
        ep_time = 0.0
        active = True
        
        for step in ep["steps"]:
            if not active: break # 如果已经早停，后续步骤全部跳过
            
            # 1. 时间检查
            if ep_time + step["time"] > time_limit_sec:
                active = False
                # 如果这一步超时了，我们假设这一步没跑完就被杀掉了，
                # 或者严格点：这一步的收益算不算？
                # 通常超时是硬截断，这一步的 Hit 拿不到
                break 
            
            ep_time += step["time"]
            
            # 2. 收益检查
            if step["hit"]:
                total_hits += 1
                stag_count = 0 # 重置忍耐
            else:
                stag_count += 1
            
            # 3. 早停检查
            if stag_count >= patience_limit:
                active = False
        
        total_time += ep_time
        
    return total_hits, total_time

if __name__ == "__main__":
    check_constraints(LOG_FILE_PATH)