import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import numpy as np
import os
import matplotlib.pyplot as plt # 如果没有图形界面，脚本会打印文本统计
from collections import defaultdict

# ================= 配置 =================
LOG_FILE = "logs/train_hyper_20260116_210603.log"   # 你的日志文件
# =======================================

def mine_training_insights(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 找不到文件: {log_file}")
        return

    print(f"⛏️ 正在深挖日志: {log_file} ...")

    # --- 正则定义 ---
    # 匹配: 🎮 Act: [Pop=16, α=0.20, SA_Len=100]
    act_pattern = re.compile(r"Act:\s*\[Pop=\s*(\d+),\s*α=([\d\.]+),\s*SA_Len=\s*(\d+)\]")
    
    # 匹配: Step [ 100/100000] ... Fit: Avg=... / Max=...
    step_pattern = re.compile(r"Step\s*\[\s*(\d+)/")
    
    # 匹配: 🎯NEW BEST!
    new_best_pattern = re.compile(r"🎯NEW BEST")
    
    # 匹配: 切换算例
    episode_pattern = re.compile(r"New Episode:\s*切换算例\s*->\s*(.*\.dat)")

    # --- 数据容器 ---
    data = {
        "pops": [], "alphas": [], "sa_lens": [], "steps": [],
        "is_new_best": [], "episode_names": []
    }
    
    # 临时变量
    current_ep_name = "Unknown"
    ep_start_step = 0
    steps_since_best = 0
    
    # 行为关联分析容器
    stagnation_actions = defaultdict(list) # {停滞步数: [alpha, ...]}
    time_actions = defaultdict(list)       # {进度%: [sa_len, ...]}

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    
    for i, line in enumerate(lines):
        # 1. 算例切换
        ep_match = episode_pattern.search(line)
        if ep_match:
            current_ep_name = ep_match.group(1).strip()
            steps_since_best = 0
            ep_start_step = 0 # 这里无法精确获知绝对步数，只能用相对推测
            continue

        # 2. 动作提取
        act_match = act_pattern.search(line)
        step_match = step_pattern.search(line)
        
        if act_match and step_match:
            step_idx = int(step_match.group(1))
            pop = int(act_match.group(1))
            alpha = float(act_match.group(2))
            sa_len = int(act_match.group(3))
            
            is_best = bool(new_best_pattern.search(line))
            
            # 记录基础数据
            data["pops"].append(pop)
            data["alphas"].append(alpha)
            data["sa_lens"].append(sa_len)
            data["steps"].append(step_idx)
            data["is_new_best"].append(is_best)
            data["episode_names"].append(current_ep_name)
            
            # 更新状态逻辑 (模拟 Agent 的视角)
            if is_best:
                steps_since_best = 0
            else:
                steps_since_best += 1
            
            # 记录“停滞-动作”关联
            # 我们将停滞步数分桶: 0-5, 6-10, ...
            stag_bucket = (steps_since_best // 5) * 5
            stagnation_actions[stag_bucket].append(alpha)

    # ================= 分析报告 =================
    
    total_steps = len(data["steps"])
    print(f"✅ 已提取 {total_steps} 步有效数据。")
    print("-" * 60)

    # --- 1. 动作边界分析 (Action Saturation) ---
    print("📊 1. 动作空间饱和度分析 (检查是否触碰边界):")
    
    def analyze_bound(name, values):
        v_min, v_max = np.min(values), np.max(values)
        p05, p95 = np.percentile(values, 5), np.percentile(values, 95)
        mean = np.mean(values)
        print(f"   - {name}: 范围 [{v_min:.2f}, {v_max:.2f}], 均值 {mean:.2f}")
        
        # 警告逻辑
        msg = []
        if p95 >= v_max * 0.99: msg.append(f"⚠️ 经常顶满上限 ({p95:.2f}) -> 建议调高上限！")
        if p05 <= v_min * 1.01 and v_min > 0: msg.append(f"⚠️ 经常触底下限 ({p05:.2f}) -> 建议调低下限！")
        if not msg: msg.append("✅ 分布健康，未触碰边界。")
        for m in msg: print(f"     {m}")

    analyze_bound("Pop Size", data["pops"])
    analyze_bound("Alpha   ", data["alphas"])
    analyze_bound("SA Len  ", data["sa_lens"])

    print("-" * 60)
    
    # --- 2. 策略逻辑验证 (Policy Logic) ---
    print("🧠 2. 智能体逻辑验证 (它学会了什么?):")
    
    # 分析：由于停滞 (Stagnation)，Alpha 是否变大？
    print("   [假设验证]：当陷入停滞时，Agent 是否学会了增大 Alpha 以跳出局部最优？")
    sorted_buckets = sorted([k for k in stagnation_actions.keys() if len(stagnation_actions[k]) > 50])
    
    print(f"   {'停滞步数':<10} | {'平均 Alpha':<10} | {'行为趋势'}")
    prev_alpha = -1
    trend_msg = "无明显规律"
    
    valid_buckets = []
    avg_alphas = []
    
    for b in sorted_buckets[:8]: # 只看前 40 步 (因为一般早停也就 40)
        vals = stagnation_actions[b]
        avg_a = np.mean(vals)
        valid_buckets.append(b)
        avg_alphas.append(avg_a)
        
        trend = "→"
        if prev_alpha != -1:
            if avg_a > prev_alpha * 1.05: trend = "↗️ (升)"
            elif avg_a < prev_alpha * 0.95: trend = "↘️ (降)"
        
        print(f"   {b}-{b+5:<8} | {avg_a:.4f}     | {trend}")
        prev_alpha = avg_a

    # 简单的线性回归判断趋势
    if len(avg_alphas) > 2:
        z = np.polyfit(valid_buckets, avg_alphas, 1)
        slope = z[0]
        if slope > 0.0005: print("\n   ✅ 结论：Agent 学会了！停滞越久，Alpha 越大。")
        elif slope < -0.0005: print("\n   ❌ 结论：Agent 行为反常，停滞时反而变得保守。")
        else: print("\n   ⚠️ 结论：Agent 似乎不在乎停滞，Alpha 保持恒定。可能需要加强停滞惩罚。")

    print("-" * 60)

    # --- 3. 算例教学质量 (Instance Profiling) ---
    print("🏫 3. 算例教学质量榜 (谁贡献了最多的梯度?):")
    
    instance_stats = defaultdict(lambda: {"steps": 0, "new_bests": 0})
    for i, name in enumerate(data["episode_names"]):
        instance_stats[name]["steps"] += 1
        if data["is_new_best"][i]:
            instance_stats[name]["new_bests"] += 1
            
    ranking = []
    for name, s in instance_stats.items():
        hit_rate = (s["new_bests"] / s["steps"]) * 100 if s["steps"] > 0 else 0
        ranking.append((name, hit_rate, s["steps"]))
        
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   {'算例名称 (Top 5 & Bottom 3)':<35} | {'爆率 (Hit%)':<12} | {'样本数'}")
    for item in ranking[:5]:
        print(f"   {item[0]:<35} | {item[1]:.2f}%       | {item[2]}")
    print("   ...")
    for item in ranking[-3:]:
        print(f"   {item[0]:<35} | {item[1]:.2f}%       | {item[2]}")
        
    print("\n   💡 建议：")
    print("   - 移除爆率 < 0.5% 的算例（纯噪音，浪费时间）。")
    print("   - 增加高爆率算例的采样权重（如果环境支持）。")

if __name__ == "__main__":
    try:
        import numpy
        mine_training_insights(LOG_FILE)
    except ImportError:
        print("请安装 numpy: pip install numpy")