import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import os
import numpy as np
from collections import defaultdict

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260116_210603.log"   # 你的日志文件
MIN_EPISODE_LENGTH = 10          # 忽略过短的 Episode
# 权重配置 (你可以根据偏好调整，目前是 5:5 开)
WEIGHT_ENDURANCE = 0.5           # 耐力权重 (持久度)
WEIGHT_HIT_RATE = 0.5            # 活跃权重 (爆率)
# ===========================================

def find_hybrid_instances(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        return

    print(f"🔍 正在进行【双维综合评估】...")
    print(f"   - 维度1: 耐力 (Endurance) - 占比 {WEIGHT_ENDURANCE*100}%")
    print(f"   - 维度2: 爆率 (Hit Rate)  - 占比 {WEIGHT_HIT_RATE*100}%")
    print(f"   - 🚫 已剔除 Fitness 绝对数值，仅评估相对优化行为")
    print("-" * 60)

    # 正则表达式
    episode_pattern = re.compile(r"New Episode:\s*切换算例\s*->\s*(.*\.dat)")
    step_pattern = re.compile(r"Step\s*\[\s*(\d+)/")
    # 匹配 NEW BEST (不管数值大小，只看有没有)
    new_best_pattern = re.compile(r"->\s*🎯NEW BEST")

    problem_data = defaultdict(list)
    
    current_problem = None
    current_ep_stat = {
        "total_steps": 0,
        "last_update_step": 0,
        "hits": 0
    }

    # --- 1. 解析日志 ---
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 切换算例
            ep_match = episode_pattern.search(line)
            if ep_match:
                if current_problem and current_ep_stat["total_steps"] >= MIN_EPISODE_LENGTH:
                    problem_data[current_problem].append(current_ep_stat)
                
                current_problem = ep_match.group(1).strip()
                current_ep_stat = {"total_steps": 0, "last_update_step": 0, "hits": 0}
                continue

            # 解析 Step
            step_match = step_pattern.search(line)
            if step_match:
                step = int(step_match.group(1))
                current_ep_stat["total_steps"] = step
                
                # 检查是否产生新解
                if new_best_pattern.search(line):
                    current_ep_stat["last_update_step"] = step
                    # 通常 Step 1 是初始化，不算“爆率” (除非你认为初始化好也算)
                    # 建议：如果 step > 1 才算 hit，更能反映 Agent 的努力
                    if step > 1:
                        current_ep_stat["hits"] += 1

    # 结算最后一个
    if current_problem and current_ep_stat["total_steps"] >= MIN_EPISODE_LENGTH:
        problem_data[current_problem].append(current_ep_stat)

    # --- 2. 计算原始指标 ---
    raw_stats = []
    
    for p_name, episodes in problem_data.items():
        if not episodes:
            continue
            
        # 计算平均耐力 (0.0 - 1.0)
        endurance_list = [e["last_update_step"] / e["total_steps"] for e in episodes]
        avg_endurance = np.mean(endurance_list)
        
        # 计算平均爆率 (0.0 - 1.0)
        hit_rate_list = [e["hits"] / e["total_steps"] for e in episodes]
        avg_hit_rate = np.mean(hit_rate_list)
        
        raw_stats.append({
            "name": p_name,
            "raw_endurance": avg_endurance,
            "raw_hit_rate": avg_hit_rate,
            "count": len(episodes)
        })

    if not raw_stats:
        print("❌ 数据不足，无法分析。")
        return

    # --- 3. 归一化 (Min-Max Scaling) ---
    # 找出最大最小值
    endurances = [x["raw_endurance"] for x in raw_stats]
    hit_rates = [x["raw_hit_rate"] for x in raw_stats]
    
    min_e, max_e = min(endurances), max(endurances)
    min_h, max_h = min(hit_rates), max(hit_rates)
    
    # 防止除以零
    range_e = max_e - min_e if max_e > min_e else 1.0
    range_h = max_h - min_h if max_h > min_h else 1.0

    # --- 4. 计算综合得分 ---
    final_ranking = []
    for item in raw_stats:
        # 归一化得分 (0-100分制)
        score_e = (item["raw_endurance"] - min_e) / range_e
        score_h = (item["raw_hit_rate"] - min_h) / range_h
        
        # 加权综合分
        hybrid_score = (score_e * WEIGHT_ENDURANCE + score_h * WEIGHT_HIT_RATE) * 100
        
        item["norm_score"] = hybrid_score
        item["norm_endurance"] = score_e * 100
        item["norm_hit_rate"] = score_h * 100
        final_ranking.append(item)

    # 按综合分降序
    final_ranking.sort(key=lambda x: x["norm_score"], reverse=True)

    # --- 5. 输出报告 ---
    print(f"{'排名':<4} | {'算例名称':<40} | {'综合分':<6} | {'耐力(%)':<8} | {'爆率(%)':<8}")
    print("-" * 90)

    for i, item in enumerate(final_ranking[:15]):
        # 原始数据用于展示
        display_endurance = item["raw_endurance"] * 100
        display_hit_rate = item["raw_hit_rate"] * 100
        
        print(f"{i+1:<4} | {item['name']:<40} | {item['norm_score']:<6.1f} | {display_endurance:<8.1f} | {display_hit_rate:<8.2f}")

    print("-" * 90)
    print("💡 选品指南：")
    print("1. **综合分** 最高的算例，意味着它既能在后期持续更新，又有密集的奖励反馈。")
    print("2. 建议直接选取 **Top 3** 作为评估算例。")
    print(f"3. 推荐列表: {[x['name'] for x in final_ranking[:3]]}")

if __name__ == "__main__":
    find_hybrid_instances(LOG_FILE_PATH)