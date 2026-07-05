import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import numpy as np
import os
from collections import defaultdict

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260112_194146.log"   # 你的日志文件
# 模拟的参数网格
PATIENCE_OPTIONS = [20, 25, 30, 35, 40, 50, 60]  # 早停步数候选项
TIME_LIMIT_OPTIONS = [300, 400, 500, 600, 800]   # 最大时间候选项
# ===========================================

def parse_log_for_simulation(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 未找到日志文件: {log_file}")
        return []

    print(f"🔍 正在深度解析日志以进行模拟...")
    
    episodes = []
    current_ep = {
        "name": "Unknown",
        "steps": [], # 存每一步的耗时和是否NewBest
        "total_time": 0.0,
        "raw_lines": []
    }
    
    # 正则
    new_episode_re = re.compile(r"New Episode:\s*切换算例\s*->\s*(.*\.dat)")
    step_re = re.compile(r"Step\s*\[\s*(\d+)/.*⏳\s*([\d\.]+)\s*s")
    new_best_re = re.compile(r"->\s*🎯NEW BEST")
    
    current_best_step_idx = 0
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 新 Episode
            ep_match = new_episode_re.search(line)
            if ep_match:
                if current_ep["steps"]:
                    episodes.append(current_ep)
                
                current_ep = {
                    "name": ep_match.group(1).strip(),
                    "steps": [],
                    "total_time": 0.0
                }
                current_best_step_idx = 0
                continue
            
            # 2. 解析 Step
            step_match = step_re.search(line)
            if step_match:
                step_idx = int(step_match.group(1))
                time_cost = float(step_match.group(2))
                
                is_new_best = bool(new_best_re.search(line))
                
                # 记录这一步的信息
                step_info = {
                    "idx": step_idx,
                    "time_cost": time_cost,
                    "is_new_best": is_new_best,
                    "timestamp": current_ep["total_time"] + time_cost # 这一步结束时的累计时间
                }
                
                current_ep["steps"].append(step_info)
                current_ep["total_time"] += time_cost
                
                if is_new_best:
                    current_best_step_idx = step_idx

    # 添加最后一个
    if current_ep["steps"]:
        episodes.append(current_ep)
        
    print(f"✅ 解析完成，共提取 {len(episodes)} 个完整 Episode 轨迹。")
    return episodes

def simulate_performance(episodes, patience, time_limit):
    """
    模拟在特定 patience 和 time_limit 下的表现
    """
    total_sim_time = 0.0
    total_hits = 0
    missed_hits = 0
    
    for ep in episodes:
        steps_since_last_best = 0
        ep_sim_time = 0.0
        active = True
        
        for step in ep["steps"]:
            # 1. 累加时间
            ep_sim_time += step["time_cost"]
            
            # 2. 检查是否超时
            if ep_sim_time > time_limit:
                active = False
            
            # 3. 检查是否产生 New Best
            if step["is_new_best"]:
                if active:
                    total_hits += 1
                    steps_since_last_best = 0 # 重置忍耐度
                else:
                    missed_hits += 1 # 本来能产生，但被我们因为时间限制截断了
            else:
                if active:
                    steps_since_last_best += 1
            
            # 4. 检查是否触发早停
            if active and steps_since_last_best >= patience:
                active = False
                # 注意：这里我们假设触发早停后，Episode 立即结束，不再产生后续时间消耗
                # 所以循环继续只是为了统计 missed_hits，但 ep_sim_time 不应该再增加了
                # 为了简化逻辑，我们用 flag 控制

            if not active:
                # 如果已经因为某种原因结束了，剩下的步骤都是 "Missed Opportunity" 的检查
                # 且时间不再累加
                pass
            else:
                # 还在运行中，时间有效
                total_sim_time += step["time_cost"]
        
        # 修正：total_sim_time 不能超过 time_limit (针对单步极长的情况)
        # 但这里是累加所有 episode，只要上面逻辑对即可

    # 计算指标
    # 效率 = 产生的有效 Hits / 总耗时 (Hits per Minute)
    efficiency = (total_hits / total_sim_time * 60) if total_sim_time > 0 else 0
    miss_rate = missed_hits / (total_hits + missed_hits) if (total_hits + missed_hits) > 0 else 0
    
    return efficiency, miss_rate, total_sim_time

def optimize_parameters(episodes):
    print("\n📊 --- 参数网格搜索 (Grid Search) ---")
    print(f"{'Patience':<8} | {'MaxTime':<8} | {'效率 (Hits/min)':<15} | {'损失率 (Miss%)':<15} | {'总耗时(h)':<10}")
    print("-" * 75)
    
    best_config = None
    best_score = -1
    
    results = []

    for p in PATIENCE_OPTIONS:
        for t in TIME_LIMIT_OPTIONS:
            eff, miss, sim_time_s = simulate_performance(episodes, p, t)
            sim_time_h = sim_time_s / 3600.0
            
            # 评分标准：效率优先，但损失率不能太高 (>5% 就要扣分)
            # Score = Efficiency * (1 - Penalty)
            penalty = 0.0
            if miss > 0.05: penalty = (miss - 0.05) * 5 # 惩罚稍微严厉一点
            if miss > 0.20: penalty = 100 # 损失超过 20% 直接枪毙
            
            score = eff * (1.0 - penalty)
            
            results.append((p, t, eff, miss, sim_time_h, score))
            
            if score > best_score:
                best_score = score
                best_config = (p, t)

    # 排序输出前 10
    results.sort(key=lambda x: x[5], reverse=True)
    
    for res in results[:10]:
        p, t, eff, miss, time_h, score = res
        mark = "⭐" if (p,t) == best_config else ""
        print(f"{p:<8} | {t:<8} | {eff:<15.2f} | {miss*100:<14.2f}% | {time_h:<10.1f} {mark}")

    return best_config

def filter_bad_instances(episodes, best_p, best_t):
    print(f"\n🗑️ --- 算例优胜劣汰 (基于最佳参数 P={best_p}, T={best_t}) ---")
    
    instance_stats = defaultdict(lambda: {"hits": 0, "time": 0.0, "steps": 0})
    
    for ep in episodes:
        steps_since_last = 0
        curr_time = 0.0
        active = True
        
        for step in ep["steps"]:
            curr_time += step["time_cost"]
            if curr_time > best_t: active = False
            
            if step["is_new_best"]:
                if active:
                    instance_stats[ep["name"]]["hits"] += 1
                    steps_since_last = 0
            else:
                if active: steps_since_last += 1
            
            if active and steps_since_last >= best_p:
                active = False
            
            if active:
                instance_stats[ep["name"]]["time"] += step["time_cost"]
                instance_stats[ep["name"]]["steps"] += 1

    # 排名
    ranking = []
    for name, stats in instance_stats.items():
        time_min = stats["time"] / 60.0
        hits = stats["hits"]
        # 效率：Hits / Min
        eff = hits / time_min if time_min > 0 else 0
        # 步均耗时
        avg_step_time = stats["time"] / stats["steps"] if stats["steps"] > 0 else 0
        
        ranking.append({
            "name": name,
            "eff": eff,
            "hits": hits,
            "time_min": time_min,
            "step_time": avg_step_time
        })
    
    ranking.sort(key=lambda x: x["eff"], reverse=True)
    
    print(f"{'排名':<4} | {'算例名称':<30} | {'效率(Hits/min)':<12} | {'总产出Hits':<10} | {'步均耗时(s)':<10}")
    print("-" * 80)
    
    remove_list = []
    for i, r in enumerate(ranking):
        # 标记低效算例
        # 标准：效率低于平均值的 1/3，或者步均耗时巨大且产出极低
        status = ""
        if r["eff"] < 0.5: # 每2分钟都产不出一个新解
            status = "❌ 建议移除"
            remove_list.append(r["name"])
        elif r["step_time"] > 8.0 and r["eff"] < 2.0: # 又慢又笨
            status = "⚠️ 观察"
        
        print(f"{i+1:<4} | {r['name']:<30} | {r['eff']:<12.2f} | {r['hits']:<10} | {r['step_time']:<10.2f} {status}")

    print("\n💡 优化建议:")
    print(f"1. 将 EARLY_STOPPING_STABLE_STEPS 设为: {best_p}")
    print(f"2. 将 MAX_SECONDS_PER_EPISODE 设为: {best_t}")
    print(f"3. 从 EVAL_PROBLEM_LIST 和 训练文件夹 中移除以下 {len(remove_list)} 个算例:")
    for bad in remove_list:
        print(f"   - {bad}")

if __name__ == "__main__":
    episodes = parse_log_for_simulation(LOG_FILE_PATH)
    if episodes:
        best_p, best_t = optimize_parameters(episodes)
        filter_bad_instances(episodes, best_p, best_t)