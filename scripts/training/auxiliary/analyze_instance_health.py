import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import numpy as np
import os
import pandas as pd
from collections import defaultdict

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260116_210603.log"  # 你的日志文件
# ===========================================

def analyze_instance_health(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        return

    print(f"🔍 正在对算例进行【全维度健康体检】...")
    print("-" * 60)

    # 正则表达式
    # 匹配算例切换: New Episode: 切换算例 -> 01Case_....dat
    new_episode_re = re.compile(r"New Episode:\s*切换算例\s*->\s*(.*\.dat)")
    
    # 匹配 Step 行: Step [ 30/200] | ...
    step_re = re.compile(r"Step\s*\[\s*(\d+)/")
    
    # 匹配 New Best: ... -> 🎯NEW BEST!
    new_best_mark_re = re.compile(r"->\s*🎯NEW BEST")
    
    # 匹配 Fitness 数值 (兼容 NEW BEST 和普通 Best)
    # 格式 1: Max=1234.56 -> 🎯NEW BEST
    # 格式 2: -> (Best: 1234.56)
    fit_val_re1 = re.compile(r"Max=([\-\d\.]+)")
    fit_val_re2 = re.compile(r"\(Best:\s*([\-\d\.]+)\)")

    # 数据存储结构
    # instances[name] = [ {ep_stats}, {ep_stats} ... ]
    instances = defaultdict(list)
    
    # 当前 Episode 上下文
    current_ep = None
    
    def init_ep_stats(name):
        return {
            "name": name,
            "start_fit": None,   # 初始分 (第1步的分数)
            "final_fit": None,   # 结束分
            "total_steps": 0,    # 总步数
            "update_count": 0,   # 更新次数
            "last_update_step": 0 # 最后一次更新的步数
        }

    # --- 1. 解析日志 ---
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 检查算例切换
            ep_match = new_episode_re.search(line)
            if ep_match:
                # 结算上一个
                if current_ep and current_ep["total_steps"] > 0:
                    instances[current_ep["name"]].append(current_ep)
                
                # 开新局
                name = ep_match.group(1).strip()
                current_ep = init_ep_stats(name)
                continue

            # 检查 Step 信息
            step_match = step_re.search(line)
            if step_match and current_ep:
                step_idx = int(step_match.group(1))
                current_ep["total_steps"] = step_idx
                
                # 提取当前 Fitness
                # 先看有没有 Max=...
                val_match = fit_val_re1.search(line)
                curr_fit = None
                if val_match:
                    curr_fit = float(val_match.group(1))
                else:
                    # 再看有没有 (Best: ...)
                    val_match2 = fit_val_re2.search(line)
                    if val_match2:
                        curr_fit = float(val_match2.group(1))
                
                if curr_fit is not None:
                    # 记录初始分
                    if current_ep["start_fit"] is None:
                        current_ep["start_fit"] = curr_fit
                        # 第一步算不算更新？通常不算“挖掘”的更新，算基准
                        current_ep["final_fit"] = curr_fit
                        current_ep["last_update_step"] = step_idx
                    else:
                        # 检查是否提升
                        if curr_fit > current_ep["final_fit"]:
                            current_ep["final_fit"] = curr_fit
                            current_ep["update_count"] += 1
                            current_ep["last_update_step"] = step_idx
    
    # 结算最后一个
    if current_ep and current_ep["total_steps"] > 0:
        instances[current_ep["name"]].append(current_ep)

    # --- 2. 统计分析 ---
    results = []
    
    for name, ep_list in instances.items():
        if not ep_list: continue
        
        n_eps = len(ep_list)
        
        # 提取数组
        update_counts = [e["update_count"] for e in ep_list]
        total_steps = [e["total_steps"] for e in ep_list]
        
        # 计算提升幅度 (Final - Start) / Start
        improvements = []
        for e in ep_list:
            base = abs(e["start_fit"]) if abs(e["start_fit"]) > 1e-6 else 1.0
            imp = (e["final_fit"] - e["start_fit"]) / base
            improvements.append(imp)
            
        # 计算耐力 (Last Update Step / Total Step)
        endurances = []
        for e in ep_list:
            if e["total_steps"] > 0:
                ratio = e["last_update_step"] / e["total_steps"]
                endurances.append(ratio)
            else:
                endurances.append(0.0)

        # 聚合
        res = {
            "name": name,
            "n_eps": n_eps,
            "avg_updates": np.mean(update_counts), # 平均更新次数
            "avg_steps": np.mean(total_steps),     # 平均总步数
            "avg_imp_pct": np.mean(improvements) * 100, # 平均提升百分比
            "avg_endurance": np.mean(endurances) * 100, # 平均最后更新位置(%)
            "late_bloomer_rate": np.mean([1 if x > 0.8 else 0 for x in endurances]) * 100 # 后劲率: >80% 还在更新的比例
        }
        results.append(res)

    # --- 3. 排序与展示 ---
    # 按“平均提升幅度”降序排列 (含金量最高的排前面)
    results.sort(key=lambda x: x["avg_imp_pct"], reverse=True)

    # 打印表格
    print(f"{'算例名称 (Top 15)':<35} | {'Ep数量':<6} | {'更新次数':<8} | {'提升幅度':<10} | {'耐力(最后更新)':<14} | {'总步数':<6}")
    print("-" * 100)
    
    for r in results[:19]:
        name_display = r['name']
        if len(name_display) > 33: name_display = name_display[:30] + "..."
        
        print(f"{name_display:<35} | {r['n_eps']:<6} | {r['avg_updates']:<8.1f} | {r['avg_imp_pct']:<9.2f}% | {r['avg_endurance']:<5.1f}% (L:{r['late_bloomer_rate']:.0f}%) | {r['avg_steps']:<6.1f}")

    print("-" * 100)
    print("💡 指标解读指南：")
    print("1. [更新次数]: 太少(<2)说明太难或Alpha太小；太多(>15)说明Alpha太大(起点太低)。")
    print("2. [提升幅度]: 越高越好。代表 DRL+SA 真正挖掘出的额外价值。")
    print("3. [耐力]: 最后一次最优解更新出现的位置。")
    print("   - < 50%: 早泄型 (Early Converge)，建议调小 Patience 节省时间。")
    print("   - > 90%: 坚韧型 (Late Bloomer)，说明时间可能给短了，或者潜力极大。")
    print("   - (L:xx%): 括号里的数字代表有多少比例的 Episode 在进度条 80% 之后还有更新。")

if __name__ == "__main__":
    analyze_instance_health(LOG_FILE_PATH)