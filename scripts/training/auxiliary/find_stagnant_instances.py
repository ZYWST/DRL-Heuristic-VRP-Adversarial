import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import os
from collections import defaultdict

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260116_210603.log"   # 请确保这是你的日志文件名
EARLY_STOP_THRESHOLD = 3         # "只有前N代更新" (用户定义为1或2，这里设3比较保险)
MIN_EPISODE_LENGTH = 15          # "总代数至少要跑多少" (防止刚跑2步就崩溃的算例被误判)
# ===========================================

def analyze_stagnation(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        return

    print(f"🔍 正在分析日志，寻找“开局即巅峰”的停滞算例...")
    print(f"   - 判定标准: 最后一次更新发生在第 {EARLY_STOP_THRESHOLD} 步(或更早)，且 Episode 总长 > {MIN_EPISODE_LENGTH} 步")
    print("-" * 60)

    # 正则表达式
    # 匹配: New Episode: 切换算例 -> problem_data_90dotV2.dat
    episode_pattern = re.compile(r"New Episode:\s*切换算例\s*->\s*(.*\.dat)")
    
    # 匹配: Step [  1/1000]
    step_pattern = re.compile(r"Step\s*\[\s*(\d+)/")
    
    # 匹配: -> 🎯NEW BEST!
    new_best_pattern = re.compile(r"->\s*🎯NEW BEST")

    # 数据存储
    # { "problem_name": [ {"total_steps": 100, "last_update": 2}, ... ] }
    problem_stats = defaultdict(list)
    
    current_problem = None
    current_episode_info = {
        "total_steps": 0,
        "last_update_step": 0
    }

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. 检查是否切换算例 (新 Episode 开始)
            ep_match = episode_pattern.search(line)
            if ep_match:
                # 结算上一个 Episode
                if current_problem and current_episode_info["total_steps"] > 0:
                    problem_stats[current_problem].append(current_episode_info)
                
                # 初始化新 Episode
                current_problem = ep_match.group(1).strip()
                current_episode_info = {
                    "total_steps": 0,
                    "last_update_step": 0 # 默认为0，表示没更新过
                }
                continue

            # 2. 检查 Step 更新
            step_match = step_pattern.search(line)
            if step_match:
                step_num = int(step_match.group(1))
                current_episode_info["total_steps"] = step_num
                
                # 3. 检查是否有 NEW BEST
                # 注意：Step 1 往往会有 NEW BEST (初始化)，这是正常的
                if new_best_pattern.search(line):
                    current_episode_info["last_update_step"] = step_num

    # 结算文件末尾的最后一个 Episode
    if current_problem and current_episode_info["total_steps"] > 0:
        problem_stats[current_problem].append(current_episode_info)

    # ================= 分析结果 =================
    stagnant_problems = []
    normal_problems = []

    print(f"{'算例名称':<40} | {'状态':<10} | {'详情 (最后更新/总步数)'}")
    print("-" * 80)

    for p_name, episodes in problem_stats.items():
        is_stagnant = False
        details = []
        
        stagnant_count = 0
        
        for ep in episodes:
            last_up = ep["last_update_step"]
            total = ep["total_steps"]
            
            # 判定逻辑
            # 如果跑得够长 (total > MIN)，但最后一次更新很早 (last <= THRESHOLD)
            if total >= MIN_EPISODE_LENGTH and last_up <= EARLY_STOP_THRESHOLD:
                stagnant_count += 1
                details.append(f"⚠️ {last_up}/{total}")
            else:
                details.append(f"✅ {last_up}/{total}")
        
        # 如果该算例的大部分 Episode 都是停滞的 (例如 > 50%)，则标记为停滞
        # 或者只要出现过一次严重停滞就标记 (根据你的剔除严格度，这里选严格模式：只要有一次表现极差就列出来)
        if stagnant_count > 0:
            stagnant_problems.append(p_name)
            status = "❌ 停滞"
        else:
            normal_problems.append(p_name)
            status = "✅ 正常"

        print(f"{p_name:<40} | {status:<10} | {', '.join(details)}")

    print("-" * 80)
    print(f"📊 统计: 总算例 {len(problem_stats)} 个，发现疑似停滞算例 {len(stagnant_problems)} 个。")
    
    if stagnant_problems:
        print("\n🗑️  建议剔除列表 (可直接复制):")
        print("-" * 20)
        for p in stagnant_problems:
            print(p)
        print("-" * 20)
        print("建议：将这些算例移动到 'backup' 文件夹，然后重新开始训练。")
    else:
        print("\n🎉 恭喜！未发现明显的早期停滞算例。")

if __name__ == "__main__":
    analyze_stagnation(LOG_FILE_PATH)