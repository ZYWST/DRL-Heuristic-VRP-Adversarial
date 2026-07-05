import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
# calibrate_reward_v2.py (模拟真实DRL环境版)
import os
import glob
import time
import numpy as np
import logging
import sys
from collections import deque

# 导入你的核心模块
from src.env.hyper_config import HyperConfig
from src.algorithms.solver_coreDRL import ParallelGraspOptimizer, GraspConfig, ProblemData

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("CalibrateV2")

def calibrate_realistic():
    # 1. 加载配置
    config = HyperConfig()
    instances_dir = config.GRASP_CONFIG.PROBLEM_INSTANCES_DIR
    problem_files = sorted(glob.glob(os.path.join(instances_dir, '*.dat')))
    
    if not problem_files:
        print("❌ 未找到算例文件！")
        return

    print(f"🔍 扫描到 {len(problem_files)} 个算例，开始【真实环境模拟】校准...")
    print(f"   - 时间限制: {config.MAX_SECONDS_PER_EPISODE} 秒")
    print(f"   - 早停阈值: {config.EARLY_STOPPING_STABLE_STEPS} 步无提升")
    print("-" * 60)

    all_valid_ratios = []       # 所有有效提升的性价比
    late_stage_ratios = []      # 后半程（更具参考价值）的性价比
    
    # 模拟参数：给一个比较“中庸”的参数，模拟Agent的平均水平
    # 不要给太强（太快收敛）也不要给太弱（搜不到解）
    SIM_PARAMS = {
        "POPULATION_SIZE": 24,
        "ALPHA": 0.25,
        "SA_METROPOLIS_LEN": 190 
    }

    for p_path in problem_files:
        p_name = os.path.basename(p_path)
        print(f"🏃 正在模拟: {p_name} ... ", end="", flush=True)
        
        # --- 初始化环境 ---
        grasp_conf = GraspConfig(PROBLEM_DATA_PATH=p_path, NUM_WORKERS=None)
        # 注入模拟参数
        grasp_conf.POPULATION_SIZE = SIM_PARAMS["POPULATION_SIZE"]
        grasp_conf.ALPHA = SIM_PARAMS["ALPHA"]
        grasp_conf.SA_METROPOLIS_LEN = SIM_PARAMS["SA_METROPOLIS_LEN"]
        
        optimizer = ParallelGraspOptimizer(grasp_conf, logger)
        
        # --- 状态变量 (完全复刻 hyper_env.py) ---
        current_step = 0
        total_elapsed_time = 0.0
        best_overall_fitness = -1e9
        steps_at_optimum = 0
        last_step_was_feasible = False
        
        episode_ratios = []
        
        # 先运行第 0 代 (初始化)
        t0 = time.time()
        stats = optimizer.run_one_generation(0)
        t_init = time.time() - t0
        best_overall_fitness = stats['new_overall_best_fitness']
        total_elapsed_time += t_init
        
        # --- 开始 Episode 循环 ---
        while True:
            current_step += 1
            
            # 1. 运行一代
            # 注意：实际耗时可能比 solver 返回的 internal time 长，我们用 wall clock
            t_start = time.time()
            stats = optimizer.run_one_generation(current_step)
            step_duration = time.time() - t_start # 真实挂钟时间
            
            # 累加时间
            total_elapsed_time += step_duration
            
            # 获取适应度
            new_best = stats['new_overall_best_fitness']
            
            # 2. 计算提升量 (Improvement)
            improvement = new_best - best_overall_fitness
            
            # 3. 记录有效性价比 (Fitness per Second)
            if improvement > 1e-6 and step_duration > 0.001:
                ratio = improvement / step_duration
                episode_ratios.append(ratio)
                all_valid_ratios.append(ratio)
            
            # 4. 更新早停计数器 (复刻 Env 逻辑)
            if new_best > best_overall_fitness:
                best_overall_fitness = new_best
                steps_at_optimum = 0 # 重置
            else:
                # 假设只要不是 -1e5 就是 feasible
                is_feasible = new_best > -99999
                if is_feasible: 
                    steps_at_optimum += 1
            
            # 5. 检查终止条件
            # A. 早停
            if steps_at_optimum >= config.EARLY_STOPPING_STABLE_STEPS:
                print(f"[早停] {current_step}步 (无提升)", end="")
                break
                
            # B. 时间限制
            if total_elapsed_time >= config.MAX_SECONDS_PER_EPISODE:
                print(f"[超时] {current_step}步 ({total_elapsed_time:.1f}s)", end="")
                break
                
        # --- Episode 结束，收集后期数据 ---
        # 我们认为 Episode 的后 50% 是“艰难时刻”，这时的数据对设定汇率最有参考意义
        if episode_ratios:
            mid_point = len(episode_ratios) // 4
            # 如果列表很短，就全部算作 late stage
            start_idx = mid_point if mid_point > 0 else 0
            late_stage_ratios.extend(episode_ratios[start_idx:])
            
        print(f" -> 收集到 {len(episode_ratios)} 个有效提升点")
        
        optimizer.shutdown()

    print("-" * 60)
    print("📊 统计结果分析")
    
    if not all_valid_ratios:
        print("❌ 数据不足！模拟过程中几乎没有产生任何提升。请检查 Solver 是否正常工作。")
        return

    # 计算关键分位数
    # P50 (中位数): 代表大多数时候的效率
    # P20 (较低值): 代表进入困难阶段后的效率 <-- 这是我们设定汇率的锚点
    
    avg_ratio = np.mean(all_valid_ratios)
    p50_total = np.percentile(all_valid_ratios, 50)
    
    # 重点关注 Late Stage
    if late_stage_ratios:
        p20_late = np.percentile(late_stage_ratios, 20)
        p50_late = np.percentile(late_stage_ratios, 50)
    else:
        p20_late = np.percentile(all_valid_ratios, 20)
        p50_late = np.percentile(all_valid_ratios, 50)

    print(f"全阶段平均产出率: {avg_ratio:.2f} Fit/s")
    print(f"困难阶段中位产出: {p50_late:.2f} Fit/s")
    print(f"困难阶段 P20产出: {p20_late:.2f} Fit/s (保守底线)")
    
    print("-" * 60)
    print("💡 推荐配置")
    
    # 策略：
    # 汇率应该设定为“困难阶段的保守底线”。
    # 为什么？
    # 如果 Rate = P20_Late (例如 100)，意味着只要产出率高于 100 Fit/s，
    # 奖励公式 (Imp/Rate - Time) = (Imp/100 - Time) > 0 就大概率是正的。
    # 这能保证 Agent 在 80% 的困难优化时间内，只要有产出，就能获得正反馈，不会摆烂。
    
    recommended_rate = max(p20_late, 1.0) # 至少为 1
    
    print(f"✅ 建议 FITNESS_TO_TIME_EXCHANGE_RATE: {recommended_rate:.1f}")
    print(f"   (原配置 10500 意味着要求产出率 > 10500 Fit/s 才能回本，这在后期是不可能的)")
    
    print("\n[请修改 hyper_config.py]:")
    print(f"FITNESS_TO_TIME_EXCHANGE_RATE = {recommended_rate:.1f}")
    print(f"REWARD_SCALE_GLOBAL_BEST = 1.0 / FITNESS_TO_TIME_EXCHANGE_RATE")
    # 既然有了精准汇率，Progress 奖励可以和 Global 保持一致，或者稍微小一点
    print(f"REWARD_SCALE_PROGRESS = REWARD_SCALE_GLOBAL_BEST * 0.5") 

if __name__ == "__main__":
    calibrate_realistic()