import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import glob
import numpy as np
import logging
import copy
from src.env.hyper_config import GraspConfig, HyperConfig
from src.algorithms.solver_coreDRL_train import ParallelGraspOptimizer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AutoTune")

def run_benchmark_v2(sample_count=15, gens_per_sample=2):
    # 1. 读取基础配置
    base_cfg = HyperConfig()
    grasp_cfg = base_cfg.GRASP_CONFIG
    
    # 获取算例
    all_files = glob.glob(os.path.join(grasp_cfg.PROBLEM_INSTANCES_DIR, "*.dat"))
    if not all_files:
        logger.error("❌ 未找到算例文件")
        return

    selected_files = np.random.choice(all_files, size=min(len(all_files), sample_count), replace=False)
    
    # 检查当前配置中的 Scaling Factor
    current_scaling = base_cfg.REWARD_SCALING_FACTOR
    
    logger.info("="*60)
    logger.info(f"🚀 开始基准测试 (Auto-Tuning V2)")
    logger.info(f"📂 样本数: {len(selected_files)} | 每样本代数: {gens_per_sample}")
    if current_scaling != 1.0:
        logger.warning(f"⚠️  警告: 检测到 REWARD_SCALING_FACTOR = {current_scaling}")
        logger.warning(f"    既然你使用了 SB3 VecNormalize，建议将此参数改为 1.0，")
        logger.warning(f"    否则会导致输入值过小，影响归一化精度。")
    logger.info("="*60)

    stats_fitness = []
    stats_time = []

    for f_path in selected_files:
        logger.info(f"Running -> {os.path.basename(f_path)} ...")
        
        # 2. 构造测试配置 (确保与训练环境一致)
        test_cfg = copy.deepcopy(grasp_cfg)
        test_cfg.PROBLEM_DATA_PATH = f_path
        test_cfg.USE_KSP_CACHE = True
        
        # [关键] 保持与训练一致的中断逻辑
        test_cfg.STRICT_DISRUPTION = getattr(grasp_cfg, 'STRICT_DISRUPTION', False) 
        
        # 使用中性参数
        test_cfg.ALPHA = 0.25 
        test_cfg.POPULATION_SIZE = 24 
        test_cfg.SA_METROPOLIS_LEN = 300 

        try:
            # 静默运行
            optimizer = ParallelGraspOptimizer(test_cfg, logging.getLogger("Silent"))
            for g in range(gens_per_sample):
                res = optimizer.run_one_generation(g+1, total_generations=gens_per_sample)
                fit = res['new_overall_best_fitness']
                time_cost = res['gen_time_seconds']
                
                if fit > -99999: # 过滤无效解
                    stats_fitness.append(fit)
                    stats_time.append(time_cost)
            optimizer.shutdown()
        except Exception as e:
            logger.error(f"Error: {e}")

    if not stats_fitness:
        logger.error("❌ 无有效数据")
        return

    # 3. 计算建议值
    avg_fit = np.mean(stats_fitness)
    avg_time = np.mean(stats_time)
    
    # 目标：1% 的适应度提升 = 1.0 倍的时间成本 (Time_Factor=1.0)
    # (Avg_Fit * 0.01) / Rate = Avg_Time * 1.0
    # Rate = (Avg_Fit * 0.01) / Avg_Time
    
    recommended_rate = (avg_fit * 0.01) / avg_time
    
    logger.info("\n" + "="*60)
    logger.info("📊 统计结果 & 推荐配置")
    logger.info("-" * 30)
    logger.info(f"平均 Fitness: {avg_fit:,.0f}")
    logger.info(f"平均 Time:    {avg_time:.2f}s")
    logger.info("-" * 30)
    logger.info(f"💡 推荐参数修改 (针对 hyper_config.py):")
    logger.info(f"1. FITNESS_TO_TIME_EXCHANGE_RATE = {recommended_rate:.2f}")
    logger.info(f"2. REWARD_SCALING_FACTOR         = 1.0  (配合 VecNormalize)")
    logger.info("="*60)
    
    return recommended_rate

if __name__ == "__main__":
    run_benchmark_v2()