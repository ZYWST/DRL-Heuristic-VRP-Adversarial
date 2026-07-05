import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import argparse
import sys
import os
import random
import numpy as np

# --- 1. 导入配置与底层核心 ---
from src.env.hyper_config import HyperConfig, GraspConfig
from src.algorithms import solver_coreDRL_train
from src.algorithms.solver_coreDRL_train import ProblemData

# --- 2. [核心] 直接复用你自己的 run_with_agent.py ---
# 从你的 run_with_agent.py 中导入 IntelligentOptimizer, AgentController 和 logger
try:
    from run_with_agent import IntelligentOptimizer, AgentController, setup_logger
except ImportError:
    print("❌ 无法导入 run_with_agent.py，请确保该文件在同一目录下且没有语法错误。")
    sys.exit(1)

# =====================================================================
# 【Monkey Patch】：拦截并修改子进程的 ProblemData
# =====================================================================
original_init_worker = solver_coreDRL_train.init_worker

def patched_init_worker(worker_config):
    # 1. 调用原始初始化，让子进程正常从硬盘读取 .dat
    original_init_worker(worker_config)
    
    # 2. 获取子进程刚刚在内存中建好的 ProblemData 对象
    pd = solver_coreDRL_train.WORKER_PD
    
    # 3. 从 config 中读取主进程传过来的缩放因子
    scale_u = getattr(worker_config, 'SCALE_U', 1.0)
    scale_c = getattr(worker_config, 'SCALE_C', 1.0)
    scale_h = getattr(worker_config, 'SCALE_H', 1.0)

    # 4. 执行深度同步缩放
    if abs(scale_u - 1.0) > 1e-5:
        pd.U = max(0, int(round(pd.U * scale_u)))
        
    if abs(scale_h - 1.0) > 1e-5:
        pd.H *= scale_h
        
    if abs(scale_c - 1.0) > 1e-5:
        # A. 缩放车辆容量
        for v_id, caps in pd.capacities.items():
            for g_id in caps:
                caps[g_id] *= scale_c
        # B. 缩放需求与供应
        # for n_id, dems in pd.demands.items():
        #     for g_id in dems:
        #         dems[g_id] *= scale_c

# 替换掉原始的 worker 初始化函数
solver_coreDRL_train.init_worker = patched_init_worker
# =====================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--model", type=str, default="./logs/best_model.zip")
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--strict", action="store_true")

    # 问题参数敏感性缩放因子
    parser.add_argument("--scale_u", type=float, default=1.0)
    parser.add_argument("--scale_c", type=float, default=1.0)
    parser.add_argument("--scale_h", type=float, default=1.0)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 初始化配置
    hyper_config = HyperConfig()
    grasp_config = GraspConfig(PROBLEM_DATA_PATH=args.problem, NUM_WORKERS=args.workers)
    grasp_config.USE_KSP_CACHE = True
    if args.strict:
        grasp_config.STRICT_DISRUPTION = True

    # 将缩放因子挂载到 grasp_config 上，传递给子进程
    grasp_config.SCALE_U = args.scale_u
    grasp_config.SCALE_C = args.scale_c
    grasp_config.SCALE_H = args.scale_h

    # 复用 run_with_agent.py 的 logger
    logger = setup_logger(grasp_config)

    # --- 主进程数据缩放 ---
    pd_main = ProblemData(args.problem, '', matrix_filepath="")
    
    if abs(args.scale_u - 1.0) > 1e-5:
        pd_main.U = max(0, int(round(pd_main.U * args.scale_u)))
        
    if abs(args.scale_h - 1.0) > 1e-5:
        pd_main.H *= args.scale_h
        
    if abs(args.scale_c - 1.0) > 1e-5:
        for v_id, caps in pd_main.capacities.items():
            for g_id in caps:
                caps[g_id] *= args.scale_c
        # for n_id, dems in pd_main.demands.items():
        #     for g_id in dems:
        #         dems[g_id] *= args.scale_c

    # =========================================================
    # [打印核查日志]
    # =========================================================
    logger.info("*" * 60)
    logger.info("🎯 [数据注入核实] 传入求解器前的最终问题参数:")
    logger.info(f"   ► 中断边数量 (U): {pd_main.U} 条")
    logger.info(f"   ► 任务总时间 (H): {pd_main.H:.2f}")
    logger.info(f"   ► 供需容量缩放 (C): {args.scale_c * 100:.0f}% ")
    logger.info("*" * 60)

    # 初始化真正的 AgentController 和 IntelligentOptimizer
    controller = AgentController(args.model, hyper_config, pd_main)
    optimizer = IntelligentOptimizer(grasp_config, logger)
    
    try:
        # [已修复截图中的 Bug]：使用 max_generations 作为参数名，或者直接位置传参
        optimizer.solve_with_agent(
            controller, 
            max_generations=args.gens, 
            time_limit=args.timeout,
            seed=args.seed
        )
    except KeyboardInterrupt:
        logger.warning("用户强制中断，正在保存当前状态...")
    finally:
        optimizer.shutdown()
        logger.info("优化器已安全关闭。")

if __name__ == '__main__':
    main()