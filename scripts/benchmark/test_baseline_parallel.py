import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import time
import sys
import argparse
import random
from collections import defaultdict

# 引入核心配置和数据结构
from src.env.hyper_config import GraspConfig
from src.algorithms.solver_coreDRL import ProblemData, PathLibrary
# 引入我们修改过的 solver
from src.algorithms.solver_baseline_advance_parallel import BaselineVNS
from src.utils.solution_auditor import SolutionAuditor

# ==========================================
# 报告生成函数 (保持不变)
# ==========================================
def generate_detailed_report(pd: ProblemData, pl: PathLibrary, sol_dict: dict, filename: str, H_limit: float):
    base_name = os.path.basename(filename).split('.')[0]
    report_path = f"report_{base_name}_{int(time.time())}.txt"
    
    def dual_print(text, file_handle):
        # print(text) # 静默模式
        file_handle.write(text + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        dual_print("\n" + "="*40, f)
        dual_print("=== 最优解详细报告 ===", f)
        dual_print(f"算例: {os.path.basename(filename)}", f)
        dual_print(f"Obj (Fitness): {sol_dict.get('fitness', 0):.2f}", f)
        dual_print("="*40, f)
        # ... (由于篇幅原因，这里省略了详细的路径重构代码，
        # 如果你需要完整的报告生成逻辑，请保留你原文件中的 generate_detailed_report 函数内容) ...

    print(f"✅ 报告已保存至: {report_path}")

# ==========================================
# 主测试逻辑
# ==========================================
def run_baseline_test():
    # 1. 参数解析
    parser = argparse.ArgumentParser(description="并行加速 Baseline VNS 测试")
    parser.add_argument('--problem', type=str, default="CHINA_Case9.dat", help='算例文件路径')
    parser.add_argument('--seed', type=int, default=1091238, help='随机种子')
    parser.add_argument('--timeout', type=float, default=1200, help='最大运行时间(秒)')
    
    # [关键修复] 添加 workers 参数定义
    # 默认使用 CPU 核心数 - 2，防止卡死系统
    default_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    parser.add_argument('--workers', type=int, default=default_workers, help='并行Worker数量')
    
    args = parser.parse_args()

    DATA_PATH = args.problem
    TEST_SEED = args.seed
    MAX_TIME = args.timeout
    NUM_WORKERS = args.workers

    # 路径检查
    if not os.path.exists(DATA_PATH):
        import glob
        dats = glob.glob("./problem_instances/*.dat")
        if dats: 
            print(f"⚠️ 指定文件不存在，自动使用: {dats[0]}")
            DATA_PATH = dats[0]
        else:
            print(f"❌ Error: Data file not found at {DATA_PATH}")
            return

    print("="*60)
    print("🚀 [Parallel Acceleration] Baseline VNS")
    print(f"   Problem: {os.path.basename(DATA_PATH)}")
    print(f"   Seed: {TEST_SEED}")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Timeout: {MAX_TIME}s")
    print("="*60)
    
    # 初始化配置
    cfg = GraspConfig()
    cfg.PROBLEM_DATA_PATH = DATA_PATH
    cfg.USE_KSP_CACHE = True
    cfg.SA_M_PATHS = 1  

    print("Loading ProblemData...")
    pd = ProblemData(DATA_PATH, '', matrix_filepath="")
    
    print("Loading PathLibrary...")
    pl = PathLibrary(pd, cfg)
    
    print("Initializing Solver...")
    solver = BaselineVNS(pd, cfg, pl)
    
    print(f"\n=== Starting Optimization ===")
    t_start = time.time()
    
    # [关键] 调用并行加速版 run 方法
    best_fit, best_sol = solver.run_parallel_acceleration(
        max_time=MAX_TIME, 
        seed=TEST_SEED,
        num_workers=NUM_WORKERS
    )
    
    t_end = time.time()
    
    print("\n" + "="*60)
    print(f"🏆 Optimization Finished")
    print(f"   Final Fitness: {best_fit:.4f}")
    print(f"   Time Elapsed: {t_end - t_start:.2f}s")
    print("="*60)

    # 简单报告
    generate_detailed_report(pd, pl, best_sol, DATA_PATH, pd.H)

if __name__ == "__main__":
    # Windows 下多进程必须保护入口
    run_baseline_test()