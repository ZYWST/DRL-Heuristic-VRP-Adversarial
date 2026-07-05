import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import sys
import glob
import subprocess
import numpy as np
import time

# ================= 配置区域 =================

TARGET_SCRIPT = "run_with_agent_ablation.py"
PROBLEM_DIR = "./problem_instances"
LOG_ROOT_DIR = "./ablation_logs"

# 1. 种子层
SEEDS = [2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]

# 2. 算例层 (自动读取)
DAT_FILES = sorted(glob.glob(os.path.join(PROBLEM_DIR, "*.dat")))

# 3. 参数层 (DRL 动作参数及其边界)
# 格式: '参数键名': (最小值, 最大值, 是否为整数)
# 这里的键名必须与 decide_parameters 返回的字典键名一致
PARAM_CONFIGS = {
    'POPULATION_SIZE':   (8, 40, True),    # [8, 40] 整数
    'ALPHA':             (0.2, 1.0, False),# [0.2, 1.0] 浮点
    'SA_METROPOLIS_LEN': (15, 350, True)   # [15, 350] 整数
}

NUM_STEPS = 10  # 等距取10个点
GENS = 900
TIMEOUT = 300

# ===========================================

def run_ablation():
    if not DAT_FILES:
        print(f"❌ 在 {PROBLEM_DIR} 未找到 .dat 文件")
        return

    # 计算总任务数用于进度条
    total_tasks = len(SEEDS) * len(PARAM_CONFIGS) * NUM_STEPS * len(DAT_FILES)
    current_task = 0

    print(f"🚀 DRL 动作参数消融实验开始")
    print(f"   - 目标参数: {list(PARAM_CONFIGS.keys())}")
    print(f"   - 采样点数: {NUM_STEPS}")
    print(f"   - 种子列表: {SEEDS}")
    print("="*60)

    # 第一层：种子
    for seed in SEEDS:
        
        # 第二层：参数种类
        for param_name, (min_val, max_val, is_int) in PARAM_CONFIGS.items():
            
            # 生成 10 个等距值
            values = np.linspace(min_val, max_val, NUM_STEPS)
            if is_int:
                # 如果是整数参数，取整并去重 (防止区间太小导致重复，但在您的区间下一般不会)
                values = np.unique(np.round(values).astype(int))
                # 确保还是 10 个点（如果区间极小，这里可能会变少，但对于 [8,40] 没问题）
                # 为了严格保证 10 次实验，即使数值一样我们也跑（保持控制变量）
                values = np.round(np.linspace(min_val, max_val, NUM_STEPS)).astype(int)

            # 第三层：参数的值
            for val_idx, val in enumerate(values):
                val_num = val # 保留数值类型用于计算
                
                # 格式化文件夹名
                if is_int:
                    val_str = f"{int(val)}"
                    log_val_folder = f"Val_{val_idx}_{int(val)}"
                else:
                    val_str = f"{val:.4f}"
                    log_val_folder = f"Val_{val_idx}_{val:.4f}"

                # 创建日志目录: ablation_logs/Seed_2024/ALPHA/Val_0_0.2000/
                log_dir = os.path.join(LOG_ROOT_DIR, f"Seed_{seed}", param_name, log_val_folder)
                os.makedirs(log_dir, exist_ok=True)
                
                print(f"\n🌱 [Seed {seed}] 参数 {param_name} = {val_str} ({val_idx+1}/{NUM_STEPS})")

                # 第四层：算例
                for problem_path in DAT_FILES:
                    current_task += 1
                    case_name = os.path.basename(problem_path)
                    log_file = os.path.join(log_dir, f"{case_name}.log")
                    
                    # 简单断点续传
                    if os.path.exists(log_file):
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            if "Optimization Finished" in f.read():
                                print(f"   🔹 跳过: {case_name}")
                                continue

                    # 构造运行命令
                    cmd = [
                        sys.executable, "-u", TARGET_SCRIPT,
                        "--problem", problem_path,
                        "--seed", str(seed),
                        "--gens", str(GENS),
                        "--timeout", str(TIMEOUT),
                        # 传入消融参数
                        "--ablation_param", param_name,
                        "--ablation_value", val_str
                    ]

                    start_t = time.time()
                    try:
                        with open(log_file, "w", encoding="utf-8") as f_log:
                            # 运行子进程
                            subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT)
                        
                        elapsed = time.time() - start_t
                        print(f"   ✅ [{current_task}/{total_tasks}] {case_name} ({elapsed:.1f}s)")
                        
                    except Exception as e:
                        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    run_ablation()