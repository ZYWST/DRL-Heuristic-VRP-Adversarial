import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import sys
import subprocess
import time

# ================= 配置区域 =================
TARGET_SCRIPT = "run_with_agent_problem_sens.py"
# 请修改为你要测试的那唯一一个算例路径
TARGET_PROBLEM = "./problem_instances/CHINA_75demands_modified_max_util_weight0223.dat"  
LOG_ROOT_DIR = "./problem_sensitivity_logs"

SEEDS = [2020, 2021,2027,2028,2029]
# SEEDS = [2022,2023,2024,2025,2026]
PARAMS = ['U', 'C', 'H']
# PARAMS = ['U']
SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] 
# SCALES = [1.0]

GENS = 300
TIMEOUT = 200
# ===========================================

def run_experiment():
    if not os.path.exists(TARGET_PROBLEM):
        print(f"❌ 找不到算例文件: {TARGET_PROBLEM}")
        return

    total_tasks = len(SEEDS) * len(PARAMS) * len(SCALES)
    current_task = 0

    print(f"🚀 问题参数敏感性分析开始 (多进程同步注入版)")
    print(f"   - 目标算例: {os.path.basename(TARGET_PROBLEM)}")
    print(f"   - 缩放因子: {SCALES}")
    print("="*60)

    for seed in SEEDS:
        for param in PARAMS:
            for scale in SCALES:
                current_task += 1
                
                # 【修复 ±19% 的 Bug】使用 round() 解决浮点精度问题
                change_pct = int(round((scale - 1.0) * 100))
                change_str = f"{change_pct:+d}%"
                
                log_dir = os.path.join(LOG_ROOT_DIR, f"Seed_{seed}", f"Param_{param}", f"Scale_{scale:.1f}_{change_str}")
                os.makedirs(log_dir, exist_ok=True)
                
                log_file = os.path.join(log_dir, "run.log")

                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        if "Optimization Finished" in f.read() or "安全关闭" in f.read():
                            print(f"   🔹 [{current_task}/{total_tasks}] 跳过已完成: Seed {seed} | {param} {change_str}")
                            continue

                cmd = [
                    sys.executable, "-u", TARGET_SCRIPT,
                    "--problem", TARGET_PROBLEM,
                    "--seed", str(seed),
                    "--gens", str(GENS),
                    "--timeout", str(TIMEOUT)
                ]
                
                if param == 'U': cmd.extend(["--scale_u", str(scale)])
                elif param == 'C': cmd.extend(["--scale_c", str(scale)])
                elif param == 'H': cmd.extend(["--scale_h", str(scale)])

                print(f"   ▶️  [{current_task}/{total_tasks}] 运行 Seed {seed} | {param} {change_str} ... ", end="", flush=True)
                
                start_t = time.time()
                try:
                    with open(log_file, "w", encoding="utf-8") as f_log:
                        subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT)
                    elapsed = time.time() - start_t
                    print(f"✅ ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"❌ Error: {e}")

    print("\n✅ 所有实验运行完毕！")

if __name__ == "__main__":
    run_experiment()