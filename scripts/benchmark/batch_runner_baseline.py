import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import sys
import glob
import subprocess
import time
from datetime import datetime

# ================= 配置区域 =================
# 1. 目标脚本
SOLVER_SCRIPT = "test_baseline_parallel.py"

# 2. 算例文件夹
PROBLEM_DIR = "problem_instances"

# 3. 实验参数配置
# SEEDS = [101, 102, 103]             # 想要测试的种子列表
# SEEDS = [2020,2021,2022,2023,2024,2025,2026,2027,2028,2029]                  # 示例种子
SEEDS = [2025,2026,2027,2028,2029]                  # 示例种子
MAX_TIME_PER_RUN = 300               # 单个任务的时间限制 (秒)

# 4. 日志保存目录
LOG_DIR = "batch_logs_baseline"
# ============================================

def run_command_and_log(cmd, log_file_path):
    """
    执行命令，并同时将输出打印到终端和写入日志文件
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, "w", encoding="utf-8") as f_log:
        # 使用 subprocess.Popen 实时捕获输出
        # bufsize=0 配合 python -u 可以最大程度保证实时性
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            shell=False,
            text=True,
            bufsize=1      # 行缓冲
        )

        while True:
            # 实时读取一行
            line = process.stdout.readline()
            
            # 如果没有读到内容且进程已结束，跳出循环
            if not line and process.poll() is not None:
                break
                
            if line:
                # 1. 打印到终端 (去掉结尾换行符避免双重换行)
                print(line.rstrip())
                # 2. 写入文件并立即刷新缓冲区
                f_log.write(line)
                f_log.flush() 

    return process.returncode

def main():
    if not os.path.exists(SOLVER_SCRIPT):
        print(f"❌ 错误: 找不到目标脚本 {SOLVER_SCRIPT}")
        return

    # 扫描算例逻辑
    if not os.path.exists(PROBLEM_DIR):
        if os.path.exists("CHINA_Case9.dat"):
            print(f"⚠️ 警告: 目录 {PROBLEM_DIR} 不存在，回退到测试单个文件: CHINA_Case9.dat")
            problem_files = ["CHINA_Case9.dat"]
        else:
            print(f"❌ 错误: 算例目录不存在 -> {PROBLEM_DIR}")
            return
    else:
        # 排序以保证执行顺序一致
        problem_files = sorted(glob.glob(os.path.join(PROBLEM_DIR, "*.dat")))
    
    if not problem_files:
        print(f"❌ 错误: 未找到 .dat 文件")
        return

    # 创建本次运行的文件夹
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_dir = os.path.join(LOG_DIR, f"run_{batch_id}")
    os.makedirs(current_log_dir, exist_ok=True)

    total_tasks = len(problem_files) * len(SEEDS)
    current_task = 0

    print(f"🚀 开始 Baseline 批量任务 (优先遍历算例) | 总计: {len(SEEDS)} 种子 x {len(problem_files)} 算例")
    print(f"📂 日志目录: {current_log_dir}\n")

    start_time_all = time.time()

    # --- [修改] 外层循环种子，内层循环算例 ---
    for seed in SEEDS:
        print(f"\n>>> 正在执行种子: {seed} (本轮共 {len(problem_files)} 个算例) <<<")
        
        for problem_path in problem_files:
            problem_name = os.path.splitext(os.path.basename(problem_path))[0]
            
            current_task += 1
            print("=" * 60)
            # 这里的打印顺序也调整了一下，方便查看
            print(f"🔄 进度 [{current_task}/{total_tasks}] | 种子: {seed} | 算例: {problem_name}")
            print("=" * 60)

            log_filename = f"{problem_name}_S{seed}.log"
            log_path = os.path.join(current_log_dir, log_filename)

            # [关键] 添加 "-u" 参数强制禁用 Python 输出缓冲
            cmd = [
                sys.executable, 
                "-u",              
                SOLVER_SCRIPT,
                "--problem", problem_path,
                "--seed", str(seed),
                "--timeout", str(MAX_TIME_PER_RUN)
            ]

            try:
                run_command_and_log(cmd, log_path)
            except KeyboardInterrupt:
                print("\n🛑 用户强制停止批量脚本。")
                sys.exit(0)
            except Exception as e:
                print(f"❌ 发生未知错误: {e}")

            print("\n") 

    total_time = time.time() - start_time_all
    print(f"🎉 所有任务完成！总耗时: {total_time:.2f} 秒")

if __name__ == "__main__":
    main()