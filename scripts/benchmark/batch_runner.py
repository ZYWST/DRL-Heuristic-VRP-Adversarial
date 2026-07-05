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
import torch

# ================= 配置区域 =================
# 1. 核心脚本与模型路径
AGENT_SCRIPT = "run_with_agent.py"
# 请确保这里指向你最好的模型 (同级目录下要有 _vecnormalize.pkl)
MODEL_PATH = "logs/best_model.zip" 

# 2. 算例文件夹 (会自动扫描该目录下所有 .dat 文件)
PROBLEM_DIR = "problem_instances"

# 3. 实验参数配置
# SEEDS = [192020, 192021,192022,192023,192024,192025,192026,192027,192028,192029]  # 你想测试的随机种子列表
# SEEDS = [192020,192021,192022,192023,192024,192025]  # 你想测试的随机种子列表
SEEDS = [292024]  # 你想测试的随机种子列表
GENS = 400                 # 每个任务跑多少代
WORKERS = 28                # 并行核数
TIME_LIMIT = 700          # 时间限制 (秒)，None 表示不限制
STRICT_MODE = True         # 是否开启严格中断模式

# 4. 日志保存目录
LOG_DIR = "batch_logs"
# ============================================

def run_command_and_log(cmd, log_file_path):
    """
    执行命令，并同时将输出打印到终端和写入日志文件 (类似 Linux 的 tee 命令)
    """
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    with open(log_file_path, "w", encoding="utf-8") as f_log:
        # 使用 subprocess.Popen 实时捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # 将 stderr 合并到 stdout
            shell=False,
            text=True,     # 以文本形式读取
            bufsize=1      # 行缓冲
        )

        # 实时读取输出
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # 1. 打印到终端 (去掉结尾换行符避免双重换行)
                print(line.rstrip())
                # 2. 写入文件
                f_log.write(line)
                f_log.flush() # 确保实时写入磁盘

    return process.returncode

def main():
    # torch.cuda.empty_cache()
    # 1. 扫描算例
    if not os.path.exists(PROBLEM_DIR):
        print(f"❌ 错误: 算例目录不存在 -> {PROBLEM_DIR}")
        return

    # 获取所有 .dat 文件
    problem_files = sorted(glob.glob(os.path.join(PROBLEM_DIR, "*.dat")))
    
    if not problem_files:
        print(f"❌ 错误: 在 {PROBLEM_DIR} 下未找到 .dat 文件")
        return

    # 创建本次批量运行的专属文件夹 (按时间戳)
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_log_dir = os.path.join(LOG_DIR, f"run_{batch_id}")
    os.makedirs(current_log_dir, exist_ok=True)

    total_tasks = len(problem_files) * len(SEEDS)
    current_task = 0

    print(f"🚀 开始批量任务 | 总计: {len(problem_files)} 算例 x {len(SEEDS)} 种子 = {total_tasks} 次运行")
    print(f"📂 日志将保存在: {current_log_dir}\n")

    start_time_all = time.time()

    # 2. 双重循环遍历
    for problem_path in problem_files:
        problem_name = os.path.splitext(os.path.basename(problem_path))[0]
        
        for seed in SEEDS:
            current_task += 1
            print("=" * 60)
            print(f"🔄 进度 [{current_task}/{total_tasks}] | 算例: {problem_name} | 种子: {seed}")
            print("=" * 60)

            # 构造日志文件名
            log_filename = f"{problem_name}_seed{seed}.log"
            log_path = os.path.join(current_log_dir, log_filename)

            # 构造命令
            # 对应 run_with_agent.py 的参数
            cmd = [
                sys.executable,  # 使用当前的 python解释器
                AGENT_SCRIPT,
                problem_path,
                "--model", MODEL_PATH,
                "--gens", str(GENS),
                "--seed", str(seed)
            ]

            if WORKERS:
                cmd.extend(["--workers", str(WORKERS)])
            
            if TIME_LIMIT:
                cmd.extend(["--time_limit", str(TIME_LIMIT)])
            
            if STRICT_MODE:
                cmd.append("--strict")

            # 执行并记录
            try:
                ret_code = run_command_and_log(cmd, log_path)
                
                if ret_code == 0:
                    print(f"✅ 任务完成. 日志已保存: {log_path}")
                else:
                    print(f"⚠️ 任务异常退出 (Code {ret_code}). 查看日志: {log_path}")
            
            except KeyboardInterrupt:
                print("\n🛑 用户强制停止批量脚本。")
                return
            except Exception as e:
                print(f"❌ 发生未知错误: {e}")

            print("\n") # 任务间空行

    total_time = time.time() - start_time_all
    print(f"🎉 所有批量任务已完成！总耗时: {total_time:.2f} 秒")
    print(f"📂 完整日志目录: {current_log_dir}")

if __name__ == "__main__":
    main()