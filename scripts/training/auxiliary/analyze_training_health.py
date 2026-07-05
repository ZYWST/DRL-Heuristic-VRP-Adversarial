import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import re
import numpy as np
import os
import sys
from collections import defaultdict, deque

# ================= 配置区域 =================
LOG_FILE_PATH = "logs/train_hyper_20260112_194146.log"  # 你的日志文件
# 阈值设置 (医生诊断标准)
THRESHOLDS = {
    "critic_spike": 10.0,       # Critic Loss 突然暴涨的倍数
    "min_entropy": 0.005,       # 熵过低的警戒线 (探索停止)
    "low_acceptance": 0.05,     # SA 接受率过低 (陷入局部最优)
    "high_acceptance": 0.95,    # SA 接受率过高 (温度太高/步长太小)
    "stagnation_steps": 10000,  # 多少步没有新高算严重停滞
}
# ===========================================

def analyze_health(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 错误：未找到日志文件 '{log_file}'")
        return

    print(f"🩺 正在对训练日志进行全盘健康体检: {log_file} ...\n")

    # --- 数据容器 ---
    data = {
        "critic_loss": [],
        "actor_loss": [],
        "ent_coef": [],
        "rewards": [],
        "ep_lengths": [],
        "acceptance_rates": [], # 如果你有 log 这个
        "sa_gains": [],         # SA 提升幅度
        "new_bests": [],        # (step, value)
        "errors": [],
        "timeouts": []
    }
    
    current_step = 0
    last_best_step = 0
    
    # 正则表达式
    # 1. SB3 表格数据 (匹配 "| train/critic_loss | 1.234 |")
    sb3_metric_re = re.compile(r"\|\s*([\w/]+)\s*\|\s*([\-\d\.e]+)\s*\|")
    # 2. 自定义 Step 日志
    step_re = re.compile(r"Step\s*\[\s*(\d+)/")
    # 3. Solver 详细日志 (Fit: Avg=... / Max=...)
    fit_re = re.compile(r"Fit: Avg=([\-\d\.]+)\s*/\s*Max=([\-\d\.]+)")
    # 4. New Best
    new_best_re = re.compile(r"🎯NEW BEST")
    # 5. 错误/警告
    error_re = re.compile(r"(Error|Exception|Traceback|Failed|Timeout)", re.IGNORECASE)

    # --- 解析日志 ---
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # A. 扫描错误
            if error_re.search(line):
                # 排除一些良性的 Warning
                if "UserWarning" not in line:
                    data["errors"].append((line_num, line.strip()))

            # B. 解析 Step
            step_match = step_re.search(line)
            if step_match:
                current_step = int(step_match.group(1))
                
                # 检查是否 New Best
                if new_best_re.search(line):
                    # 尝试提取数值
                    val_match = fit_re.search(line)
                    if val_match:
                        val = float(val_match.group(2))
                        data["new_bests"].append((current_step, val))
                        last_best_step = current_step

            # C. 解析 SB3 训练指标
            # SB3 的日志通常是表格形式，我们需要提取 value
            if "|" in line:
                metric_match = sb3_metric_re.search(line)
                if metric_match:
                    key = metric_match.group(1).strip()
                    try:
                        val = float(metric_match.group(2))
                        if key == "train/critic_loss": data["critic_loss"].append(val)
                        elif key == "train/actor_loss": data["actor_loss"].append(val)
                        elif key == "train/ent_coef": data["ent_coef"].append(val)
                        elif key == "rollout/ep_rew_mean": data["rewards"].append(val)
                        elif key == "rollout/ep_len_mean": data["ep_lengths"].append(val)
                    except ValueError: pass

    # ================= 诊断报告 =================
    
    print("=" * 60)
    print("📋 诊断报告")
    print("=" * 60)

    # 1. 错误扫描
    if data["errors"]:
        print(f"❌ [风险] 发现 {len(data['errors'])} 个错误/超时信号！")
        for i, (ln, msg) in enumerate(data["errors"][:5]):
            print(f"   - Line {ln}: {msg[:80]}...")
        if len(data["errors"]) > 5: print("   ... (更多错误已折叠)")
    else:
        print("✅ [通过] 日志清洁，未发现底层报错或超时。")

    print("-" * 30)

    # 2. 熵 (Entropy) 健康度
    if data["ent_coef"]:
        curr_ent = data["ent_coef"][-1]
        start_ent = data["ent_coef"][0]
        drop_rate = (start_ent - curr_ent) / len(data["ent_coef"])
        
        print(f"🔍 [探索] 当前熵系数: {curr_ent:.4f} (初始: {start_ent:.2f})")
        
        if curr_ent < THRESHOLDS["min_entropy"]:
            print(f"⚠️ [警告] 熵系数过低 (<{THRESHOLDS['min_entropy']})！")
            print("   -> 风险：模型已停止探索，可能过早收敛到局部最优。")
            print("   -> 建议：调大 target_entropy 或 增加 init_ent_coef。")
        elif curr_ent > 0.5 and current_step > 50000:
            print(f"⚠️ [警告] 熵系数下降太慢！")
            print("   -> 风险：模型还在疯狂随机试探，收敛困难。")
        else:
            print("✅ [通过] 探索-利用平衡良好。")
    else:
        print("⚪ [未知] 未找到 'train/ent_coef' 数据。")

    print("-" * 30)

    # 3. Critic (价值观) 稳定性
    if len(data["critic_loss"]) > 10:
        recent_loss = data["critic_loss"][-10:]
        avg_loss = np.mean(recent_loss)
        max_loss = np.max(recent_loss)
        
        # 简单尖峰检测
        spike_detected = False
        if len(data["critic_loss"]) > 20:
            prev_avg = np.mean(data["critic_loss"][-20:-10])
            if avg_loss > prev_avg * THRESHOLDS["critic_spike"] and avg_loss > 0.1:
                spike_detected = True
        
        print(f"🔍 [价值] 近期 Critic Loss 均值: {avg_loss:.4f}")
        
        if spike_detected:
            print(f"❌ [危险] 检测到 Loss 剧烈飙升 (> {THRESHOLDS['critic_spike']}倍)！")
            print("   -> 风险：梯度爆炸，Critic 网络崩溃。")
            print("   -> 建议：检查 clip_reward 是否生效，或降低学习率。")
        elif np.isnan(avg_loss) or np.isinf(avg_loss):
            print(f"❌ [致命] Loss 出现 NaN/Inf！训练已崩溃。")
        else:
            print("✅ [通过] 价值网络训练稳定。")
    else:
        print("⚪ [未知] 数据不足。")

    print("-" * 30)

    # 4. 优化活力 (Stagnation)
    steps_since_last_best = current_step - last_best_step
    print(f"🔍 [活力] 上次 New Best 在 Step {last_best_step} (距今 {steps_since_last_best} 步)")
    
    if steps_since_last_best > THRESHOLDS["stagnation_steps"]:
        print(f"⚠️ [警告] 严重停滞！超过 {THRESHOLDS['stagnation_steps']} 步没有新纪录。")
        print("   -> 建议：")
        print("      1. 检查是否所有算例都已收敛到理论最优？")
        print("      2. 如果没有，说明陷入深层局部最优，需要增大探索 (Entropy) 或 引入重启机制。")
    else:
        print("✅ [通过] 优化活力充足，持续有新突破。")

    print("-" * 30)

    # 5. 回合长度 (Early Stopping)
    if data["ep_lengths"]:
        avg_len = np.mean(data["ep_lengths"][-10:])
        print(f"🔍 [策略] 近期平均回合长度: {avg_len:.1f}")
        
        # 假设最大长度是 400s 对应的步数 (比如 40)
        # 如果长度很短，说明全是早停；如果很长，说明都在跑满
        if avg_len < 10:
             print("⚠️ [注意] 回合过短，可能过于频繁触发早停。")
        elif avg_len > 35: # 假设 max 是 40
             print("ℹ️ [提示] 大部分回合跑满时间，说明算例较难，Agent 很有耐心。")
        else:
             print("✅ [通过] 动态早停机制工作正常 (有长有短)。")

    print("=" * 60)
    print("💡 综合评价:")
    
    score = 100
    if data["errors"]: score -= 20
    if steps_since_last_best > THRESHOLDS["stagnation_steps"]: score -= 30
    if data["ent_coef"] and data["ent_coef"][-1] < THRESHOLDS["min_entropy"]: score -= 10
    
    print(f"   健康评分: {score}/100")
    if score >= 90: print("   结论: 💪 壮得像头牛！保持现状。")
    elif score >= 70: print("   结论: 🙂 健康，但有小隐患，建议微调。")
    else: print("   结论: 🚑 需要急救！请参考上述建议修改。")

if __name__ == "__main__":
    analyze_health(LOG_FILE_PATH)