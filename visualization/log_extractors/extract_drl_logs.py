import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import glob
import re
import pandas as pd
import argparse
from collections import defaultdict

def extract_problem_name(filename):
    """
    只提取算例名，忽略种子
    """
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # 尝试去掉 _S101, _seed2026 这种后缀，只保留前面部分作为算例名
    # 匹配模式：任意字符 + (_S数字 或 _seed数字)
    match = re.search(r'^(.*)_(?:S|seed)\d+', name_without_ext, re.IGNORECASE)
    
    if match:
        problem_name = match.group(1)
        if problem_name.endswith('_'): problem_name = problem_name[:-1]
        return problem_name
    
    # 如果没匹配到种子格式，尝试去掉最后一段 _数字 (防止是时间戳或编号)
    # 如果你也想保留纯文件名作为算例名，可以直接返回 name_without_ext
    return name_without_ext

def get_best_value_universal(file_path):
    """
    全能数值提取器
    """
    content = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            return None

    # 定义三种匹配策略
    strategies = [
        # 1. DRL 新版: 1234.56 -> 🎯NEW BEST! (或者没有🎯)
        r"(-?[\d\.]+)\s*->\s*(?:🎯)?NEW BEST!",
        # 2. DRL 旧版: -> (Best: 1234.56)
        r"->\s*\(Best:\s*(-?[\d\.]+)\)",
        # 3. Baseline: Final Fitness: 1234.56
        r"Final Fitness:\s*(-?[\d\.]+)"
    ]

    for pattern in strategies:
        matches = re.findall(pattern, content)
        if matches:
            try:
                # 取最后一个匹配到的值
                return float(matches[-1])
            except ValueError:
                continue
    return None

def main():
    parser = argparse.ArgumentParser(description="灵活日志提取工具 (忽略种子对齐)")
    parser.add_argument('--folder', type=str, default="./batch_logs", help='日志文件夹路径')
    parser.add_argument('--output', type=str, default="flexible_summary.xlsx", help='输出 Excel 文件名')
    args = parser.parse_args()

    log_dir = args.folder
    output_file = args.output

    if not os.path.exists(log_dir):
        candidates = glob.glob("batch_logs*")
        if candidates:
            print(f"⚠️ 目录 '{log_dir}' 不存在，切换到: {candidates[0]}")
            log_dir = candidates[0]
        else:
            print(f"❌ 找不到日志目录: {log_dir}")
            return

    log_files = glob.glob(os.path.join(log_dir, "**/*.log"), recursive=True)
    print(f"📂 找到 {len(log_files)} 个日志文件，开始提取...\n")

    # 使用字典存储： Key=算例名, Value=[数值1, 数值2, ...]
    results = defaultdict(list)
    
    success_count = 0

    for file_path in log_files:
        # 排除 fake 报告
        if "fake" in file_path: continue

        problem = extract_problem_name(file_path)
        val = get_best_value_universal(file_path)
        
        if val is not None:
            results[problem].append(val)
            success_count += 1
            # print(f"✅ {problem:<20} | {val:.2f}") # 刷屏可注释
        else:
            pass

    if not results:
        print("❌ 未提取到任何有效数据。")
        return

    print(f"📊 提取完毕，共获取 {success_count} 个有效数值。正在整理...")

    # 整理数据：将列表转换为 DataFrame 行
    # 为了表格整齐，我们先对每个算例的数值列表进行排序（降序：最好的在最前）
    formatted_data = []
    max_cols = 0
    
    sorted_problems = sorted(results.keys())
    
    for prob in sorted_problems:
        vals = results[prob]
        # 降序排序，这样 Result_1 就是该算例目前跑出的最高分
        vals.sort(reverse=True) 
        
        row = {"Problem": prob}
        for i, v in enumerate(vals):
            row[f"Run_{i+1}"] = v
        
        formatted_data.append(row)
        if len(vals) > max_cols:
            max_cols = len(vals)

    # 创建 DataFrame
    df = pd.DataFrame(formatted_data)
    
    # 调整列顺序: Problem, Run_1, Run_2, ...
    cols = ["Problem"] + [f"Run_{i+1}" for i in range(max_cols)]
    df = df.reindex(columns=cols)

    print("-" * 40)
    print("📋 结果预览 (Top 5 Rows):")
    print(df.head())
    print("-" * 40)

    try:
        df.to_excel(output_file, index=False)
        print(f"✅ Excel 已保存至: {os.path.abspath(output_file)}")
        print(f"   (提示: 数据已按从大到小排序，Run_1 即为该算例的历史最优)")
    except Exception as e:
        print(f"❌ 保存 Excel 失败: {e}")

if __name__ == "__main__":
    main()