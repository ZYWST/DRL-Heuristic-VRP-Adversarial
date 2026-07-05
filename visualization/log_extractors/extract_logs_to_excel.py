import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import os
import glob
import re
import pandas as pd
import argparse

def extract_info_from_filename(filename):
    """
    从文件名中解析 算例名(Problem) 和 种子(Seed)
    支持格式示例: 
    - CHINA_Case9_S101.log
    - CHINA_Case9_seed101.log
    - CHINA_Case9_S101_20240120.log
    """
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # 尝试匹配 _S{数字} 或 _seed{数字}
    match = re.search(r'^(.*)_(?:S|seed)(\d+)', name_without_ext, re.IGNORECASE)
    
    if match:
        problem_name = match.group(1)
        # 有时候名字里还会有下划线结尾，去掉它
        if problem_name.endswith('_'):
            problem_name = problem_name[:-1]
        seed = int(match.group(2))
        return problem_name, seed
    
    # 如果匹配失败，返回原始文件名作为算例名
    return name_without_ext, 0

def get_final_fitness(file_path):
    """
    [修改点] 读取文件，查找 'Final Fitness:' 后面的数值
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 正则解释：
            # Final Fitness: -> 匹配字面量
            # \s* -> 匹配任意数量的空格
            # (-?[\d\.]+)    -> 捕获组：可选负号，数字和小数点
            matches = re.findall(r"Final Fitness:\s*(-?[\d\.]+)", content)
            
            if matches:
                # 取列表最后一个元素，并转为 float
                return float(matches[-1])
            else:
                return None
    except Exception as e:
        print(f"⚠️ 读取文件出错 {file_path}: {e}")
        return None

def main():
    # 1. 设置参数
    parser = argparse.ArgumentParser(description="VNS 日志提取工具")
    parser.add_argument('--folder', type=str, default="./batch_logs_baseline", help='日志所在的文件夹路径')
    parser.add_argument('--output', type=str, default="vns_results.xlsx", help='输出 Excel 文件名')
    args = parser.parse_args()

    log_dir = args.folder
    output_file = args.output

    if not os.path.exists(log_dir):
        # 尝试模糊查找
        candidates = glob.glob("batch_logs*")
        if candidates:
            print(f"⚠️ 指定目录 '{log_dir}' 不存在，自动切换到: {candidates[0]}")
            log_dir = candidates[0]
        else:
            print(f"❌ 错误: 找不到日志目录 '{log_dir}'")
            return

    # 2. 递归查找所有 .log 文件
    log_files = glob.glob(os.path.join(log_dir, "**/*.log"), recursive=True)
    
    if not log_files:
        print(f"❌ 在 '{log_dir}' 下未找到 .log 文件")
        return

    print(f"📂 找到 {len(log_files)} 个日志文件，开始提取 'Final Fitness'...")

    data_list = []

    # 3. 遍历提取
    for file_path in log_files:
        # 排除 fake 报告的日志（如果有的话，通常 .log 是运行日志，fake 是 .txt 报告，这里以防万一）
        if "fake" in file_path: continue

        # 解析文件名
        problem, seed = extract_info_from_filename(file_path)
        
        # 提取内容
        fit_val = get_final_fitness(file_path)
        
        if fit_val is not None:
            data_list.append({
                "Problem": problem,
                "Seed": seed,
                "FinalFitness": fit_val
            })
            # print(f"   ✅ {problem} | S{seed} -> {fit_val}")
        else:
            # 如果没找到关键字，可能运行报错了或者没跑完
            # print(f"   ⚠️ 无数据: {os.path.basename(file_path)}")
            pass

    if not data_list:
        print("❌ 未提取到任何有效数据 (请检查日志中是否包含 'Final Fitness:')。")
        return

    # 4. 创建 DataFrame 并转换格式
    df = pd.DataFrame(data_list)
    
    # 数据透视：行=Problem, 列=Seed, 值=FinalFitness
    # aggfunc='last' 确保如果有重复运行，取最新的
    pivot_df = df.pivot_table(index='Problem', columns='Seed', values='FinalFitness', aggfunc='last')
    
    # 排序
    pivot_df.sort_index(inplace=True)
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    print("-" * 40)
    print("📊 提取结果预览:")
    print(pivot_df.head())
    print("-" * 40)

    # 5. 保存到 Excel
    try:
        pivot_df.to_excel(output_file)
        print(f"✅ 成功保存至: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ 保存 Excel 失败 (请检查文件是否被打开): {e}")

if __name__ == "__main__":
    main()