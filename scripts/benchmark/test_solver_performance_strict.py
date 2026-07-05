import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
# test_solver_performance_viz.py (V6: 流量报告修正 + 开关配置)
import logging
import sys
import time
import os
import json
import argparse
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from src.env.hyper_config import GraspConfig
from src.algorithms.solver_coreDRL_train import (
    ParallelGraspOptimizer,
    init_worker,
    ProblemData,
    PathLibrary,
    Task,
    SimulatedAnnealing,
    Solution
)
from src.utils.solution_auditor import SolutionAuditor # [NEW] 导入审计器

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TestSolver")

def reconstruct_and_save_solution(optimizer, problem_path, output_file="solution_report.txt", seed=None, timestamp=None):
    """
    重构最优解，采用与 Solver 一致的“流量去重”和“严格中断”逻辑。
    """
    logger.info("="*50)
    logger.info("正在重构最优解详细路径及中断分析...")
    
    best_sol_dict = getattr(optimizer, 'best_overall_solution_assignments', None)
    if not best_sol_dict:
        logger.warning("未找到有效解，无法保存路径。")
        return

    if not os.path.exists(problem_path):
        logger.error(f"无法找到算例文件: {problem_path}")
        return
        
    local_problem_data = ProblemData(problem_path, '')
    H = local_problem_data.H
    U_limit = local_problem_data.U
    
    local_problem_data.tasks_by_vehicle = {v_id: [] for v_id in local_problem_data.vehicles.keys()}
    local_problem_data.task_map = {v_id: {} for v_id in local_problem_data.vehicles.keys()}
    
    assignments_data = best_sol_dict.get('assignments', {})
    for v_id_str, tasks_list in assignments_data.items():
        v_id = int(v_id_str)
        local_problem_data.tasks_by_vehicle[v_id] = [] 
        for t_dict in tasks_list:
            task = Task(t_dict['v'], t_dict['p'], t_dict['d'], t_dict['g'], t_dict['q'])
            local_problem_data.tasks_by_vehicle[v_id].append(task)
            local_problem_data.task_map[v_id][task.delivery_node] = task

    reconstruct_config = optimizer.config 
    path_lib = PathLibrary(local_problem_data, reconstruct_config)
    
    final_solution = Solution(local_problem_data, reconstruct_config.SA_M_PATHS)    
    saved_node_sequences = best_sol_dict.get('node_sequences')
    saved_path_choices = best_sol_dict.get('path_choices')
    
    if saved_node_sequences and saved_path_choices:
        final_solution.node_sequences = saved_node_sequences
        final_solution.path_choices = saved_path_choices
        for v_id in local_problem_data.vehicles:
            if v_id not in final_solution.node_sequences:
                final_solution.node_sequences[v_id] = []
                final_solution.path_choices[v_id] = []
    else:
        logger.warning("未检测到路径信息，回退到模拟...")
        reconstruct_config.SA_METROPOLIS_LEN = 100
        sa = SimulatedAnnealing(local_problem_data, reconstruct_config, path_lib)
        final_solution, _ = sa.run()
    
    # 获取严格模式配置
    strict_disruption = getattr(reconstruct_config, 'STRICT_DISRUPTION', False)

    # =========================================================================
    # 阶段 A: 计算流量 (修正版: 去重逻辑)
    # =========================================================================
    total_edge_flows = defaultdict(float)
    full_paths_map = {} # 存储每个车的完整物理路径，供严格模式判定使用
    
    for v_id, node_seq in final_solution.node_sequences.items():
        if not node_seq: continue
        
        veh_info = local_problem_data.vehicles[v_id]
        current_node = veh_info['L']
        physical_path_nodes = [current_node]
        
        current_load = {g: 0.0 for g in local_problem_data.goods}
        processed_pickups = set()
        processed_deliveries = set()
        
        # [Fix] 流量记忆字典
        vehicle_edge_max_load = {}

        tasks = local_problem_data.tasks_by_vehicle.get(v_id, [])
        for t in tasks:
            if t.pickup_node == current_node and t not in processed_pickups:
                current_load[t.good_id] += t.quantity
                processed_pickups.add(t)
            if t.delivery_node == current_node and t not in processed_deliveries:
                current_load[t.good_id] -= t.quantity
                processed_deliveries.add(t)

        sequence_to_visit = node_seq + [veh_info['L']]
        path_choices = final_solution.path_choices[v_id]
        
        for i, next_node in enumerate(sequence_to_visit):
            choice_idx = path_choices[i] if i < len(path_choices) else 0
            paths = path_lib.get_k_shortest_paths(current_node, next_node, reconstruct_config.SA_M_PATHS)
            if not paths: continue 
            
            safe_choice = choice_idx % len(paths)
            leg_path = paths[safe_choice]['path']
            physical_path_nodes.extend(leg_path[1:])
            
            # [Fix] 流量计算: 仅累加增量
            for j in range(len(leg_path) - 1):
                u, v = leg_path[j], leg_path[j+1]
                
                flow_to_add = 0.0
                for g_id, qty in current_load.items():
                    if qty <= 1e-6: continue
                    prev_max = vehicle_edge_max_load.get((u, v, g_id), 0.0)
                    if qty > prev_max:
                        increment = qty - prev_max
                        flow_to_add += increment
                        vehicle_edge_max_load[(u, v, g_id)] = qty
                
                if flow_to_add > 0:
                    total_edge_flows[(u, v)] += flow_to_add
            
            current_node = next_node
            for t in tasks:
                if t.pickup_node == next_node and t not in processed_pickups:
                    current_load[t.good_id] += t.quantity
                    processed_pickups.add(t)
                if t.delivery_node == next_node and t not in processed_deliveries:
                    current_load[t.good_id] -= t.quantity
                    processed_deliveries.add(t)

        full_paths_map[v_id] = physical_path_nodes

    # 确定中断边
    disrupted_edges = set()
    sorted_edges = sorted(total_edge_flows.items(), key=lambda item: item[1], reverse=True)
    top_u_edges_info = []
    for i in range(min(U_limit, len(sorted_edges))):
        edge_tuple = sorted_edges[i][0]
        flow_val = sorted_edges[i][1]
        disrupted_edges.add(edge_tuple)
        top_u_edges_info.append((edge_tuple, flow_val))

    # =========================================================================
    # 阶段 B: 生成详细报告 (修正版: 严格中断判定)
    # =========================================================================
    report_lines = []
    mode_str = "严格模式 (整车失效)" if strict_disruption else "默认模式 (后续无效)"
    report_lines.append(f"=== 最优解详细报告 ===")
    report_lines.append(f"算例: {os.path.basename(problem_path)}")
    report_lines.append(f"时间窗 H: {H}")
    report_lines.append(f"允许中断数 U: {U_limit}")
    report_lines.append(f"中断判定策略: {mode_str}\n")
    
    report_lines.append(f"=== ⚠️ 识别到的中断边 (流量 Top-{len(top_u_edges_info)}) ===")
    for idx, (edge, flow) in enumerate(top_u_edges_info):
        report_lines.append(f"{idx+1}. 边 {edge[0]} -> {edge[1]} (流量: {flow:.2f})")
    report_lines.append("")
    
    total_obj_value = 0.0
    total_dist_all = 0.0
    
    sorted_vehicle_ids = sorted(local_problem_data.vehicles.keys())
    
    for v_id in sorted_vehicle_ids:
        veh_info = local_problem_data.vehicles[v_id]
        depot = veh_info['L']
        speed = veh_info['v']
        node_seq = final_solution.node_sequences.get(v_id, [])
        
        report_lines.append(f"--- 车辆 ID: {v_id} (Depot: {depot}, Speed: {speed}) ---")
        
        if not node_seq:
            report_lines.append("   💤 该车辆闲置 (未分配任务)")
            report_lines.append("")
            continue

        # [新增] 严格模式下的整车中断检查
        is_strictly_broken = False
        strict_break_loc = None
        if strict_disruption:
            full_path = full_paths_map.get(v_id, [])
            for j in range(len(full_path) - 1):
                e = (full_path[j], full_path[j+1])
                if e in disrupted_edges:
                    is_strictly_broken = True
                    strict_break_loc = f"{e[0]}->{e[1]}"
                    break

        current_node = depot
        current_time = 0.0
        current_load = {g: 0.0 for g in local_problem_data.goods}
        
        # 局部中断标记 (用于默认模式)
        is_locally_disrupted = False
        first_disruption_loc = None
        
        path_str_list = [f"Depot({depot})"]
        
        sequence_to_visit = node_seq + [depot]
        path_choices = final_solution.path_choices[v_id]
        
        for i, next_node in enumerate(sequence_to_visit):
            choice_idx = path_choices[i] if i < len(path_choices) else 0
            paths = path_lib.get_k_shortest_paths(current_node, next_node, reconstruct_config.SA_M_PATHS)
            
            if not paths:
                path_str_list.append(f" -> UNREACHABLE({next_node})")
                break
                
            safe_choice = choice_idx % len(paths)
            chosen_leg = paths[safe_choice]
            dist = chosen_leg['dist']
            travel_time = (dist / speed) * 60 
            total_dist_all += dist
            leg_path = chosen_leg['path']
            
            # 检查局部路段中断
            leg_broken_at = None
            if not is_locally_disrupted:
                for j in range(len(leg_path) - 1):
                    e = (leg_path[j], leg_path[j+1])
                    if e in disrupted_edges:
                        is_locally_disrupted = True
                        leg_broken_at = e
                        first_disruption_loc = f"{e[0]}->{e[1]}"
                        break
            
            current_time += travel_time
            
            tasks = local_problem_data.tasks_by_vehicle.get(v_id, [])
            p_tasks = [t for t in tasks if t.pickup_node == next_node]
            for t in p_tasks: current_load[t.good_id] += t.quantity
            
            d_tasks = [t for t in tasks if t.delivery_node == next_node]
            delivery_info_strs = []
            
            for t in d_tasks:
                current_load[t.good_id] -= t.quantity
                weight = local_problem_data.weights.get(next_node, {}).get(t.good_id, 1.0)
                
                # [核心逻辑] 判定是否有效
                valid = False
                fail_reason = ""
                
                if current_time > H:
                    fail_reason = "超时"
                elif strict_disruption and is_strictly_broken:
                    fail_reason = f"严格中断于{strict_break_loc}"
                elif (not strict_disruption) and is_locally_disrupted:
                    fail_reason = f"路径中断于{first_disruption_loc}"
                else:
                    valid = True
                
                if valid:
                    contrib = weight * t.quantity * (H - current_time)
                    total_obj_value += contrib
                    delivery_info_strs.append(f"[卸 G{t.good_id}:{t.quantity:.0f} | ✅贡献:{contrib:.1f}]")
                else:
                    delivery_info_strs.append(f"[卸 G{t.good_id}:{t.quantity:.0f} | ❌无效({fail_reason})]")

            node_type = "Node"
            if p_tasks: node_type = "Supply"
            elif d_tasks: node_type = "Demand"
            
            action_str = ""
            if p_tasks: action_str += " " + "".join([f"[装 G{t.good_id}:{t.quantity:.0f}]" for t in p_tasks])
            if d_tasks: action_str += " " + "".join(delivery_info_strs)
                
            step_marker = "->"
            if leg_broken_at: step_marker = f"-[❌断于{leg_broken_at[0]}-{leg_broken_at[1]}]->"
            
            path_str_list.append(f"\n   {step_marker} {node_type}({next_node}) @{current_time:.1f}min {action_str}")
            current_node = next_node
            
        full_physical_path_str = "->".join(map(str, full_paths_map.get(v_id, [])))
        
        report_lines.append("逻辑路径详情:" + "".join(path_str_list))
        report_lines.append(f"   🛣️  完整物理路径: {full_physical_path_str}")
        
        if is_strictly_broken and strict_disruption:
            report_lines.append(f"   🛑 [严格模式] 车辆路径在 {strict_break_loc} 中断，全车失效！")
        elif is_locally_disrupted:
            report_lines.append(f"   ⚠️ [默认模式] 车辆路径在 {first_disruption_loc} 中断，后续失效。")
        else:
            report_lines.append(f"   ✅ 车辆路径完整有效。")
        report_lines.append("")

    report_lines.append(f"==========================================")
    report_lines.append(f"总行驶距离: {total_dist_all:.2f} km")
    report_lines.append(f"总目标函数值: {total_obj_value:.2f}")
    report_lines.append(f"==========================================")
    
    # ... (审计部分代码保持不变) ...
    # 将算例名与时间戳和种子加入输出文件名，避免覆盖并便于追踪
    try:
        case_name = os.path.splitext(os.path.basename(problem_path))[0]
        base, ext = os.path.splitext(output_file)
        if not ext:
            ext = '.txt'
        timestamp_part = f"_{timestamp}" if timestamp else ""
        seed_part = f"_seed{seed}" if seed is not None else ""
        output_file_with_case = f"{base}_{case_name}{timestamp_part}{seed_part}{ext}"

        with open(output_file_with_case, "w", encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        logger.info(f"✅ 详细解报告已保存至: {output_file_with_case}")
        # 在终端打印报告中的“总目标函数值”以便快速查看
        try:
            logger.info(f"报告中的 总目标函数值: {total_obj_value:.2f}")
        except Exception:
            # 如果 total_obj_value 未定义或出错，忽略打印
            pass
    except Exception as e:
        logger.error(f"保存文件失败: {e}")

# ... save_fitness_plot 函数保持不变 ...
def save_fitness_plot(history_data, output_file="fitness_convergence.png", problem_path=None, timestamp=None, seed=None):
    if not history_data['avg']: return
    logger.info("正在绘制适应度收敛曲线...")
    generations = range(1, len(history_data['avg']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, history_data['avg'], label='Average Fitness', color='blue', linestyle='--', alpha=0.7)
    plt.plot(generations, history_data['gen_best'], label='Gen Best Fitness', color='orange', alpha=0.8)
    plt.plot(generations, history_data['global_best'], label='Global Best Fitness', color='red', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Optimization Process: Fitness Convergence')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    try:
        # 如果提供了算例名或时间戳，则把它们加入文件名
        base, ext = os.path.splitext(output_file)
        if not ext:
            ext = '.png'
        case_part = f"_{os.path.splitext(os.path.basename(problem_path))[0]}" if problem_path else ""
        ts_part = f"_{timestamp}" if timestamp else ""
        seed_part = f"_seed{seed}" if seed is not None else ""
        output_file_with_case = f"{base}{case_part}{ts_part}{seed_part}{ext}"

        plt.savefig(output_file_with_case, dpi=150)
        logger.info(f"✅ 适应度曲线图已保存至: {output_file_with_case}")
    except Exception as e:
        logger.error(f"保存图表失败: {e}")
    finally:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="运行严格模式的求解器性能测试")
    # 添加算例路径参数
    parser.add_argument('--problem', type=str, default="CHINA_Case7.dat", help='算例文件路径 (.dat)')
    parser.add_argument('--total-time', type=float, default=None, help='总求解时间上限（秒）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    args = parser.parse_args()

    test_problem_path = args.problem
    
    # 简单的文件存在性检查
    if not os.path.exists(test_problem_path):
        # 如果指定的文件不存在，尝试在 problem_instances 文件夹找
        import glob
        dats = glob.glob("./problem_instances/*.dat")
        if dats: 
            logger.warning(f"指定文件 {test_problem_path} 不存在，默认使用: {dats[0]}")
            test_problem_path = dats[0]
        else:
            logger.error(f"找不到算例文件: {test_problem_path}")
            return

    config = GraspConfig(
        PROBLEM_DATA_PATH=test_problem_path,
        NUM_WORKERS=None, 
        POPULATION_SIZE=24,
        SA_METROPOLIS_LEN_BASELINE=50, 
        SA_ALPHA=0.9,
        SA_M_PATHS=8,
        SA_PENALTY_FACTOR = 1e4,
        SA_MAX_ALLOWED_DUPLICATES = 10,
        SA_EXCESS_DUPLICATE_PENALTY = 1e-5
    )
    config.SA_METROPOLIS_LEN = 180
    config.SA_SOFT_CONGESTION_PENALTY_FACTOR = 8.0
    # [新增] 配置文件开关
    # True: 严格模式 (整车失效)
    # False: 默认模式 (现状，仅后续失效)
    config.STRICT_DISRUPTION = True   # <--- 在这里修改以切换模式

    logger.info("="*50)
    logger.info(f"开始性能测试: {os.path.basename(test_problem_path)}")
    logger.info(f"中断判定策略: {'严格模式 (整车失效)' if config.STRICT_DISRUPTION else '默认模式 (现状)'}")
    logger.info("="*50)

    # 处理随机种子
    if args.seed is None:
        seed_value = int(time.time())
        logger.info(f"未提供种子，使用时间戳生成种子: {seed_value}")
    else:
        seed_value = args.seed
        logger.info(f"使用提供的随机种子: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)

    optimizer = ParallelGraspOptimizer(config, logger)
    history = {'avg': [], 'gen_best': [], 'global_best': []}
    
    NUM_GENERATIONS = 200
    total_time = 0
    total_time_limit = args.total_time
    completed_gens = 0
    
    try:
        for gen in range(1, NUM_GENERATIONS + 1):
            
            t0 = time.time()
            # 运行一代
            stats = optimizer.run_one_generation(gen, total_generations=NUM_GENERATIONS)
            t1 = time.time()
            elapsed = t1 - t0
            total_time += elapsed
            completed_gens = gen
            
            gen_best_fitness = stats['best_fitness_in_gen']
            global_best_fitness = stats['new_overall_best_fitness']
            current_avg_fitness = stats.get('mean_fitness_in_gen', -1e5)
            internal_time = stats['gen_time_seconds']

            history['avg'].append(current_avg_fitness)
            history['gen_best'].append(gen_best_fitness)
            history['global_best'].append(global_best_fitness)

            logger.info(f"   [外部计时] 耗时: {elapsed:.4f}s")
            logger.info(f"   [内部计时] 耗时: {internal_time:.4f}s")
            logger.info(f"   [Stats] Avg: {current_avg_fitness:.2f} | GenBest: {gen_best_fitness:.2f} | GlobalBest: {global_best_fitness:.2f}")
            
            if gen_best_fitness <= -99999:
                logger.warning("   ⚠️ 警告: 本代未找到可行解")
            # 检查外部总时间上限
            if total_time_limit is not None and total_time >= total_time_limit:
                logger.info(f"达到总时间上限 {total_time_limit}s，提前终止测试（已完成 {completed_gens} 代）")
                break
                
    except KeyboardInterrupt:
        logger.info("测试中断。")
    finally:
        # 保存时把时间戳与种子写入文件名
        ts = time.strftime("%Y%m%d_%H%M%S")
        reconstruct_and_save_solution(optimizer, test_problem_path, seed=seed_value, timestamp=ts)
        save_fitness_plot(history, problem_path=test_problem_path, timestamp=ts, seed=seed_value)
        optimizer.shutdown()
    
    # auditor = SolutionAuditor(pd, pl, best_sol)
    # auditor.run_audit()

    logger.info("="*50)
    if completed_gens == 0:
        logger.info("测试未完成任何代，平均耗时无法计算")
    else:
        logger.info(f"测试完成。平均每代耗时: {total_time / completed_gens:.4f}s (共完成 {completed_gens} 代)")
    logger.info("="*50)

if __name__ == "__main__":
    main()