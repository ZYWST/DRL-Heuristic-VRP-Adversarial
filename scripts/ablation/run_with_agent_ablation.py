import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
# run_with_agent.py (Final Version: With Detailed Reporting)

import time
import os
import sys
import logging
import numpy as np
import argparse
import pickle
import random
from collections import defaultdict
from typing import List, Dict, Any, Optional

# --- RL相关导入 ---
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import set_random_seed

# --- 导入配置与求解器核心组件 ---
from src.env.hyper_config import HyperConfig
from src.algorithms.solver_coreDRL_train import (
    ParallelGraspOptimizer,
    GraspConfig,
    ProblemData,
    PathLibrary,        # [新增] 用于重构路径
    Task,               # [新增] 用于重构任务对象
    Solution,           # [新增] 用于构建解对象
    SimulatedAnnealing  # [新增] 用于回退逻辑
)

import warnings
# 屏蔽所有 Numba 的废弃和性能警告
# 必须放在 import solver_coreDRL_train 之前
try:
    from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
except ImportError:
    # 如果 Numba 版本较老，可能没有这些特定的异常类，则使用通用的忽略
    pass

# 额外的宽泛过滤，确保万无一失
warnings.filterwarnings("ignore", module="numba")
warnings.filterwarnings("ignore", message=".*reflected set.*")

def setup_logger(config: GraspConfig) -> logging.Logger:
    logger = logging.getLogger("IntelligentSolver")
    logger.setLevel(config.LOG_LEVEL)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(config.LOG_FORMAT, datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

import torch
torch.cuda.empty_cache()
# ==============================================================================
# [新增] 详细报告生成器 (移植自测试脚本)
# ==============================================================================
def generate_detailed_report(optimizer, problem_path, output_file="solution_report.txt", seed=None):
    """
    重构最优解，生成包含流量分析、中断判定和详细路径的报告。
    """
    logger = logging.getLogger("IntelligentSolver")
    logger.info("="*50)
    logger.info("正在重构最优解详细路径及中断分析...")
    
    best_sol_dict = getattr(optimizer, 'best_overall_solution_assignments', None)
    if not best_sol_dict:
        logger.warning("❌ 未找到有效解，无法生成详细报告。")
        return

    if not os.path.exists(problem_path):
        logger.error(f"无法找到算例文件: {problem_path}")
        return
        
    # 1. 重建局部 ProblemData (避免污染全局)
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

    # 2. 准备重构配置
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
        logger.warning("未检测到路径信息，尝试使用 SA 模拟回退...")
        reconstruct_config.SA_METROPOLIS_LEN = 100
        sa = SimulatedAnnealing(local_problem_data, reconstruct_config, path_lib)
        final_solution, _ = sa.run()
    
    strict_disruption = getattr(reconstruct_config, 'STRICT_DISRUPTION', False)

    # 3. 计算流量 (去重逻辑)
    total_edge_flows = defaultdict(float)
    full_paths_map = {} 
    
    for v_id, node_seq in final_solution.node_sequences.items():
        if not node_seq: continue
        
        veh_info = local_problem_data.vehicles[v_id]
        current_node = veh_info['L']
        physical_path_nodes = [current_node]
        
        current_load = {g: 0.0 for g in local_problem_data.goods}
        processed_pickups = set()
        processed_deliveries = set()
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

    # 4. 确定中断边
    disrupted_edges = set()
    sorted_edges = sorted(total_edge_flows.items(), key=lambda item: item[1], reverse=True)
    top_u_edges_info = []
    for i in range(min(U_limit, len(sorted_edges))):
        edge_tuple = sorted_edges[i][0]
        flow_val = sorted_edges[i][1]
        disrupted_edges.add(edge_tuple)
        top_u_edges_info.append((edge_tuple, flow_val))

    # 5. 生成文本报告
    report_lines = []
    mode_str = "严格模式 (整车失效)" if strict_disruption else "默认模式 (后续无效)"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    report_lines.append(f"=== 最优解详细报告 (Generated: {ts}) ===")
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

    # 6. 保存文件
    try:
        case_name = os.path.splitext(os.path.basename(problem_path))[0]
        base, ext = os.path.splitext(output_file)
        if not ext: ext = '.txt'
        seed_part = f"_seed{seed}" if seed is not None else ""
        
        # 加上时间戳避免覆盖
        ts_file = time.strftime("%Y%m%d_%H%M%S")
        output_file_with_case = f"{base}_{case_name}_{ts_file}{seed_part}{ext}"

        with open(output_file_with_case, "w", encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        logger.info(f"✅ [Result] 详细解报告已保存至: {output_file_with_case}")
        # 终端打印报告中的“总目标函数值”以便快速查看
        try:
            logger.info(f"总目标函数值: {total_obj_value:.2f}")
        except Exception:
            pass
    except Exception as e:
        logger.error(f"保存文件失败: {e}")

def generate_lenient_report(optimizer_or_sol_dict, problem_path, output_file="solution_report_lenient.txt", seed=None):
    """
    【宽容版报告生成器 - 修复版】
    完全按照 Solver 内部的迭代逻辑来计算流量、断边和目标函数。
    使用纯 Python 逻辑替代 Numba 函数，确保稳定性。
    """
    logger = logging.getLogger("LenientReporter")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    
    logger.info("="*50)
    logger.info("正在生成宽容版报告 (Aligning with Solver Logic)...")
    
    # --- 1. 提取最优解 ---
    if isinstance(optimizer_or_sol_dict, dict):
        best_sol_dict = optimizer_or_sol_dict
        reconstruct_config = GraspConfig()
        reconstruct_config.PROBLEM_DATA_PATH = problem_path
    else:
        best_sol_dict = getattr(optimizer_or_sol_dict, 'best_overall_solution_assignments', None)
        reconstruct_config = optimizer_or_sol_dict.config

    if not best_sol_dict:
        logger.error("❌ 未找到有效解。")
        return

    # --- 2. 重建环境 ---
    if not os.path.exists(problem_path):
        logger.error(f"无法找到算例文件: {problem_path}")
        return
        
    pd = ProblemData(problem_path, '', matrix_filepath="")
    pl = PathLibrary(pd, reconstruct_config)
    sol = Solution(pd, reconstruct_config.SA_M_PATHS)
    
    # 填充解
    assignments_data = best_sol_dict.get('assignments', {})
    sol.node_sequences = best_sol_dict.get('node_sequences', {})
    sol.path_choices = best_sol_dict.get('path_choices', {})
    
    # 填充 pd 任务映射 (Solver 计算必须)
    pd.tasks_by_vehicle = {v: [] for v in pd.vehicles}
    pd.task_map = {v: {} for v in pd.vehicles}
    
    for v_str, tasks in assignments_data.items():
        v = int(v_str)
        for t_data in tasks:
            t_obj = Task(t_data['v'], t_data['p'], t_data['d'], t_data['g'], t_data['q'])
            pd.tasks_by_vehicle[v].append(t_obj)
            pd.task_map[v][t_obj.delivery_node] = t_obj

    # --- 3. 调用 Solver 评估 (复现高分) ---
    sa_evaluator = SimulatedAnnealing(pd, reconstruct_config, pl)
    sa_evaluator.curr = sol
    
    # 这一步计算出的 fitness 和 total_util 就是你要的“大数”
    fitness = sa_evaluator.evaluate_full(sol)
    
    # 提取关键数据
    disrupted_edges = sol.cached_disrupted_edges  # 这是 Solver 认定的断边集合
    total_utility = sol.total_util                # 这是不含惩罚的高分
    
    logger.info(f"Solver Calculated Total Utility: {total_utility:.4f}")

    # =========================================================
    # [新增] 纯 Python 版的中断检查函数 (替代 Numba)
    # =========================================================
    def check_disruption_python(path_segment, disrupted_set):
        """
        检查路径段是否包含任何中断边。
        path_segment: list 或 numpy array
        disrupted_set: set of tuples
        """
        if len(path_segment) < 2: return False
        for i in range(len(path_segment) - 1):
            u = int(path_segment[i])
            v = int(path_segment[i+1])
            if (u, v) in disrupted_set:
                return True
        return False
    # =========================================================

    # --- 4. 生成报告 ---
    report_lines = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    report_lines.append(f"=== 最优解详细报告 (Solver Logic / Lenient Mode) ===")
    report_lines.append(f"生成时间: {ts}")
    report_lines.append(f"算例: {os.path.basename(problem_path)}")
    report_lines.append(f"Random Seed: {seed}")
    report_lines.append(f"中断策略: 局部判定 (Local Disruption)")
    report_lines.append(f"流量计算: 车辆计数 (Vehicle Count)")
    report_lines.append("-" * 50)
    
    # 输出中断边
    report_lines.append(f"=== ⚠️ 识别到的中断边 (Top-{pd.U}) ===")
    sorted_edges = sorted(
        [(e, count) for e, count in sol.global_edge_counts.items() if e in disrupted_edges],
        key=lambda x: x[1], reverse=True
    )
    for i, (edge, count) in enumerate(sorted_edges):
        report_lines.append(f"{i+1}. 边 {edge[0]} -> {edge[1]} (车次: {count})")
    report_lines.append("")

    # --- 5. 路径回放 ---
    total_dist_all = 0.0
    sorted_v_ids = sorted(pd.vehicles.keys())
    
    for v_id in sorted_v_ids:
        if v_id not in sol.node_sequences or not sol.node_sequences[v_id]:
            continue
            
        veh_info = pd.vehicles[v_id]
        speed = max(0.1, veh_info['v'])
        
        # 检查 Solver 是否判定该车无效 (极其罕见，通常是死循环或超载)
        v_stat = sol.v_stats.get(v_id, {})
        if not v_stat.get('valid', False):
            report_lines.append(f"--- 车辆 {v_id}: ❌ 路径无效 (Solver Rejected) ---")
            continue
            
        report_lines.append(f"--- 车辆 ID: {v_id} (Depot: {veh_info['L']}) ---")
        
        curr_node = veh_info['L']
        curr_time = 0.0
        path_str_list = [f"Depot({curr_node})"]
        
        seq = sol.node_sequences[v_id]
        choices = sol.path_choices[v_id]
        full_seq = seq + [curr_node]
        
        path_nodes_accum = [curr_node] 
        tasks = pd.tasks_by_vehicle.get(v_id, [])
        processed_pickups = set()
        
        for i, next_node in enumerate(full_seq):
            choice = choices[i] if i < len(choices) else 0
            paths = pl.get_k_shortest_paths(curr_node, next_node, reconstruct_config.SA_M_PATHS)
            
            if not paths:
                path_str_list.append(f" -> UNREACHABLE")
                break
                
            leg = paths[choice % len(paths)]
            leg_dist = leg['dist']
            leg_nodes = leg['path'] # 包含起点和终点
            
            total_dist_all += leg_dist
            curr_time += (leg_dist / speed) * 60
            
            # 这里的 leg_nodes[1:] 是为了避免重复添加起点
            path_nodes_accum.extend(leg_nodes[1:])
            
            # 检查当前路段断裂情况 (仅用于显示⚠️，不影响后续)
            is_leg_broken = check_disruption_python(leg_nodes, disrupted_edges)
            arrow = f"-[⚠️断]->" if is_leg_broken else "->"
            
            action_strs = []
            
            # 装货
            for t in tasks:
                if t.pickup_node == next_node and t not in processed_pickups:
                    action_strs.append(f"装 G{t.good_id}:{t.quantity:.0f}")
                    processed_pickups.add(t)
            
            # 卸货
            if next_node in pd.task_map.get(v_id, {}):
                t = pd.task_map[v_id][next_node]
                if t.delivery_node == next_node:
                    # 复现 Solver 的判定逻辑：回溯路径段
                    try:
                        p_idx = path_nodes_accum.index(t.pickup_node)
                        # 截取 P -> D 的完整路径段
                        relevant_segment = path_nodes_accum[p_idx:]
                        
                        # 使用 Python 版函数检查
                        is_path_clear = not check_disruption_python(relevant_segment, disrupted_edges)
                    except ValueError:
                        is_path_clear = False

                    if curr_time > pd.H:
                        status = "❌超时"
                    elif is_path_clear:
                        # 只有路径通畅才算分
                        score = t.quantity * pd.weights[next_node].get(t.good_id, 1.0) * (pd.H - curr_time)
                        status = f"✅贡献:{score:.1f}"
                    else:
                        status = "⚠️无效(路断)"
                    
                    action_strs.append(f"卸 G{t.good_id}:{t.quantity:.0f}|{status}")

            curr_node = next_node
            actions_joined = f"  [{' '.join(action_strs)}]" if action_strs else ""
            path_str_list.append(f"\n   {arrow} Node({next_node}) @{curr_time:.1f}min {actions_joined}")

        report_lines.append("".join(path_str_list))
        report_lines.append("")

    report_lines.append("="*50)
    report_lines.append(f"Total Distance: {total_dist_all:.2f} km")
    # 强制写入 Solver 计算的高分
    report_lines.append(f"Total Objective Value: {total_utility:.4f}") 
    report_lines.append(f"Solver Fitness (w/ Penalty): {fitness:.4f}")
    report_lines.append("="*50)

    # 保存
    with open(output_file, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"✅ 宽容版报告已保存至: {output_file}")
    logger.info(f"🎯 报告数值已对齐: {total_utility:.4f}")



# ==============================================================================
# 1. 智能体控制器 (Agent Controller)
# ==============================================================================
class AgentController:
    def __init__(self, model_path: str, config: HyperConfig, problem_data: ProblemData):
        self.config = config
        
        # 1. 加载模型
        try:
            self.model = SAC.load(model_path, device='cpu')
            print(f"✅ 模型已加载: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

        # 2. 尝试加载 VecNormalize 统计数据
        vec_path = model_path.replace(".zip", "") + "_vecnormalize.pkl"
        if not os.path.exists(vec_path):
             vec_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")

        self.vec_norm = None
        if os.path.exists(vec_path):
            try:
                with open(vec_path, "rb") as f:
                    self.vec_norm = pickle.load(f)
                print(f"✅ 归一化统计数据已加载: {vec_path}")
                self.vec_norm.training = False 
            except Exception as e:
                print(f"⚠️ 归一化文件加载出错: {e}。")
        else:
            print(f"⚠️ 未找到归一化文件 ({vec_path})。如果是用 VecNormalize 训练的，必须加载此文件！")

        # 3. 初始化状态追踪变量
        self._static_features_normalized = self._calculate_static_features(problem_data)
        self._fitness_history = []
        self._best_overall_fitness = -1e9
        self._steps_at_optimum = 0
        self._last_step_feasible = False
        self._total_elapsed_time = 0.0

    def _normalize_static_feature(self, value: float, bounds: List[float]) -> float:
        min_val, max_val = bounds
        if max_val == min_val: return 0.0
        return 2 * ((value - min_val) / (max_val - min_val)) - 1

    def _calculate_static_features(self, problem_data: ProblemData) -> np.ndarray:
        static_features = np.array([
            len(problem_data.nodes),
            len(problem_data.vehicles),
            len(problem_data.goods),
            sum(abs(d) for demands in problem_data.demands.values() for d in demands.values())
        ], dtype=np.float32)

        bounds = [
            self.config.NODE_COUNT_BOUNDS, 
            self.config.VEHICLE_COUNT_BOUNDS, 
            self.config.GOODS_COUNT_BOUNDS,
            self.config.TOTAL_DEMAND_BOUNDS
        ]
        
        normalized = np.array([
            self._normalize_static_feature(static_features[i], bounds[i])
            for i in range(len(static_features))
        ], dtype=np.float32)
        return normalized

    def _get_observation(self, stats: Dict[str, Any], gen_time: float) -> np.ndarray:
        new_best = stats.get('new_overall_best_fitness', -1e5)
        
        if abs(new_best - self._best_overall_fitness) < 1e-5:
            if new_best > -99999: 
                self._steps_at_optimum += 1
        else:
            self._steps_at_optimum = 0
        
        self._best_overall_fitness = max(self._best_overall_fitness, new_best)
        self._fitness_history.append(self._best_overall_fitness)
        self._total_elapsed_time += gen_time

        time_ratio = self._total_elapsed_time / self.config.MAX_SECONDS_PER_EPISODE
        time_norm = 2 * min(time_ratio, 1.0) - 1

        w_size = self.config.PROGRESS_WINDOW_SIZE
        if len(self._fitness_history) >= w_size:
            past_fit = self._fitness_history[-w_size]
            denom = abs(past_fit) if abs(past_fit) > 1e-6 else 1.0
            imp = (self._best_overall_fitness - past_fit) / denom
            progress_norm = np.tanh(imp * 10.0)
        else:
            progress_norm = 0.0

        max_stag = self.config.EARLY_STOPPING_STABLE_STEPS
        stag_norm = 2 * (min(self._steps_at_optimum, max_stag) / max_stag) - 1
        acc_rate = stats.get('avg_acceptance_rate', 0.5)
        acc_norm = 2 * acc_rate - 1

        best_fit = stats.get('new_overall_best_fitness', -1e-6)
        mean_fit = stats.get('mean_fitness_in_gen', best_fit)
        denom_gap = abs(best_fit) if abs(best_fit) > 1e-6 else 1.0
        gap = (best_fit - mean_fit) / denom_gap
        gap_norm = np.tanh(gap * 5.0)

        init_fit = stats.get('avg_initial_fitness', best_fit)
        final_fit = stats.get('avg_final_fitness', init_fit)
        denom_gain = abs(init_fit) if abs(init_fit) > 1e-6 else 1.0
        gain = (final_fit - init_fit) / denom_gain
        gain_norm = np.tanh(gain * 20.0)

        dynamic_features = np.array([
            time_norm, progress_norm, stag_norm, acc_norm, gap_norm, gain_norm
        ], dtype=np.float32)

        raw_obs = np.concatenate([self._static_features_normalized, dynamic_features])
        
        if self.vec_norm is not None:
            return self.vec_norm.normalize_obs(raw_obs)
        else:
            return raw_obs

    def decide_parameters(self, last_gen_stats: Dict[str, Any], gen_time: float) -> Dict[str, Any]:
        """预测下一代参数 (带脉冲式扰动机制)"""
        # 1. 获取观察值 (这会更新 self._steps_at_optimum)
        obs = self._get_observation(last_gen_stats, gen_time)
        
        # 2. 模型预测 (基础动作)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # =========================================================
        # [新增] 停滞扰动机制 (Stagnation Perturbation)
        # =========================================================
        STAGNATION_THRESHOLD = 5   # 阈值
        BASE_NOISE_SCALE = 0.5     # 强度
        
        if self._steps_at_optimum >= STAGNATION_THRESHOLD:
            # 计算动态强度
            scale = BASE_NOISE_SCALE + (self._steps_at_optimum // 10) * 0.1
            scale = min(scale, 0.8)
            
            # 生成噪声
            noise = np.random.normal(loc=0.0, scale=scale, size=action.shape)
            
            # print(f"   ⚡ [主动扰动] 检测到停滞 {self._steps_at_optimum} 代，注入噪声 (σ={scale:.2f})...")
            
            # 叠加并截断
            action = np.clip(action + noise, -1.0, 1.0)
            
            # -----------------------------------------------------
            # [关键修正] 注入噪声后，必须重置计数器！
            # 这样才能保证是“脉冲式”扰动，而不是“持续性”干扰
            # -----------------------------------------------------
            self._steps_at_optimum = 0 
            
        # =========================================================
        
        # 3. 解码 (逻辑不变)
        pop_norm = (action[0] + 1) / 2
        min_pop, max_pop = self.config.POP_SIZE_BOUNDS
        pop_scaled = min_pop + pop_norm * (max_pop - min_pop)

        alpha_norm = (action[1] + 1) / 2
        min_alpha, max_alpha = self.config.ALPHA_BOUNDS

        len_norm = (action[2] + 1) / 2
        min_len, max_len = self.config.SA_LEN_BOUNDS
        len_scaled = min_len + len_norm * (max_len - min_len)

        return {
            "POPULATION_SIZE": int(round(pop_scaled)),
            "ALPHA": min_alpha + alpha_norm * (max_alpha - min_alpha),
            "SA_METROPOLIS_LEN": int(round(len_scaled)),
        }


# ==========================================
# [修正版] 消融实验控制器 (针对 DRL 动作)
# ==========================================
class MaskedAgentController(AgentController):
    def __init__(self, model_path, hyper_config, problem_data, mask_param=None, mask_value=None):
        super().__init__(model_path, hyper_config, problem_data)
        self.mask_param = mask_param
        self.mask_value = mask_value
        
        if self.mask_param:
            print(f"🛡️ [Ablation] 动作屏蔽已激活: 锁定 '{self.mask_param}' = {self.mask_value}")

    def decide_parameters(self, last_gen_stats: Dict[str, Any], gen_time: float) -> Dict[str, Any]:
        # 1. 让 DRL 正常预测所有参数 (保持另外两个参数的动态性)
        params = super().decide_parameters(last_gen_stats, gen_time)
        
        # 2. 强制覆盖被消融的那个参数
        if self.mask_param is not None and self.mask_value is not None:
            # 确保键名匹配
            target_key = self.mask_param
            
            # 针对整数类型的参数进行强制转换
            if target_key in ["POPULATION_SIZE", "SA_METROPOLIS_LEN"]:
                final_val = int(round(self.mask_value))
            else:
                final_val = float(self.mask_value)
            
            # 覆盖 DRL 的输出
            params[target_key] = final_val
            
            # [调试] 偶尔打印一下证明覆盖成功
            # if np.random.rand() < 0.01:
            #     print(f"   [Debug] Masked {target_key}: DRL->{params[target_key]} | Fixed->{final_val}")

        return params    
    
# ==============================================================================
# 2. 优化器封装 (包含时间限制 & 报告生成)
# ==============================================================================
class IntelligentOptimizer(ParallelGraspOptimizer):
    def solve_with_agent(self, agent_controller: AgentController, max_generations: int, time_limit: Optional[float] = None, seed: Optional[int] = None):
        self.logger.info("🚀 启动 AI 求解器...")
        if time_limit:
            self.logger.info(f"⏳ 设定最大运行时间: {time_limit} 秒")
        if getattr(self.config, 'STRICT_DISRUPTION', False):
            self.logger.info("⚠️ 警告：当前启用【严格中断模式】(整车失效)")
            
        start_time = time.time()
        
        last_gen_stats = {
            'new_overall_best_fitness': -1e5,
            'mean_fitness_in_gen': -1e5,
            'avg_acceptance_rate': 0.5,
            'avg_initial_fitness': -1e5,
            'avg_final_fitness': -1e5
        }
        
        last_gen_duration = 0.1 

        for generation in range(1, max_generations + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                self.logger.info(f"⏰ 已达到时间限制 ({time_limit}s)，停止优化。")
                break

            params = agent_controller.decide_parameters(last_gen_stats, last_gen_duration)
            
            self.config.POPULATION_SIZE = params["POPULATION_SIZE"]
            self.config.ALPHA = params["ALPHA"]
            self.config.SA_METROPOLIS_LEN = params["SA_METROPOLIS_LEN"]
            
            t0 = time.time()
            current_stats = self.run_one_generation(generation, max_generations)
            t1 = time.time()
            
            last_gen_duration = t1 - t0
            last_gen_stats = current_stats

        total_elapsed = time.time() - start_time
        self.logger.info(f"✅ 任务完成。总耗时: {total_elapsed:.2f}s")
        
        # [修改] 优化结束后，调用生成详细报告
        self.save_solution_to_file(self.config.FINAL_SOLUTION_PATH) # 保留原有的简单保存
        generate_detailed_report(self, self.config.PROBLEM_DATA_PATH, "solution_report.txt", seed=seed)
        # generate_lenient_report(self, self.config.PROBLEM_DATA_PATH, "solution_report_fixed.txt", seed=seed)

    def save_solution_to_file(self, filepath: str):
        if not self.best_overall_solution_assignments: return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Best Fitness: {self.best_overall_fitness:.4f}\n")
                f.write("param tasks: V P D G Q :=\n")
                assignments = self.best_overall_solution_assignments['assignments']
                for v_id in sorted(assignments.keys()):
                    try:
                        v_num = int(v_id)
                    except:
                        v_num = v_id
                    sorted_tasks = sorted(assignments[v_id], key=lambda t: t['d'])
                    for t in sorted_tasks:
                        f.write(f"{v_num} {t['p']} {t['d']} {t['g']} {t['q']:.2f}\n")
                f.write(";\n")
        except Exception:
            pass

# ==============================================================================
# 3. 主入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="DRL Ablation Study Runner")
    
    # 1. 基础参数 (Standard Args)
    parser.add_argument("--problem", type=str, required=True, help="算例文件路径 (.dat)")
    parser.add_argument("--model", type=str, default="./logs/best_model.zip", help="DRL模型路径")
    parser.add_argument("--gens", type=int, default=100, help="最大迭代代数")
    parser.add_argument("--workers", type=int, default=None, help="CPU核数 (None表示自动)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--timeout", type=float, default=None, help="最大运行时间(秒)")
    parser.add_argument("--strict", action="store_true", help="开启严格中断模式")

    # 2. 消融实验专用参数 (Ablation Args)
    # 接收参数名 (如 ALPHA) 和 固定值 (如 0.5)
    parser.add_argument("--ablation_param", type=str, default=None, help="需要固定的参数名称")
    parser.add_argument("--ablation_value", type=float, default=None, help="该参数的固定数值")

    args = parser.parse_args()

    # --- 环境初始化 ---
    if args.seed is not None:
        print(f"🎲 [Ablation] 设置全局随机种子: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        # 如果您原本有 set_random_seed 函数，请保留调用
        try:
            set_random_seed(args.seed)
        except NameError:
            pass

    if not os.path.exists(args.problem):
        print(f"❌ 错误: 算例文件不存在 {args.problem}")
        sys.exit(1)

    # --- 配置加载 ---
    hyper_config = HyperConfig()
    grasp_config = GraspConfig(PROBLEM_DATA_PATH=args.problem, NUM_WORKERS=args.workers)
    
    grasp_config.USE_KSP_CACHE = True
    if args.strict:
        grasp_config.STRICT_DISRUPTION = True

    # --- 日志与数据 ---
    logger = setup_logger(grasp_config)
    logger.info(f"正在加载算例: {os.path.basename(args.problem)}")
    
    pd = ProblemData(args.problem, '', matrix_filepath="")
    
    # --- 控制器初始化 (关键分支) ---
    if args.ablation_param is not None and args.ablation_value is not None:
        logger.info(f"🧪 [消融模式启动] 正在锁定动作: {args.ablation_param} -> {args.ablation_value}")
        
        # 使用带屏蔽功能的控制器
        controller = MaskedAgentController(
            model_path=args.model,
            hyper_config=hyper_config,
            problem_data=pd,
            mask_param=args.ablation_param,
            mask_value=args.ablation_value
        )
    else:
        logger.info("🤖 [标准模式] DRL 动态控制所有参数")
        # 回退到普通控制器 (防止脚本被误调用时出错)
        controller = AgentController(args.model, hyper_config, pd)
    
    # --- 启动优化器 ---
    optimizer = IntelligentOptimizer(grasp_config, logger)
    
    try:
        optimizer.solve_with_agent(
            controller, 
            max_generations=args.gens, 
            time_limit=args.timeout,
            seed=args.seed
        )
    except KeyboardInterrupt:
        logger.warning("用户强制中断，正在保存当前状态...")
        optimizer.save_solution_to_file(grasp_config.FINAL_SOLUTION_PATH)
        # 如果需要生成报告，请确保 generate_detailed_report 已导入
        try:
            generate_detailed_report(optimizer, args.problem, "solution_report.txt", seed=args.seed)
        except:
            pass
    finally:
        optimizer.shutdown()

if __name__ == '__main__':
    main()# python3 run_with_agent.py problem_instances/mid_scale/Case_Clustered_Seed71808.dat  --model logs/best_model.zip  --gens 200 --seed 42 --time_limit 1200