#物资填缝

import time
import random
import copy
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any

# 复用现有的核心数据结构
from src.algorithms.solver_coreDRL import ProblemData, PathLibrary, Solution, Task, SimulatedAnnealing

class BaselineVNS:
    """
    Baseline Algorithm: VNS + Giant Tour + Capacity-Based Split
    [Ultimate Fixed Version]: 强制去重 + 绝对容量红线
    """
    
    def __init__(self, problem_data: ProblemData, config, path_library: PathLibrary):
        self.pd = problem_data
        self.cfg = config
        self.pl = path_library
        
        # 借用 SA 的评估器
        self.evaluator = SimulatedAnnealing(problem_data, config, path_library)
        
        # 预处理：生成原子任务列表 (强制去重版)
        self.atomic_tasks = self._generate_atomic_tasks_defensive()
        
        # 预处理：供应点索引
        self.supply_nodes = self._index_supply_nodes()

    def _generate_atomic_tasks_defensive(self) -> List[Dict]:
        """
        [终极修正版] 将所有需求拆解为原子任务
        Feature: 引入 seen 集合，物理层面上禁止重复生成任务
        """
        tasks = []
        seen_demands = set() # 记录已生成的 (node, good)
        duplicate_count = 0
        
        for n, demands in self.pd.demands.items():
            for g, q in demands.items():
                # 严格判定需求 (q < 0)
                if q < -1e-6:
                    # 强制清洗类型
                    d_node_int = int(n)
                    good_int = int(g)
                    qty = abs(q)
                    
                    # 1. 强制去重检查
                    if (d_node_int, good_int) in seen_demands:
                        duplicate_count += 1
                        continue # 跳过重复项
                    
                    seen_demands.add((d_node_int, good_int))
                    
                    tasks.append({
                        'd_node': d_node_int, 
                        'good': good_int, 
                        'qty': qty
                    })
        
        if duplicate_count > 0:
            print(f"⚠️ [DEFENSE] 成功拦截并剔除了 {duplicate_count} 个重复需求任务!")
        
        return tasks

    def _index_supply_nodes(self):
        """记录每种货物的供应点及其初始库存"""
        supply_map = defaultdict(list)
        for n, supplies in self.pd.demands.items():
            for g, q in supplies.items():
                if q > 1e-6: # 供应点
                    supply_map[int(g)].append({'node': int(n), 'qty': q})
        return supply_map

    def _find_feasible_supply(self, d_node, good, qty, current_supply_state):
        """在当前库存状态下，为需求点找到最近的、有足够库存的供应点。"""
        candidates = self.supply_nodes.get(good, [])
        best_s_node = None
        min_dist = 1e9
        
        for cand in candidates:
            s_node = cand['node']
            rem_qty = current_supply_state.get((s_node, good), cand['qty'])
            
            if rem_qty >= qty: # 库存充足
                # 使用 PathLibrary 查询距离 (Cache K=1)
                paths = self.pl.get_k_shortest_paths(s_node, d_node, 1)
                if paths:
                    dist = paths[0]['dist']
                    if dist < min_dist:
                        min_dist = dist
                        best_s_node = s_node
        
        return best_s_node

    def decode_split_advanced(self, task_permutation: List[Dict]) -> Solution:
        """
        [进阶解码]: 带有“填缝”能力的解码器
        当首个任务装不下时，尝试向后搜索能填满剩余容量的小任务。
        """
        sol = Solution(self.pd, 1)
        for v in self.pd.vehicles:
            sol.node_sequences[v] = []
            sol.path_choices[v] = []
            
        # 初始化供应状态
        supply_state = {}
        for g, nodes in self.supply_nodes.items():
            for info in nodes:
                supply_state[(info['node'], g)] = info['qty']

        vehicle_ids = list(self.pd.vehicles.keys())
        v_idx = 0
        
        # 使用列表来维护待处理任务，方便移除
        pending_tasks = list(task_permutation) # 浅拷贝
        
        current_v_id = vehicle_ids[v_idx] if vehicle_ids else None
        batch_pickups = []
        current_load = defaultdict(float)
        
        while pending_tasks and v_idx < len(vehicle_ids):
            # 1. 尝试处理清单中的任务（不仅是第一个，而是遍历寻找能装下的）
            # 我们优先看第一个任务(main task)，如果装不下，就找后面的(filler task)
            
            task_loaded_flag = False
            removal_indices = []
            
            # 获取当前车辆载重限制
            current_v_cap = self.pd.capacities[current_v_id]
            
            # --- 填缝逻辑开始 ---
            # 遍历 pending_tasks，找到能装入当前车辆的任务
            # 注意：这里我们稍微贪婪一点，只要能装就装，不再严格死守 VNS 的顺序
            # 但为了尊重 VNS，我们按顺序扫描
            
            i = 0
            while i < len(pending_tasks):
                task = pending_tasks[i]
                
                # A. 找供应点 (逻辑不变)
                s_node = self._find_feasible_supply(task['d_node'], task['good'], task['qty'], supply_state)
                
                can_load = False
                if s_node is not None:
                    # B. 检查容量
                    test_load = current_load.copy()
                    test_load[task['good']] += task['qty']
                    
                    cap_ok = True
                    usage_ratio = 0.0
                    for g, q in test_load.items():
                        c = current_v_cap.get(int(g), 0)
                        if c > 1e-6:
                            usage_ratio += q / c
                        elif q > 1e-6:
                            cap_ok = False
                            break
                    
                    # 允许装载的条件
                    if cap_ok and usage_ratio <= 0.9999:
                        can_load = True
                
                if can_load:
                    # 装车!
                    t_obj = Task(current_v_id, int(s_node), int(task['d_node']), int(task['good']), task['qty'])
                    batch_pickups.append(t_obj)
                    
                    current_load[task['good']] += task['qty']
                    supply_state[(s_node, task['good'])] -= task['qty']
                    
                    # 标记要移除的任务
                    pending_tasks.pop(i) 
                    # 此时 i 不需要加 1，因为后面的元素前移了
                    task_loaded_flag = True
                    
                    # 如果车辆已经很满（比如 > 95%），可以提前停止扫描，节省时间
                    if usage_ratio > 0.95:
                        break
                else:
                    # 这个任务装不下，看下一个
                    i += 1
            
            # --- 填缝逻辑结束 ---
            
            # 如果这一轮扫描下来，一个任务都没装进去（说明当前车辆满了，或者任务太大连空车都装不下）
            if not task_loaded_flag:
                # 封车，保存当前车辆路径
                self._commit_vehicle_route(sol, current_v_id, batch_pickups)
                
                # 换下一辆车
                v_idx += 1
                if v_idx < len(vehicle_ids):
                    current_v_id = vehicle_ids[v_idx]
                    batch_pickups = []
                    current_load = defaultdict(float)
                else:
                    # 没车了，跳出
                    break
            else:
                # 如果装入了任务，我们继续用这辆车在下一轮循环里尝试装更多
                # 除非它真的装不下了（在上面的扫描中会被跳过）
                # 为了防止死循环（比如剩下一个巨型任务，所有车都装不下），
                # 我们需要检测：如果当前车辆虽有装载但无法再装入剩余任何任务 -> 换车
                # 简单处理：只要发生过装载，就让 while 循环继续，再次扫描剩余列表
                # 如果再次扫描发现全都不行，就会进入 if not task_loaded_flag 分支换车
                pass

        # 处理最后一辆车的尾部数据
        if batch_pickups and v_idx < len(vehicle_ids):
             self._commit_vehicle_route(sol, current_v_id, batch_pickups)
             
        # Shortest Path 补全
        for v, seq in sol.node_sequences.items():
            sol.path_choices[v] = [0] * len(seq)
            
        return sol

    def _commit_vehicle_route(self, sol, v_id, batch_pickups):
        """辅助函数：把 batch_pickups 提交到 solution"""
        if not batch_pickups: return
        
        # 简单的路径构建：P -> D
        # 更好的做法可能是：TSP 优化一下这些点，但 Baseline 简单拼接即可
        p_nodes = [t.pickup_node for t in batch_pickups]
        d_nodes = [t.delivery_node for t in batch_pickups]
        
        # 简单去重相邻点
        seq = []
        for n in p_nodes + d_nodes:
            if not seq or seq[-1] != n:
                seq.append(n)
        
        sol.node_sequences[v_id].extend(seq)
        
        if v_id not in self.pd.tasks_by_vehicle: self.pd.tasks_by_vehicle[v_id] = []
        self.pd.tasks_by_vehicle[v_id].extend(batch_pickups)
        
        if v_id not in self.pd.task_map: self.pd.task_map[v_id] = {}
        for t in batch_pickups:
             self.pd.task_map[v_id][t.delivery_node] = t

    def evaluate(self, sol: Solution):
        self.evaluator.pd.tasks_by_vehicle = self.pd.tasks_by_vehicle
        self.evaluator.pd.task_map = self.pd.task_map
        fit, _, _ = self.evaluator.evaluate(sol)
        return fit

    def run(self, max_iterations=1000, max_time=60, seed: int = None):
        """VNS 主循环 (带实时最优解日志)"""
        start_time = time.time()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 初始解
        curr_perm = list(self.atomic_tasks)
        random.shuffle(curr_perm) 
        
        self.pd.tasks_by_vehicle = {}
        self.pd.task_map = {}
        
        curr_sol = self.decode_split_advanced(curr_perm)
        curr_fit = self.evaluate(curr_sol)
        
        best_perm = list(curr_perm)
        best_fit = curr_fit
        best_sol_dict = curr_sol.to_full_dict()
        
        k = 1 
        iter_count = 0
        
        # 算子名称映射，用于日志显示
        op_names = {1: "Swap", 2: "Shift", 3: "Reversal"}
        
        print(f"[Baseline] Init Fit: {curr_fit:.2f} (Seed: {seed})")
        try:
            while iter_count < max_iterations and (time.time() - start_time) < max_time:
                iter_count += 1
                
                # 记录当前使用的算子 k，用于日志
                current_op_k = k 
                
                cand_perm = list(curr_perm)
                
                if k == 1:   
                    self._op_swap(cand_perm)
                elif k == 2: 
                    self._op_shift(cand_perm)
                else:        
                    self._op_reversal(cand_perm)
                
                self.pd.tasks_by_vehicle = {}
                self.pd.task_map = {}
                
                cand_sol = self.decode_split_advanced(cand_perm)
                cand_fit = self.evaluate(cand_sol)
                
                # VNS 接受准则
                if cand_fit > curr_fit:
                    curr_perm = cand_perm
                    curr_fit = cand_fit
                    k = 1 # 改进后回到最小邻域
                    
                    # [关键修改] 更新全局最优并立即打印
                    if curr_fit > best_fit:
                        best_fit = curr_fit
                        best_perm = list(curr_perm)
                        best_sol_dict = cand_sol.to_full_dict()
                        
                        # 计算耗时
                        elapsed = time.time() - start_time
                        op_name = op_names.get(current_op_k, "Unknown")
                        
                        # 打印高亮日志
                        print(f"[Baseline] 🚀 New Best! Iter: {iter_count:<6} | Time: {elapsed:6.2f}s | Fit: {best_fit:10.2f} | Op: {op_name}")
                else:
                    k += 1
                    if k > 3: k = 1 
        except KeyboardInterrupt:
            # 捕获 Ctrl+C 信号
            print(f"\n[Baseline] ⚠️ 检测到热中断 (Ctrl+C)！正在终止优化并保存当前最优解...")        
        return best_fit, best_sol_dict

    def _op_swap(self, perm):
        if len(perm) < 2: return
        i, j = random.sample(range(len(perm)), 2)
        perm[i], perm[j] = perm[j], perm[i]

    def _op_shift(self, perm):
        if len(perm) < 2: return
        i = random.randint(0, len(perm)-1)
        task = perm.pop(i)
        j = random.randint(0, len(perm)) 
        perm.insert(j, task)

    def _op_reversal(self, perm):
        if len(perm) < 3: return
        i, j = random.sample(range(len(perm)), 2)
        start, end = min(i, j), max(i, j)
        perm[start:end+1] = perm[start:end+1][::-1]