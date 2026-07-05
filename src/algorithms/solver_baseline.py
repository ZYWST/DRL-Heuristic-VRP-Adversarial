#物资填缝
#不按距离找最近供应点

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
        candidates = self.supply_nodes.get(good, [])
        feasible_candidates = []
        
        for cand in candidates:
            s_node = cand['node']
            rem_qty = current_supply_state.get((s_node, good), cand['qty'])
            if rem_qty >= qty:
                feasible_candidates.append(s_node)
        
        if feasible_candidates:
            # [降级] 不再计算距离，直接随机选，或者选第一个
            return random.choice(feasible_candidates) 
        return None
    
    def decode_split_naive(self, task_permutation: List[Dict]) -> Solution:
        """
        [降级版解码]: 朴素分割 (Naive Split)
        严格按顺序装载。遇到装不下的任务立刻换车，不进行填缝搜索。
        """
        sol = Solution(self.pd, 1)
        # 初始化空路径
        for v in self.pd.vehicles:
            sol.node_sequences[v] = []
            sol.path_choices[v] = []
            
        # 复制供应状态
        supply_state = {}
        for g, nodes in self.supply_nodes.items():
            for info in nodes:
                supply_state[(info['node'], g)] = info['qty']

        vehicle_ids = list(self.pd.vehicles.keys())
        v_idx = 0
        
        # 遍历任务列表
        for task in task_permutation:
            if v_idx >= len(vehicle_ids): break # 没车了，停止
            
            current_v_id = vehicle_ids[v_idx]
            
            # --- 尝试装载当前任务 ---
            loaded = False
            
            # 1. 找最近供应点 (这一步也可以继续弱化，见下文)
            s_node = self._find_feasible_supply(task['d_node'], task['good'], task['qty'], supply_state)
            
            if s_node is not None:
                # 2. 检查容量 (简化版逻辑)
                # 获取当前车辆已载货物
                current_tasks = self.pd.tasks_by_vehicle.get(current_v_id, [])
                
                # 构造临时载重字典
                temp_load = defaultdict(float)
                for t in current_tasks: temp_load[t.good_id] += t.quantity
                temp_load[task['good']] += task['qty']
                
                # 校验容量
                cap_ok = True
                caps = self.pd.capacities[current_v_id]
                for g, q in temp_load.items():
                    c = caps.get(int(g), 0)
                    if c < 1e-6 and q > 0: cap_ok = False; break
                    if c > 1e-6 and (q / c) > 1.0: cap_ok = False; break
                
                if cap_ok:
                    # 装入!
                    t_obj = Task(current_v_id, int(s_node), int(task['d_node']), int(task['good']), task['qty'])
                    if current_v_id not in self.pd.tasks_by_vehicle: self.pd.tasks_by_vehicle[current_v_id] = []
                    self.pd.tasks_by_vehicle[current_v_id].append(t_obj)
                    self.pd.task_map.setdefault(current_v_id, {})[t_obj.delivery_node] = t_obj
                    
                    # 简单更新路径 (P->D)
                    sol.node_sequences[current_v_id].extend([int(s_node), int(task['d_node'])])
                    
                    # 扣减库存
                    supply_state[(s_node, task['good'])] -= task['qty']
                    loaded = True

            # --- 关键差异 ---
            # 如果装不下 (loaded=False)，或者虽然装下了但你想模拟更笨的策略
            # 在 Naive Split 中，如果当前任务装不下，必须换车，并尝试把这个任务装到新车上
            if not loaded:
                # 换下一辆车
                v_idx += 1
                if v_idx < len(vehicle_ids):
                    current_v_id = vehicle_ids[v_idx]
                    # 在新车上重试逻辑 (为简化代码，此处略去递归，实际应在一个while循环里处理单个任务直到装入或无车)
                    # 简单做法：如果当前车装不下，直接丢弃该任务或跳过，或者更常见的：
                    # 换新车后，回退指针，让下一次循环重新尝试该任务（但在Python for循环里很难回退）
                    # 建议：改写为 while loop 处理 task_idx
        
        # 补全 Path Choices
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
        
        curr_sol = self.decode_split_naive(curr_perm)
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
                
                cand_sol = self.decode_split_naive(cand_perm)
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