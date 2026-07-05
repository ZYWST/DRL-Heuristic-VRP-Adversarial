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
from concurrent.futures import ProcessPoolExecutor

WORKER_SOLVER = None

def init_worker_batch_eval(pd_path, cfg_opts):
    """Worker 初始化：加载只读数据"""
    global WORKER_SOLVER
    from src.algorithms.solver_coreDRL import ProblemData, PathLibrary
    from src.env.hyper_config import GraspConfig
    # 避免循环导入
    # 动态导入 BaselineVNS 类（只需要用来调用 decode 和 evaluate，不需要实例化整个流程）
    
    cfg = GraspConfig()
    for k, v in cfg_opts.items(): setattr(cfg, k, v)
    
    # 加载数据
    pd = ProblemData(pd_path, '', matrix_filepath="")
    pl = PathLibrary(pd, cfg)
    
    # 我们把 pd, cfg, pl 存起来
    WORKER_SOLVER = (pd, cfg, pl)

def worker_try_batch_moves(start_perm, operator_type, batch_size, seed):
    """
    Worker 任务：
    拿到当前解 start_perm，尝试 batch_size 次随机操作。
    返回这批尝试中找到的【最好的 Fitness 和 Permutation】。
    """
    try:
        pd, cfg, pl = WORKER_SOLVER
        from src.algorithms.solver_baseline_advance import BaselineVNS
        # 实例化一个轻量级对象用于调用方法
        solver = BaselineVNS(pd, cfg, pl)
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 基准：虽然 Worker 拿到了 start_perm，但不需要重算它的分数，
        # 我们只需要找比它更好的，或者这一批里最好的。
        
        local_best_fit = -1e9
        local_best_perm = None
        
        # 在 Worker 内部循环，减少 IPC 通信开销
        for _ in range(batch_size):
            # 1. 复制
            cand_perm = list(start_perm)
            
            # 2. 扰动 (根据主进程指定的算子类型)
            if operator_type == 1: solver._op_swap(cand_perm)
            elif operator_type == 2: solver._op_shift(cand_perm)
            else: solver._op_reversal(cand_perm)
            
            # 3. 解码 & 评估
            solver.pd.tasks_by_vehicle = {}
            solver.pd.task_map = {}
            
            # 这里是耗时大户
            sol = solver.decode_split_advanced(cand_perm)
            fit = solver.evaluate(sol)
            
            # 4. 记录 Worker 内部的最优解
            if fit > local_best_fit:
                local_best_fit = fit
                local_best_perm = cand_perm
                
        # 返回这一批里最好的结果
        return local_best_fit, local_best_perm
        
    except Exception as e:
        import traceback
        return -1e9, str(traceback.format_exc())

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

    def run_parallel_acceleration(self, max_time=60, seed=None, num_workers=4):
        """
        并行加速版 VNS：
        主进程控制迭代流程。
        每次迭代将“寻找邻域解”的任务分发给多个 Worker 并行执行。
        """
        start_time = time.time()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 1. 初始解
        curr_perm = list(self.atomic_tasks)
        random.shuffle(curr_perm)
        
        curr_sol = self.decode_split_advanced(curr_perm)
        curr_fit = self.evaluate(curr_sol)
        
        best_fit = curr_fit
        best_perm = list(curr_perm)
        best_sol_dict = curr_sol.to_full_dict()
        
        print(f"[Baseline-Accel] Init Fit: {curr_fit:.2f} | Workers: {num_workers}")
        
        # 准备 Worker 数据
        cfg_opts = {
            'PROBLEM_DATA_PATH': self.cfg.PROBLEM_DATA_PATH,
            'USE_KSP_CACHE': True,
            'SA_M_PATHS': 1
        }
        
        # 算子映射
        op_names = {1: "Swap", 2: "Shift", 3: "Reversal"}
        k = 1
        iter_count = 0
        
        # 每个 Worker 每次承担的计算量
        # 关键点：不能太小，太小会被通信拖死；不能太大，太大不够灵敏
        BATCH_SIZE_PER_WORKER = 500 
        
        with ProcessPoolExecutor(max_workers=num_workers, 
                                 initializer=init_worker_batch_eval,
                                 initargs=(self.cfg.PROBLEM_DATA_PATH, cfg_opts)) as executor:
            
            try:
                while (time.time() - start_time) < max_time:
                    iter_count += 1
                    iter_start = time.time()
                    
                    # 生成随机种子，防止 Worker 行为雷同
                    seeds = [random.randint(1, 10000000) for _ in range(num_workers)]
                    
                    # === 并行核心 ===
                    # 发送任务：大家一起基于 curr_perm，用算子 k，找更好的解
                    futures = [
                        executor.submit(worker_try_batch_moves, curr_perm, k, BATCH_SIZE_PER_WORKER, s)
                        for s in seeds
                    ]
                    
                    # 收集结果
                    results = [f.result() for f in futures]
                    
                    # 剔除无效结果
                    valid_results = [r for r in results if isinstance(r[0], (int, float)) and r[0] > -1e8]
                    
                    if not valid_results: continue
                    
                    # 找出本轮所有 Worker 找到的解里，最好的那个
                    round_best_fit, round_best_perm = max(valid_results, key=lambda x: x[0])
                    
                    # VNS 接受准则 (这里实现了 "Best Improvement" 策略)
                    # 我们不仅看 1 个邻居，我们看了 (Workers * Batch_Size) 个邻居，选最好的
                    if round_best_fit > curr_fit:
                        # 找到了更好的解 -> 移动
                        curr_fit = round_best_fit
                        curr_perm = round_best_perm
                        
                        # 重置算子
                        current_op_k = k
                        k = 1 
                        
                        # 更新全局最优
                        if curr_fit > best_fit:
                            best_fit = curr_fit
                            best_perm = list(curr_perm)
                            
                            # 主进程重构完整解对象
                            temp_sol = self.decode_split_advanced(best_perm)
                            best_sol_dict = temp_sol.to_full_dict()
                            
                            elapsed = time.time() - start_time
                            # 计算吞吐量：每秒评估了多少个解
                            total_checked = num_workers * BATCH_SIZE_PER_WORKER
                            speed = total_checked / (time.time() - iter_start)
                            
                            print(f"[Baseline] 🚀 New Best! Iter: {iter_count:<5} | Time: {elapsed:6.2f}s | Fit: {best_fit:10.2f} | Op: {op_names[current_op_k]} | Speed: {speed:.0f} evals/s")
                    else:
                        # 这么多人试了这么多下，都没找到更好的 -> 切换算子
                        k += 1
                        if k > 3: k = 1
                        
            except KeyboardInterrupt:
                print("\n[Baseline] ⚠️ Interrupted.")
        
        return best_fit, best_sol_dict

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