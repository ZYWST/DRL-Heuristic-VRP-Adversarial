# solver_core.py (V5: KSP预计算加速 + RCL概率选择 + ACO增强)+双重score，强制安排第一阶段未分配任务的车辆
# 版本特性：
# 1. PathLibrary: 不再计算路径，直接读取 'ksp_cache.pkl'，查询时间 O(1)。
# 2. 构造策略: 回归 RCL 截断(保证局部性)，配合轮盘赌(Score权重)进行概率选择。
# 3. 优化: 保留了 V3 的信息素剪枝、停滞重启机制。


# solver_core.py (已使用 Numba 加速并重构I/O瓶颈)  优化最优解的打印
#考虑空间局部性的版本
#加入路径求解的空间差异化和软拥堵惩罚
#实施“车辆-区域亲和度学习”机制
#加入蚁群相关思想————去除轮盘赌和超参数的“弱蚁群”
#加入了SA算子
#优化内存
#分配的score按目标函数来算
#强化探索：偏置随机排名(已废除）、信息素重置；同时信息素剪枝优化数据传输
#将小规模版本算法的设计为参数合并到这里面来，保留了增量设计
#修正了流量判定规则
#ksp自动化

#【new】修复了随机种子
import os

# --- 强制单线程设置 (防止底层库死锁) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import re
import copy 
import time
import random
import logging
import math
import sys
import gc
import pickle  # [V4 新增] 用于读取 KSP 缓存
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from src.env.hyper_config import GraspConfig

from numba import jit
from numba import types as nb_types 
from numba.typed import Dict as NumbaTypedDict 

from src.env import ksp_manager  # [新增]
# ---------------------------------------------------------------------------
# 日志开关：是否在每个 step 打印求解信息。
# 你可以通过注释/取消注释下面这行来开启或关闭每-step 日志。
# 例如：取消注释 `ENABLE_PER_STEP_LOGS = True` 将开启；注释掉则关闭。
ENABLE_PER_STEP_LOGS = True
# 默认关闭（如果未定义，则视为 False）
# ---------------------------------------------------------------------------
# ==============================================================================
# Numba JIT 加速函数 (保持不变)
# ==============================================================================

@jit(nopython=True, cache=True)
def _jit_check_capacity(good_ids, quantities, vehicle_capacities) -> bool:
    total_load_by_good = NumbaTypedDict.empty(key_type=nb_types.int64, value_type=nb_types.float64)
    for i in range(len(good_ids)):
        g_id = good_ids[i]
        q = quantities[i]
        total_load_by_good[g_id] = total_load_by_good.get(g_id, 0.0) + q
    total_capacity_usage_ratio = 0.0
    for good_id, total_load in total_load_by_good.items():
        capacity = vehicle_capacities.get(good_id)
        if capacity is None or capacity <= 1e-6:
            if total_load > 1e-6: return False 
        else:
            total_capacity_usage_ratio += total_load / capacity
    return total_capacity_usage_ratio <= 1.0 + 1e-6

@jit(nopython=True, cache=True)
def _jit_calculate_duplicates(full_path_np: np.ndarray, task_nodes_set: Set[int]) -> int:
    total_duplicates = 0
    counts = NumbaTypedDict.empty(key_type=nb_types.int64, value_type=nb_types.int64)
    for node in full_path_np:
        if node in task_nodes_set:
            counts[node] = counts.get(node, 0) + 1
    for count in counts.values():
        if count > 1: total_duplicates += (count - 1)
    return total_duplicates

@jit(nopython=True, cache=True)
def _jit_check_disruption(delivery_path_np: np.ndarray, disrupted_edges_set: Set[Tuple[int, int]]) -> bool:
    for i in range(len(delivery_path_np) - 1):
        edge = (int(delivery_path_np[i]), int(delivery_path_np[i+1]))
        if edge in disrupted_edges_set: return True
    return False

# ==============================================================================
# 核心数据结构
# ==============================================================================

class Task:
    def __init__(self, vehicle_id: int, pickup_node: int, delivery_node: int, good_id: int, quantity: float):
        self.vehicle_id = vehicle_id
        self.pickup_node = pickup_node
        self.delivery_node = delivery_node
        self.good_id = good_id
        self.quantity = quantity
    def __repr__(self):
        return f"Task(V{self.vehicle_id}: P{self.pickup_node}->D{self.delivery_node}, Q{self.quantity:.1f})"

class ProblemData:
    def __init__(self, data_filepath: str, tasks_filepath: str, matrix_filepath: str = "100nodes_distance.txt"):
        self.H, self.U = 0.0, 0
        self.nodes, self.edges, self.distances = {}, [], {}
        self.goods, self.vehicles = set(), {}
        self.capacities, self.demands, self.weights = {}, {}, {}
        self.tasks_by_vehicle, self.task_map = {}, {}
        self.shortcut_distances = {} 

        if data_filepath and os.path.exists(data_filepath):
            self._parse_data_file(data_filepath)
        
        # 仍然加载矩阵文件，用于 GRASP 阶段的快速启发式估算 (Heuristic Estimate)
        if matrix_filepath and os.path.exists(matrix_filepath):
            self._parse_matrix_file(matrix_filepath)

    def _parse_data_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
        def parse_block(pattern, text):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if not match: return []
            return [line.strip().split() for line in match.group(1).strip().split('\n') 
                    if line.strip() and not line.strip().startswith('#')]

        for row in parse_block(r'param\s*:N:px\s+py\s*:=\s*(.*?);', content):
            self.nodes[int(row[0])] = {'px': float(row[1]), 'py': float(row[2])}
            self.distances[int(row[0])] = {}
        for row in parse_block(r'param\s*:E:d\s*=\s*(.*?);', content):
            u, v, d = int(row[0]), int(row[1]), float(row[2])
            self.edges.append((u, v))
            self.distances[u][v] = d
        for row in parse_block(r'param\s*:K:v\s+L\s*:=\s*(.*?);', content):
            k_id = int(row[0])
            self.vehicles[k_id] = {'v': float(row[1]), 'L': int(row[2])}
            self.capacities[k_id], self.tasks_by_vehicle[k_id] = {}, []

        goods_match = re.search(r'set\s+G\s*:=\s*(.*?);', content, re.IGNORECASE)
        self.goods = {int(g) for g in re.findall(r'\d+', goods_match.group(1))} if goods_match else set()

        cap_data = parse_block(r'param\s+c\s*:(.*?);', content)
        if cap_data and self.goods:
            g_ids = [int(g) for g in re.findall(r'\d+', " ".join(cap_data[0])) if int(g) in self.goods]
            for row in cap_data[1:]:
                try:
                    for i, cap in enumerate(row[1:]):
                        if i < len(g_ids): self.capacities[int(row[0])][g_ids[i]] = float(cap)
                except: continue

        dem_data = parse_block(r'param\s+a\s*:(.*?);', content)
        if dem_data and self.goods:
            for n in self.nodes: self.demands[n] = {g: 0.0 for g in self.goods}
            g_ids = [int(g) for g in re.findall(r'\d+', " ".join(dem_data[0])) if int(g) in self.goods]
            for row in dem_data[1:]:
                try:
                    for i, val in enumerate(row[1:]):
                        if i < len(g_ids): self.demands[int(row[0])][g_ids[i]] = float(val)
                except: continue

        for n in self.nodes: self.weights[n] = {g: 1.0 for g in self.goods}
        w_data = parse_block(r'param\s+w\s*:(.*?);', content)
        if w_data and self.goods:
            g_ids = [int(g) for g in re.findall(r'\d+', " ".join(w_data[0])) if int(g) in self.goods]
            for row in w_data[1:]:
                try:
                    for i, val in enumerate(row[1:]):
                        if val != '.' and i < len(g_ids): self.weights[int(row[0])][g_ids[i]] = float(val)
                except: continue

        hm = re.search(r'param\s+H\s*:=\s*([\d.]+)', content)
        if hm: self.H = float(hm.group(1))
        um = re.search(r'param\s+U\s*:=\s*(\d+)', content)
        if um: self.U = int(um.group(1))

    def _parse_matrix_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                header = f.readline().strip().split()
                col_ids = [int(x) for x in header]
                for line in f:
                    parts = line.strip().split()
                    if not parts: continue
                    row_id = int(parts[0])
                    dists = parts[1:]
                    self.shortcut_distances[row_id] = {}
                    for i, d_str in enumerate(dists):
                        if i < len(col_ids):
                            self.shortcut_distances[row_id][col_ids[i]] = float(d_str)
        except Exception as e:
            print(f"[ERROR] Loading matrix failed: {e}")

    def get_matrix_distance(self, u: int, v: int) -> float:
        if u in self.shortcut_distances and v in self.shortcut_distances[u]:
            return self.shortcut_distances[u][v] / 1000.0 # 假设单位是米，转公里
        if u in self.distances and v in self.distances[u]:
            return self.distances[u][v]
        return 10000.0

# ==============================================================================
# [V4 核心] PathLibrary (基于预计算缓存)
# ==============================================================================

class PathLibrary:
    def __init__(self, problem_data: ProblemData, config: GraspConfig):
        self.config = config
        self.precomputed = {}
        self.graph = None
        
        # === 核心修改 START ===
        
        # 1. 尝试通过 Manager 自动获取 (读取 or 计算)
        if config.USE_KSP_CACHE:
            # 这里的 PROBLEM_DATA_PATH 需要确保在 config 中存在
            # 如果 ProblemData 初始化时没存 path，就从 config 拿
            data_path = getattr(config, 'PROBLEM_DATA_PATH', 'unknown_data.dat')
            
            # 这一行替代了原本所有的 "try open pickle... except..." 代码
            self.precomputed = ksp_manager.get_or_compute_ksp(
                problem_data, 
                config, 
                data_path
            )

        # 2. 关键判断：如果成功获取到了缓存（非空），直接结束！
        # 这样就完全跳过了后面耗时耗内存的 NetworkX 初始化
        if self.precomputed:
            return 
            
        # === 核心修改 END ===
        
        # 3. 兜底逻辑 (原来的 if not cache_loaded 块)
        # 只有在 config.USE_KSP_CACHE = False 或者 缓存获取失败时才会执行这里
        print("[Warning] PathLibrary: Running in slow fallback mode (NetworkX).")
        import networkx as nx
        self.graph = nx.Graph()
        self.path_cache = {} 
        for u, neighbors in problem_data.distances.items():
            for v, dist in neighbors.items():
                self.graph.add_edge(u, v, weight=dist)

    def get_k_shortest_paths(self, u: int, v: int, k: int):
        # 1. 查表路径 (O(1))
        if self.precomputed:
            if u in self.precomputed and v in self.precomputed[u]:
                return self.precomputed[u][v]['paths'][:k]
            return []
            
        # 2. 实时计算路径 (NetworkX)
        if self.graph:
            cache_key = (u, v)
            if cache_key in self.path_cache:
                return self.path_cache[cache_key][:k]
            
            try:
                import networkx as nx
                paths = []
                # 使用 shortest_simple_paths 的迭代器
                gen = nx.shortest_simple_paths(self.graph, u, v, weight='weight')
                for _ in range(k):
                    path = next(gen)
                    dist = nx.path_weight(self.graph, path, weight='weight')
                    paths.append({'path': path, 'dist': dist})
                self.path_cache[cache_key] = paths
                return paths
            except (nx.NetworkXNoPath, StopIteration):
                return []
        return []
    
# ==============================================================================
# 解与算法 (Solution, SA)
# ==============================================================================

class Solution:
    def __init__(self, problem_data: ProblemData, m_paths: int):
        self.problem_data = problem_data
        self.node_sequences = {}
        self.path_choices = {}
        self.fitness = -1e5
        
        # === [New] 必须初始化的缓存 ===
        self.v_stats = {} 
        self.global_edge_counts = defaultdict(int)
        self.cached_disrupted_edges = set() 
        self.total_util = 0.0   # [New] O(1) 维护的总效用
        self.total_dups = 0     # [New] O(1) 维护的总重复数
        
        self._initialize(m_paths)

    def _initialize(self, m_paths: int):
        for v_id, tasks in self.problem_data.tasks_by_vehicle.items():
            if not tasks:
                self.node_sequences[v_id], self.path_choices[v_id] = [], []
                continue
            
            # 拓扑排序构建初始解
            pickups = [t.pickup_node for t in tasks]
            deliveries = [t.delivery_node for t in tasks]
            nodes = list(set(pickups + deliveries))
            in_degree = {n: 0 for n in nodes}
            graph = defaultdict(list)
            for t in tasks:
                if t.pickup_node != t.delivery_node:
                    if t.pickup_node in in_degree and t.delivery_node in in_degree:
                        graph[t.pickup_node].append(t.delivery_node)
                        in_degree[t.delivery_node] += 1
            
            queue = [n for n in nodes if in_degree[n] == 0]
            seq = []
            while queue:
                random.shuffle(queue)
                u = queue.pop(0)
                seq.append(u)
                for v in graph[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0: queue.append(v)
            
            if len(seq) < len(nodes): 
                rem = [n for n in nodes if n not in seq]
                random.shuffle(rem)
                seq.extend(rem)
            
            self.node_sequences[v_id] = seq
            self.path_choices[v_id] = [random.randint(0, m_paths-1) for _ in range(len(seq))]

    def mutate(self, target_v_id=None) -> int:
        # 1. 选车
        active = [v for v, seq in self.node_sequences.items() if len(seq) >= 2]
        if not active: return -1
        
        # 如果指定了就用指定的，没指定就随机
        v_id = target_v_id if target_v_id is not None else random.choice(active)
        
        # 2. 执行变异
        r = random.random()
        if r < 0.3:
            self._mutate_path(v_id)
        elif r < 0.7:
            rr = random.random()
            if rr < 0.4: self._mutate_single(v_id)
            elif rr < 0.7: self._mutate_pair(v_id)
            else: self._mutate_node_swap(v_id)
            
        return v_id # 返回被修改的车辆ID

    def copy(self):
        n = Solution.__new__(Solution)
        n.problem_data = self.problem_data
        n.node_sequences = copy.deepcopy(self.node_sequences)
        n.path_choices = copy.deepcopy(self.path_choices)
        n.fitness = self.fitness
        
        # [New] 必须拷贝缓存
        n.v_stats = copy.deepcopy(getattr(self, 'v_stats', {}))
        n.global_edge_counts = getattr(self, 'global_edge_counts', defaultdict(int)).copy()
        n.cached_disrupted_edges = getattr(self, 'cached_disrupted_edges', set()).copy()
        n.total_util = getattr(self, 'total_util', 0.0)
        n.total_dups = getattr(self, 'total_dups', 0)
        return n

    # --- 统一且鲁棒的序列合法性检查 (O(N) + O(T)) ---
    def _is_sequence_valid(self, seq: List[int], tasks: List[Task]) -> bool:
        # 1. 构建索引映射 O(N)
        idx_map = {node: i for i, node in enumerate(seq)}
        
        # 2. 遍历该车辆的所有任务进行检查 O(T)
        # 解决了 "字典覆盖" 导致的约束丢失问题
        for t in tasks:
            p, d = t.pickup_node, t.delivery_node
            
            # 确保 P 和 D 都在序列中 (理论上必须在)
            if p in idx_map and d in idx_map:
                # 核心约束: P 必须在 D 之前
                if idx_map[p] > idx_map[d]:
                    return False
        return True

    # --- 变异算子 1: 单点移动 (Shift) ---
    def _mutate_single(self, v_id):
        seq = self.node_sequences[v_id]
        if len(seq) < 2: return
        
        # 备份
        old_seq = seq[:]
        
        # 执行移动
        idx_from = random.randint(0, len(seq) - 1)
        node = seq.pop(idx_from)
        idx_to = random.randint(0, len(seq)) # 注意 len 已经少1了
        seq.insert(idx_to, node)
        
        # [Fix] 全局校验：如果不合法，立即回滚
        if not self._is_sequence_valid(seq, self.problem_data.tasks_by_vehicle[v_id]):
            self.node_sequences[v_id] = old_seq

    # --- 变异算子 2: 双点重插 (Pair Re-insert) ---
    def _mutate_pair(self, v_id):
        seq = self.node_sequences[v_id]
        tasks = self.problem_data.tasks_by_vehicle[v_id]
        if not tasks or len(seq) < 2: return

        # 备份
        old_seq = seq[:]
        
        # 随机选一个任务，将其 P 和 D 重新插入
        # 注意：这只保证了被选任务的 P<D，但可能破坏其他共享节点的任务顺序
        t = random.choice(tasks)
        try:
            # 移除 P 和 D
            # 注意处理 P=D 的情况 (虽少见) 或者 P/D 多次出现的情况
            # 这里简单处理：移除列表中的第一个匹配项
            if t.pickup_node in seq: seq.remove(t.pickup_node)
            if t.delivery_node in seq: seq.remove(t.delivery_node)
            
            # 重新插入
            # 保证 i < j
            idx1 = random.randint(0, len(seq))
            idx2 = random.randint(0, len(seq))
            i, j = sorted([idx1, idx2])
            
            seq.insert(j, t.delivery_node)
            seq.insert(i, t.pickup_node)
            
            # [Fix] 全局校验：必须检查对 *其他* 任务的影响
            if not self._is_sequence_valid(seq, tasks):
                self.node_sequences[v_id] = old_seq
                
        except ValueError:
            # 如果 remove 失败
            self.node_sequences[v_id] = old_seq

    # --- 变异算子 3: 节点交换 (Swap) ---
    # 改名自 _mutate_node_sequence 以避免混淆
    def _mutate_node_swap(self, v_id):
        seq = self.node_sequences[v_id]
        if len(seq) < 2: return
        
        idx1, idx2 = random.sample(range(len(seq)), 2)
        
        # 交换
        seq[idx1], seq[idx2] = seq[idx2], seq[idx1]
        
        # [Fix] 全局校验
        # 传入 tasks 列表而不是构建有缺陷的 dict
        if not self._is_sequence_valid(seq, self.problem_data.tasks_by_vehicle[v_id]):
            # 回滚
            seq[idx1], seq[idx2] = seq[idx2], seq[idx1]

    # --- 路径选择变异 (保持原样) ---
    def _mutate_path(self, v_id):
        choices = self.path_choices[v_id]
        if choices:
            idx = random.randint(0, len(choices)-1)
            choices[idx] = random.randint(0, 5)
    
    def to_full_dict(self):
        ad = {}
        for v, ts in self.problem_data.tasks_by_vehicle.items():
            ad[v] = [{'v':t.vehicle_id,'p':t.pickup_node,'d':t.delivery_node,'g':t.good_id,'q':t.quantity} for t in ts]
        return {'assignments': ad, 'node_sequences': self.node_sequences, 'path_choices': self.path_choices, 'fitness': self.fitness}


# ==============================================================================
# 核心修改：SimulatedAnnealing 类
# ==============================================================================
class SimulatedAnnealing:
    def __init__(self, pd: ProblemData, cfg: GraspConfig, pl: PathLibrary):
        self.pd, self.cfg, self.pl = pd, cfg, pl
        self.curr = Solution(pd, cfg.SA_M_PATHS)
        
        # 初始化：算一次全量，建立基准
        self.evaluate_full(self.curr) 
        self.best = self.curr.copy()

    # --- [保留] 单车路径计算逻辑 (逻辑不变) ---
    def _calc_vehicle_route(self, v_id, seq, choices, disrupted_set):
        if not seq: 
            return {'valid': True, 'util': 0.0, 'path': [], 'edges': [], 'dups': 0}

        curr_node = self.pd.vehicles[v_id]['L']
        path_nodes = [curr_node]
        curr_time = 0.0
        util_score = 0.0
        
        processed_pickups = set()
        processed_deliveries = set()
        current_load = {g: 0.0 for g in self.pd.goods}
        tasks = self.pd.tasks_by_vehicle.get(v_id, [])
        task_map = self.pd.task_map.get(v_id, {})
        full_seq = seq + [curr_node] if seq[-1] != curr_node else seq
        
        # 起点装货
        for t in tasks:
            if t.pickup_node == curr_node:
                current_load[t.good_id] += t.quantity
                processed_pickups.add(t)

        edges_in_route = [] 
        
        for i, next_node in enumerate(full_seq):
            if i >= len(seq) and next_node == curr_node: choice = 0
            else: choice = choices[i] if i < len(choices) else 0
            
            paths = self.pl.get_k_shortest_paths(curr_node, next_node, self.cfg.SA_M_PATHS)
            if not paths: return {'valid': False} 
            
            leg = paths[choice % len(paths)]
            
            # 容量检查
            active_goods = [(g, q) for g, q in current_load.items() if q > 1e-6]
            if active_goods:
                usage = sum(q / self.pd.capacities[v_id].get(g, 1e-6) 
                            for g, q in active_goods if self.pd.capacities[v_id].get(g, 0) > 0)
                if usage > 1.001: return {'valid': False}

            # 记录边
            for j in range(len(leg['path'])-1):
                edges_in_route.append((leg['path'][j], leg['path'][j+1]))

            curr_time += leg['dist'] / self.pd.vehicles[v_id]['v'] * 60
            path_nodes.extend(leg['path'][1:])
            curr_node = next_node
            
            # 装卸货与计分
            for t in tasks:
                if t.pickup_node == curr_node and t not in processed_pickups:
                    current_load[t.good_id] += t.quantity
                    processed_pickups.add(t)
            
            if curr_node in task_map:
                t = task_map[curr_node]
                if t.delivery_node == curr_node and t not in processed_deliveries:
                    current_load[t.good_id] -= t.quantity
                    processed_deliveries.add(t)
                    if curr_time <= self.pd.H:
                        w = self.pd.weights[curr_node].get(t.good_id, 1.0)
                        # 检查中断
                        try:
                            p_idx = path_nodes.index(t.pickup_node) 
                            path_segment = np.array(path_nodes[p_idx:], dtype=np.int64)
                            if not _jit_check_disruption(path_segment, disrupted_set):
                                util_score += w * t.quantity * (self.pd.H - curr_time)
                        except ValueError: pass

        t_nodes = {t.pickup_node for t in tasks} | {t.delivery_node for t in tasks}
        dups = 0 if not t_nodes else _jit_calculate_duplicates(np.array(path_nodes, dtype=np.int64), t_nodes)

        return {'valid': True, 'util': util_score, 'path': path_nodes, 'edges': edges_in_route, 'dups': dups}

    # --- [新] 初始化评估 (只跑一次) ---
    def evaluate_full(self, sol: Solution):
        sol.v_stats = {}
        sol.global_edge_counts = defaultdict(int)
        sol.total_util = 0.0
        sol.total_dups = 0
        edge_flows = {} 
        
        # 1. 基础计算
        temp_stats = {}
        for v_id in sol.node_sequences:
            res = self._calc_vehicle_route(v_id, sol.node_sequences[v_id], sol.path_choices[v_id], set())
            if not res['valid']: 
                sol.fitness = -1e5
                return -1e5
            temp_stats[v_id] = res
            for e in res['edges']:
                sol.global_edge_counts[e] += 1
                edge_flows[e] = edge_flows.get(e, 0) + 1.0

        # 2. 锁定中断边
        sol.cached_disrupted_edges = set()
        if edge_flows:
            sorted_e = sorted(edge_flows.items(), key=lambda x: x[1], reverse=True)
            for i in range(min(self.pd.U, len(sorted_e))):
                sol.cached_disrupted_edges.add(sorted_e[i][0])
        
        # 3. 正式计算总分
        for v_id, res in temp_stats.items():
            final_res = self._calc_vehicle_route(v_id, sol.node_sequences[v_id], sol.path_choices[v_id], sol.cached_disrupted_edges)
            sol.v_stats[v_id] = final_res
            sol.total_util += final_res['util']
            sol.total_dups += final_res['dups']
            
        self._update_fitness_from_totals(sol)
        return sol.fitness

    # --- [新] 从缓存的总值直接算 fitness (O(1)) ---
    def _update_fitness_from_totals(self, sol: Solution):
        pen, soft_pen = 0.0, 0.0
        cp = self.cfg.c_params
        
        # 1. 硬约束惩罚 (保持不变)
        if sol.total_dups > self.cfg.SA_MAX_ALLOWED_DUPLICATES:
            pen = self.cfg.SA_EXCESS_DUPLICATE_PENALTY * cp.congestion_aversion
        elif sol.total_dups > 0:
            pen = sol.total_dups * self.cfg.SA_PENALTY_FACTOR * cp.congestion_aversion
            
        # 2. 软拥堵惩罚 (保持不变)
        if cp.congestion_aversion > 1e-3:
            factor = self.cfg.SA_SOFT_CONGESTION_PENALTY_FACTOR * cp.congestion_aversion
            for c in sol.global_edge_counts.values():
                if c > 1: 
                    soft_pen += (c**2) * factor

        # =========================================================
        # [修正] 3. 运营成本惩罚 (Distance Penalty)
        # =========================================================
        dist_penalty = 0.0
        
        if self.cfg.DISTANCE_PENALTY_FACTOR > -1e6:
            current_total_dist = 0.0
            
            # [安全获取距离矩阵]
            # 优先使用注入的 self.trans，如果没有则尝试从 ProblemData 获取
            dist_source = getattr(self, 'trans', None)
            if dist_source is None and hasattr(self, 'pd'):
                 # 尝试常见的命名
                 dist_source = getattr(self.pd, 'dist_matrix', None)
                 if dist_source is None:
                     dist_source = getattr(self.pd, 'distances', None)

            if dist_source:
                for (u, v), count in sol.global_edge_counts.items():
                    if count > 0:
                        # 兼容 dict.get 和 numpy/list 索引
                        if isinstance(dist_source, dict):
                            edge_dist = dist_source.get((u, v), 0.0)
                        else:
                            try:
                                edge_dist = dist_source[u][v]
                            except:
                                edge_dist = 0.0
                        current_total_dist += edge_dist * count
            
            sol.total_dist = current_total_dist
            dist_penalty = current_total_dist * self.cfg.DISTANCE_PENALTY_FACTOR

        # 更新 Fitness
        sol.fitness = sol.total_util - pen - soft_pen - dist_penalty


    # --- [新] 极速运行循环 (备份 -> 变异 -> 回滚) ---
    def run(self):
        # === [Fix] 熔断保护 ===
        # 如果初始解本身就是无效的 (fitness 为负无穷)，说明 evaluate_full 提前退出了
        # 此时 v_stats 可能是不完整的，强行运行会导致 KeyError。
        # 直接返回失败即可。
        if self.curr.fitness <= -1e5 + 1.0: # 加上一点浮点数容差
            # 返回格式必须和你代码中 worker_task 接收的解包数量一致
            # 假设 worker_task 是 best_sol, fit, metrics = sa.run()
            # 这里的 metrics 返回空字典即可
            return self.best, self.curr.fitness, {}
        t = self.cfg.SA_INITIAL_TEMP
        
        # [Fix] 1. 初始化统计数据
        initial_fitness = self.curr.fitness
        accepted_moves = 0
        total_moves = 0
        
        while t > self.cfg.SA_FINAL_TEMP:
            for _ in range(self.cfg.SA_METROPOLIS_LEN):
                # 1. 准备：选车并备份
                active = [v for v, seq in self.curr.node_sequences.items() if len(seq) >= 2]
                if not active: continue
                v_id = random.choice(active)
                # ===============================================================
                # [关键修复] 自愈机制 (Self-Healing)
                # 防止 evaluate_full 漏算或数据丢失导致的 KeyError
                # ===============================================================
                if v_id not in self.curr.v_stats:
                    # 发现数据丢失，立刻原地重算补救
                    # 使用当前已知的缓存中断边进行计算
                    rescue_res = self._calc_vehicle_route(
                        v_id, 
                        self.curr.node_sequences[v_id], 
                        self.curr.path_choices[v_id], 
                        self.curr.cached_disrupted_edges
                    )
                    # 如果重算出来还是无效，这辆车就彻底没法用了，跳过
                    if not rescue_res['valid']:
                        continue
                    # 补回数据
                    self.curr.v_stats[v_id] = rescue_res
                # ===============================================================

                # 备份旧状态 (此时 v_stats[v_id] 一定存在)
                old_seq = self.curr.node_sequences[v_id][:]
                old_choices = self.curr.path_choices[v_id][:]
                old_stats = self.curr.v_stats[v_id]  # <--- 这里再也不会报 KeyError 了
                old_fitness = self.curr.fitness
                
                # 2. 变异
                self.curr.mutate(target_v_id=v_id)
                
                # 3. 计算新状态
                new_res = self._calc_vehicle_route(
                    v_id, self.curr.node_sequences[v_id], self.curr.path_choices[v_id], 
                    self.curr.cached_disrupted_edges
                )
                
                if not new_res['valid']:
                    self.curr.node_sequences[v_id] = old_seq
                    self.curr.path_choices[v_id] = old_choices
                    continue

                # 4. 更新全局数据
                for e in old_stats['edges']:
                    self.curr.global_edge_counts[e] -= 1
                    if self.curr.global_edge_counts[e] == 0: del self.curr.global_edge_counts[e]
                for e in new_res['edges']:
                    self.curr.global_edge_counts[e] += 1
                
                delta_util = new_res['util'] - old_stats['util']
                delta_dups = new_res['dups'] - old_stats['dups']
                self.curr.total_util += delta_util
                self.curr.total_dups += delta_dups
                
                self._update_fitness_from_totals(self.curr)
                new_fitness = self.curr.fitness
                
                # [Fix] 2. 统计尝试次数
                total_moves += 1

                # 5. 接受准则
                if new_fitness > old_fitness or (t>0 and random.random() < math.exp((new_fitness - old_fitness)/t)):
                    # [接受]
                    # [Fix] 3. 统计接受次数
                    accepted_moves += 1
                    
                    self.curr.v_stats[v_id] = new_res
                    if new_fitness > self.best.fitness:
                        self.best = self.curr.copy()
                else:
                    # [拒绝] -> 回滚
                    self.curr.node_sequences[v_id] = old_seq
                    self.curr.path_choices[v_id] = old_choices
                    
                    for e in new_res['edges']:
                        self.curr.global_edge_counts[e] -= 1
                        if self.curr.global_edge_counts[e] == 0: del self.curr.global_edge_counts[e]
                    for e in old_stats['edges']:
                        self.curr.global_edge_counts[e] += 1
                        
                    self.curr.total_util -= delta_util
                    self.curr.total_dups -= delta_dups
                    self.curr.fitness = old_fitness

            t *= self.cfg.SA_ALPHA
            
        # [Fix] 4. 打包 Metrics
        metrics = {
            'acceptance_rate': accepted_moves / max(1, total_moves),
            'initial_fitness': initial_fitness,
            'final_fitness': self.curr.fitness,
        }
        
        # [Fix] 返回 3 个值
        return self.best, self.best.fitness, metrics

# ==============================================================================
# GRASP 部分
# ==============================================================================

class DemandSupplyTracker:
    def __init__(self, pd):
        self.d_nodes, self.s_nodes = {}, {}
        for n, goods in pd.demands.items():
            for g, q in goods.items():
                if q < 0: self.d_nodes[(n,g)] = {'n':n,'g':g,'rem':abs(q)}
                elif q > 0: self.s_nodes[(n,g)] = {'n':n,'g':g,'rem':q}

class AssignmentSolution:
    def __init__(self, pd, cfg):
        self.pd, self.cfg = pd, cfg
        self.assignments = defaultdict(list)
        self.tracker = DemandSupplyTracker(pd)
        self.locs = {v: info['L'] for v, info in pd.vehicles.items()}
        self.times = {v: 0.0 for v in pd.vehicles}
        
        self.nb_caps = NumbaTypedDict.empty(nb_types.int64, nb_types.DictType(nb_types.int64, nb_types.float64))
        for v, caps in pd.capacities.items():
            d = NumbaTypedDict.empty(nb_types.int64, nb_types.float64)
            for g, c in caps.items(): d[g] = float(c)
            self.nb_caps[v] = d

    def add_task(self, v_id, s_node, d_node, g_id, q):
        # [Fix] 1. 动态核验：防止候选者列表数据过期导致的超量配送
        # 获取当前实时的剩余需求和剩余供应
        current_rem_demand = self.tracker.d_nodes.get((d_node, g_id), {}).get('rem', 0.0)
        current_rem_supply = self.tracker.s_nodes.get((s_node, g_id), {}).get('rem', 0.0)
        
        # 真正的执行量是：原计划量、剩余需求、剩余供应 三者的最小值
        # 这一步至关重要！它解决了 "V1送完了，V2还拿着旧的订单条来送货" 的问题
        real_q = min(q, current_rem_demand, current_rem_supply)
        
        # 如果实际上已经不需要送了，或者没货了，直接返回失败
        if real_q < 1e-6: 
            return False

        if len(self.assignments[v_id]) >= self.cfg.MAX_TASKS_PER_VEHICLE: return False
        
        # Check Cap (使用 real_q 计算)
        curr = self.assignments[v_id]
        g_ids = np.array([t.good_id for t in curr] + [g_id], dtype=np.int64)
        qs = np.array([t.quantity for t in curr] + [real_q], dtype=np.float64)
        if not _jit_check_capacity(g_ids, qs, self.nb_caps[v_id]): return False
        
        # [NEW] 更新时间
        curr_loc = self.locs[v_id]
        d1 = self.pd.get_matrix_distance(curr_loc, s_node)
        d2 = self.pd.get_matrix_distance(s_node, d_node)
        
        speed = self.pd.vehicles[v_id]['v']
        cost_time = ((d1 + d2) / max(0.1, speed)) * 60
        
        self.times[v_id] += cost_time
        
        # 使用 real_q 创建任务
        t = Task(v_id, s_node, d_node, g_id, real_q)
        self.assignments[v_id].append(t)
        
        # 扣减 (使用 real_q)
        self.tracker.d_nodes[(d_node,g_id)]['rem'] -= real_q
        self.tracker.s_nodes[(s_node,g_id)]['rem'] -= real_q
        self.locs[v_id] = d_node
        return True



    def to_dict(self):
        return {'assignments': {v: [{'v':t.vehicle_id,'p':t.pickup_node,'d':t.delivery_node,'g':t.good_id,'q':t.quantity} for t in ts] for v, ts in self.assignments.items()}}
    
# ==============================================================================
# Worker Logic (全局变量 + Init + Worker Function)
# ==============================================================================

WORKER_PD: Optional[ProblemData] = None
WORKER_PL: Optional[PathLibrary] = None

def init_worker(worker_config):  # 1. 参数改为接收 config 对象
    global WORKER_PD, WORKER_PL
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # 假设你的距离矩阵文件名是固定的，或者也在 config 里
    matrix_path = "None.txt" 
    
    # 2. 从 worker_config 中获取路径
    WORKER_PD = ProblemData(worker_config.PROBLEM_DATA_PATH, '', matrix_filepath=matrix_path)
    
    # 3. 将 worker_config 传递给 PathLibrary
    WORKER_PL = PathLibrary(WORKER_PD, worker_config)

def worker_task(cfg: GraspConfig, aff_mat: Dict, trans_mat: Dict, seed: int = None, task_idx: int = 0):
    
    # [修改 2] 强制重置子进程的随机种子 (最先执行)
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    cp = cfg.c_params
    sol = AssignmentSolution(WORKER_PD, cfg)
    H_LIMIT = WORKER_PD.H
    
    # [Fix Matrix] 定义距离获取辅助函数
    # 优先读矩阵(O(1)极速)，如果读不到(返回10000)，则读KSP缓存(O(1)查表)
    def get_heuristic_dist(u, v):
        d = WORKER_PD.get_matrix_distance(u, v)
        if d >= 9999.0 and WORKER_PL is not None:
            # 尝试从 KSP 缓存中获取第一条最短路
            ksp = WORKER_PL.get_k_shortest_paths(u, v, 1)
            if ksp:
                return ksp[0]['dist']
        return d

    # 最大尝试次数
    max_attempts = len(WORKER_PD.nodes) * 2
    
    # === 策略判断 ===
    use_fast_path = (cp.prune_strength > 0.99)
    
    for _ in range(max_attempts):
        candidates = []
        
        remaining_d = {k:v for k,v in sol.tracker.d_nodes.items() if v['rem'] > 1e-6}
        if not remaining_d: break
        
        active_vehicles = [v for v in WORKER_PD.vehicles if len(sol.assignments[v]) < cfg.MAX_TASKS_PER_VEHICLE]
        
        # === 候选生成循环 ===
        for v_id in active_vehicles:
            v_curr_loc = sol.locs[v_id]
            speed = max(0.1, WORKER_PD.vehicles[v_id]['v'])
            
            for (dn, dg), d_info in remaining_d.items():
                
                best_item_tuple = None
                best_score_fast = -1e9
                temp_candidates_tuples = []
                
                for (sn, sg), s_info in sol.tracker.s_nodes.items():
                    if sg != dg or s_info['rem'] < 1e-6: continue
                    cap = WORKER_PD.capacities[v_id].get(dg, 0)
                    if cap <= 1e-6: continue
                    
                    q = min(d_info['rem'], s_info['rem'], cap)
                    
                    # [Fix Matrix] 使用增强版距离获取
                    dist_empty = get_heuristic_dist(v_curr_loc, sn)
                    dist_task = get_heuristic_dist(sn, dn)
                    
                    total_dist = dist_empty + dist_task
                    arrival_time = sol.times[v_id] + (total_dist / speed) * 60
                    
                    # 评分公式
                    base_score = q * WORKER_PD.weights[dn].get(dg, 1.0)
                    
                    if cp.temporal_sens > 1e-3:
                        time_margin = max(0, H_LIMIT - arrival_time)
                        time_term = 1.0 + cp.temporal_sens * (time_margin - 1.0)
                        if time_term < 1e-6: time_term = 1e-6 
                    else:
                        time_term = 1.0
                    
                    if cp.spatial_sens > 1e-3:
                        dist_denom = 1.0 + cp.spatial_sens * (total_dist + 9.0)
                    else:
                        dist_denom = 1.0
                    
                    aco_val = 1.0
                    if cp.history_sens > 1e-3:
                         aff = aff_mat.get(v_id, {}).get(sn, 1.0)
                         trans = trans_mat.get((v_curr_loc, sn), 1.0)
                         aco_val = 1.0 + cp.history_sens * (aff * trans - 1.0)
                    
                    final_score = (base_score * time_term * aco_val) / dist_denom

                    # 通道分流
                    if use_fast_path:
                        if final_score > best_score_fast:
                            best_score_fast = final_score
                            best_item_tuple = (v_id, sn, dn, dg, q, final_score)
                    else:
                        temp_candidates_tuples.append((final_score, v_id, sn, dn, dg, q))
                
                # 结算
                if use_fast_path:
                    if best_item_tuple:
                        candidates.append({
                            'v': best_item_tuple[0], 's': best_item_tuple[1], 
                            'd': best_item_tuple[2], 'g': best_item_tuple[3], 
                            'q': best_item_tuple[4], 'score': best_item_tuple[5]
                        })
                else:
                    if temp_candidates_tuples:
                        temp_candidates_tuples.sort(key=lambda x: x[0], reverse=True)
                        N_total = len(temp_candidates_tuples)
                        K = int(N_total * (1.0 - cp.prune_strength) + 1.0 * cp.prune_strength)
                        K = max(1, K)
                        for item in temp_candidates_tuples[:K]:
                            candidates.append({
                                'v': item[1], 's': item[2], 'd': item[3], 
                                'g': item[4], 'q': item[5], 'score': item[0]
                            })

        if not candidates: break
        
        # --- 准入过滤 ---
        BIG_M = 1e9
        threshold = -BIG_M * (1.0 - cp.filter_strict)
        valid_candidates = [c for c in candidates if c['score'] > threshold]
        
        if not valid_candidates: break
        
        # --- 概率选择与批处理 ---
        valid_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        total_valid = len(valid_candidates)
        batch_size = int(1 + (total_valid - 1) * cp.prune_strength)
        
        tasks_to_try = []
        
        if batch_size == 1:
            # === 单步模式 (并行版逻辑) ===
            rcl_len = min(len(valid_candidates), 10)
            rcl_pool = valid_candidates[:rcl_len]
            
            if cp.select_temp < 1e-3:
                chosen = random.choice(rcl_pool)
            else:
                scores = np.array([max(1e-6, c['score']) for c in rcl_pool])
                try:
                    weights = np.power(scores, cp.select_temp)
                    chosen = random.choices(rcl_pool, weights=weights, k=1)[0]
                except:
                    chosen = random.choice(rcl_pool)
            tasks_to_try.append(chosen)
            
        else:
            # === 改进版极速模式：收缩式批量轮盘赌 (Alpha Inject) ===
            pool_candidates = valid_candidates[:batch_size]
            pool_len = len(pool_candidates)

            # [Alpha Inject] 获取 DRL 动作 Alpha
            alpha = getattr(cfg, 'ALPHA', 1.0) 
            
            num_to_select = int(pool_len * alpha)
            num_to_select = max(1, num_to_select)
            num_to_select = min(num_to_select, pool_len)

            if num_to_select == pool_len:
                tasks_to_try = pool_candidates
            else:
                scores = np.array([c['score'] for c in pool_candidates])
                scores = np.maximum(scores, 1e-6)

                temp = cp.select_temp
                try:
                    if temp < 1e-3:
                        weights = np.ones_like(scores)
                    else:
                        weights = np.power(scores, temp)
                    
                    weight_sum = np.sum(weights)
                    if weight_sum > 1e-9:
                        weights = weights / weight_sum
                        selected_indices = np.random.choice(
                            pool_len,
                            size=num_to_select,
                            replace=False, 
                            p=weights
                        )
                        tasks_to_try = [pool_candidates[i] for i in selected_indices]
                    else:
                        tasks_to_try = pool_candidates[:num_to_select]
                except Exception:
                    tasks_to_try = pool_candidates[:num_to_select]
        
        # --- 执行添加 ---
        added_count = 0
        for task in tasks_to_try:
            success = sol.add_task(task['v'], task['s'], task['d'], task['g'], task['q'])
            if success:
                added_count += 1
        
        if added_count == 0:
            break
            
    # 检查有效性
    has_tasks = any(len(tasks) > 0 for tasks in sol.assignments.values())
    if not has_tasks: return -1e5, None, {} # [Fix Metrics] 返回空字典防报错
    
    # SA 部分
    sol_dict = sol.to_dict()
    pd_copy = copy.copy(WORKER_PD)
    pd_copy.tasks_by_vehicle = {}
    pd_copy.task_map = {}
    
    for v, tasks in sol_dict['assignments'].items():
        pd_copy.tasks_by_vehicle[v] = []
        pd_copy.task_map[v] = {}
        for t in tasks:
            tsk = Task(t['v'], t['p'], t['d'], t['g'], t['q'])
            pd_copy.tasks_by_vehicle[v].append(tsk)
            pd_copy.task_map[v][tsk.delivery_node] = tsk
            
    sa = SimulatedAnnealing(pd_copy, cfg, WORKER_PL)
    sa.trans = trans_mat
    # [Fix Metrics] 接收 3 个返回值
    best_sol, fit, metrics = sa.run()
    
    # [Fix Metrics] 返回 fit, sol, metrics
    return (task_idx, fit, best_sol.to_full_dict(), metrics)

# ==============================================================================
# Optimizer (主进程)
# ==============================================================================

class ParallelGraspOptimizer:
    def __init__(self, config: GraspConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.best_overall_fitness = -1e5
        self.best_overall_solution_assignments = None
        self.history = []
        self.last_reset_gen = 0
        # 累计求解时间（秒）
        self._cumulative_time = 0.0

        self.aff = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.trans = defaultdict(lambda: 1.0)
    # =========================================================
        # [修改] 自动化 KSP 缓存 & 数据加载
        # =========================================================
        if config.USE_KSP_CACHE:
            self.logger.info(f"正在准备算例数据与缓存: {config.PROBLEM_DATA_PATH}")
            
            # 1. 加载数据对象 (必须保留 self.problem_data 给 Env 使用!)
            # matrix_filepath="" 表示主进程不加载距离矩阵，节省内存，只保留拓扑结构
            self.problem_data = ProblemData(config.PROBLEM_DATA_PATH, '', matrix_filepath="")
            
            # 2. 触发 KSP 计算/读取
            # 传入 self.problem_data，避免重复解析文件
            from src.env import ksp_manager
            ksp_manager.get_or_compute_ksp(
                self.problem_data, 
                config, 
                config.PROBLEM_DATA_PATH
            )
            
            # 注意：不要 del self.problem_data，Env 需要它！
            
        else:
            # 如果不缓存，也要加载 pd 给 Env 用
            self.problem_data = ProblemData(config.PROBLEM_DATA_PATH, '', matrix_filepath="")

        # =========================================================        
        num_workers = self.config.NUM_WORKERS or os.cpu_count()
        self.logger.info(f"正在初始化进程池 ({num_workers} workers)...")
        
        # 进程池初始化时会自动加载 KSP 缓存
        self.executor = ProcessPoolExecutor(
            max_workers=num_workers, 
            initializer=init_worker, 
            initargs=(self.config,)
        )
        self.logger.info("进程池初始化完毕。")

    def shutdown(self):
        if self.executor:
            # [修改] 将 wait=False 改为 wait=True
            # 作用：强制等待所有子进程关闭并释放资源（FD）后，代码才继续向下执行
            self.executor.shutdown(wait=True) 
            self.executor = None

    def _update_learning(self, best_sol):
        if not best_sol: return
        rho, Q = self.config.ACO_EVAPORATION, self.config.ACO_Q
        
        # 1. 挥发点信息素
        for k1 in self.aff:
            for k2 in self.aff[k1]: 
                self.aff[k1][k2] = max(0.1, self.aff[k1][k2] * (1-rho))
        
        # 2. 挥发边信息素 + [V3 剪枝逻辑保留]
        keys_to_remove = []
        for k in self.trans:
            self.trans[k] = max(0.1, self.trans[k] * (1-rho))
            # 剪枝：如果接近 1.0，删除以减少进程通信开销
            if abs(self.trans[k] - 1.0) < 0.05: 
                keys_to_remove.append(k)
        
        for k in keys_to_remove:
            del self.trans[k]
            
        # 3. 沉积 (只增强全局最优)
        for v, tasks in best_sol['assignments'].items():
            for t in tasks: self.aff[v][t['p']] += Q * rho
        for v, seq in best_sol['node_sequences'].items():
            for i in range(len(seq)-1): self.trans[(seq[i], seq[i+1])] += Q * rho

    def run_one_generation(self, gen_idx, total_generations=1000):
        """
        :param gen_idx: 当前代数 (1-based)
        :param total_generations: 预计总代数
        """
        t0 = time.time()
        
        # 记录本代开始前的全局最优，用于判断是否更新
        prev_global_best = self.best_overall_fitness
        
        # --- 1. 获取当前动作/参数信息 (从配置中读取) ---
        current_pop = self.config.POPULATION_SIZE
        current_alpha = self.config.ALPHA
        current_sa_len = self.config.SA_METROPOLIS_LEN
        
        # ------------------------------------------------------------------
        # [V5+ 最终增强] 自适应动态重启 (保持原有逻辑)
        # ------------------------------------------------------------------
        cp = self.config.c_params
        progress = gen_idx / max(1, total_generations)
        base_limit = 3 + (12 * progress)
        current_stagnation_limit = int(base_limit / max(1e-4, cp.patience_inv))
        
        if (gen_idx - self.last_reset_gen) > current_stagnation_limit:
            if len(self.history) > current_stagnation_limit:
                recent_best = max(self.history[-current_stagnation_limit:])
                baseline = self.history[-current_stagnation_limit - 1]
                if recent_best <= baseline + 1e-6:
                    # 触发柔性重启
                    self.aff = defaultdict(lambda: defaultdict(lambda: 0.01))
                    self.trans = defaultdict(lambda: 0.01)
                    if self.best_overall_solution_assignments:
                        self._update_learning(self.best_overall_solution_assignments)
                    self.last_reset_gen = gen_idx
        # ------------------------------------------------------------------

        # 制作快照并并行执行
        aff_snap = {k: dict(v) for k,v in self.aff.items()}
        trans_snap = dict(self.trans)
        
        # 生成确定性种子列表
        task_seeds = [random.randint(0, 2**32 - 1) for _ in range(self.config.POPULATION_SIZE)]

        # [修改 1] 提交任务时传入索引 i (task_idx)
        futures = []
        for i in range(self.config.POPULATION_SIZE):
            seed = task_seeds[i]
            # 注意：请确保 worker_task 的签名已修改为接收 (..., seed, task_idx)
            fut = self.executor.submit(worker_task, self.config, aff_snap, trans_snap, seed, i)
            futures.append(fut)
            
        # [修改 2] 收集原始结果 (先存入列表，暂不处理)
        raw_results = []
        for fut in as_completed(futures):
            try:
                res = fut.result(timeout=120)
                raw_results.append(res)
            except TimeoutError:
                self.logger.warning("Worker Timeout!")
            except Exception as e:
                self.logger.error(f"Worker Failed: {e}")
        
        if not raw_results:
             # 如果全部失败，避免崩溃
             self.logger.critical("⚠️ ALL WORKERS FAILED!")
             # 这里可以根据需要返回空字典或默认值，代码继续向下会由 fits 为空处理

        # [修改 3] 强制按 Task ID 排序 -> 核心确定性来源
        # 这样无论哪个 Worker 先跑完，我们处理数据的顺序永远是 0->1->2...
        # 假设 worker_task 返回 (task_idx, f, sol, metrics)
        raw_results.sort(key=lambda x: x[0])

        fits = []
        best_gen_fit = -1e5
        best_gen_sol = None
        
        # 统计累加器
        total_accept_rate = 0.0
        total_init_fit = 0.0
        total_final_fit = 0.0
        valid_worker_count = 0 

        # [修改 4] 按顺序处理有序结果
        for result in raw_results:
            # 防御性解包：兼容可能未更新 worker_task 的情况（但这会失去确定性）
            if len(result) == 4:
                task_idx, f, sol, metrics = result
            elif len(result) == 3:
                # 兼容旧代码，但强烈建议更新 worker_task 以返回 task_idx
                f, sol, metrics = result
            else:
                continue

            if f > -1e5:
                fits.append(f)
                
                # 更新本代最优
                # [关键] 因为是按 ID 顺序遍历，f == best 时不会更新 (保留 ID 小的)
                # 这保证了确定性
                if f > best_gen_fit: 
                    best_gen_fit = f
                    best_gen_sol = sol
                
                # 累加指标 (保持你的逻辑)
                total_accept_rate += metrics.get('acceptance_rate', 0.5) 
                total_init_fit += metrics.get('initial_fitness', f) 
                total_final_fit += f
                valid_worker_count += 1
        
        # --- 2. 统计数据计算 (保持原有逻辑) ---
        mean_fitness = np.mean(fits) if fits else -1e5
        std_fitness = np.std(fits) if fits else 0.0
        elapsed_time = time.time() - t0
        
        # 更新累计时间 (保持你的逻辑)
        try:
            self._cumulative_time += elapsed_time
        except Exception:
            self._cumulative_time = elapsed_time
        
        # 计算平均指标 (保持你的逻辑)
        avg_acc_rate = total_accept_rate / valid_worker_count if valid_worker_count > 0 else 0.0
        avg_init_fit = total_init_fit / valid_worker_count if valid_worker_count > 0 else mean_fitness
        avg_final_fit = total_final_fit / valid_worker_count if valid_worker_count > 0 else mean_fitness
        
        # --- 3. 更新全局最优 (保持原有逻辑) ---
        updated_flag = False
        epsilon = 1e-4 
        
        if best_gen_fit > self.best_overall_fitness + epsilon:
            self.best_overall_fitness = best_gen_fit
            self.best_overall_solution_assignments = best_gen_sol
            self.history.append(best_gen_fit) 
            updated_flag = True
        else:
            self.history.append(self.best_overall_fitness)
        
        # 信息素更新
        self._update_learning(best_gen_sol)
        gc.collect()

        # --- 4. 日志打印 (保持原有逻辑) ---
        update_str = "🎯NEW BEST!" if updated_flag else f"(Best: {self.best_overall_fitness:.2f})"
        
        log_msg = (
            f"Step [{gen_idx:3d}/{total_generations}] | "
            f"⏳ {self._cumulative_time:6.2f}s | "
            f"🎮 Act: [Pop={current_pop:2d}, α={current_alpha:.2f}, SA_Len={current_sa_len:3d}] | "
            f"📈 Obj: Avg={mean_fitness:8.2f} / Max={best_gen_fit:8.2f} -> {update_str}"
        )

        enable_step = globals().get('ENABLE_PER_STEP_LOGS', False)
        if enable_step:
            self.logger.info(log_msg)
        else:
            if updated_flag:
                try:
                    if best_gen_fit >= 0:
                        self.logger.info(log_msg)
                except Exception:
                    self.logger.info(log_msg)
            else:
                pass

        # 返回字典 (保持原有逻辑)
        return {
            'best_fitness_in_gen': best_gen_fit,
            'mean_fitness_in_gen': mean_fitness,
            'std_fitness_in_gen': std_fitness,
            'new_overall_best_fitness': self.best_overall_fitness,
            'gen_time_seconds': elapsed_time,
            'avg_acceptance_rate': avg_acc_rate,
            'avg_initial_fitness': avg_init_fit,
            'avg_final_fitness': avg_final_fit
        }