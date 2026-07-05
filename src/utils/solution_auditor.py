import numpy as np
from collections import defaultdict
import logging

# 引用核心数据结构
from src.algorithms.solver_coreDRL_train import ProblemData, PathLibrary

class SolutionAuditor:
    def __init__(self, pd: ProblemData, pl: PathLibrary, sol_dict: dict):
        self.pd = pd
        self.pl = pl
        self.sol = sol_dict
        self.epsilon = 1e-5
        
        self.audit_fitness = 0.0
        self.solver_fitness = sol_dict.get('fitness', 0.0)
        
        self.edge_flows = defaultdict(float)
        self.disrupted_edges = set()
        self.vehicle_audit_log = {}
        self.violations = []
        self.stats = {}

    def run_audit(self):
        print("\n" + "="*60)
        print("🕵️  开始执行解方案审查与统计 (Solution Audit)")
        print(f"   [目标] 检查解的物理自洽性 (Fix: 修复了重复计费Bug)")
        print("="*60)
        
        self._phase1_calculate_network_flow()
        self._phase2_verify_constraints_and_objective()
        self._phase3_generate_statistics()
        self._print_report()

    def _phase1_calculate_network_flow(self):
        """阶段一：流量重算 (使用指针逻辑防止重复计算)"""
        self.edge_flows.clear()
        
        for v_id, seq in self.sol['node_sequences'].items():
            if not seq: continue
            
            tasks = self.sol['assignments'].get(v_id, [])
            pickup_ptr = 0
            delivery_ptr = 0
            
            curr_node = self.pd.vehicles[v_id]['L']
            full_seq = seq + [curr_node]
            
            # 当前车上载重
            current_load = defaultdict(float)
            
            # 处理起点的初始装载 (极其罕见)
            pickup_ptr, delivery_ptr = self._process_node_load(
                curr_node, tasks, pickup_ptr, delivery_ptr, current_load
            )
            
            choices = self.sol['path_choices'].get(v_id, [0] * len(full_seq))
            
            for i, next_node in enumerate(full_seq):
                path_idx = choices[i] if i < len(choices) else 0
                ksp = self.pl.get_k_shortest_paths(curr_node, next_node, path_idx + 1)
                
                if not ksp: 
                    curr_node = next_node
                    continue 
                
                actual_path = ksp[path_idx]['path']
                
                # 计算流量
                active_flow = sum(q for q in current_load.values() if q > self.epsilon)
                if active_flow > self.epsilon:
                    for j in range(len(actual_path) - 1):
                        u, v = actual_path[j], actual_path[j+1]
                        self.edge_flows[(u, v)] += active_flow
                
                # 到达节点：更新载重
                pickup_ptr, delivery_ptr = self._process_node_load(
                    next_node, tasks, pickup_ptr, delivery_ptr, current_load
                )
                
                curr_node = next_node

        # 确定中断边
        sorted_edges = sorted(self.edge_flows.items(), key=lambda x: x[1], reverse=True)
        self.disrupted_edges = set()
        count = 0
        for edge, flow in sorted_edges:
            if count >= self.pd.U: break
            if flow > self.epsilon:
                self.disrupted_edges.add(edge)
                count += 1
        
        self.stats['max_flow_edge'] = sorted_edges[0] if sorted_edges else (None, 0.0)

    def _phase2_verify_constraints_and_objective(self):
        """阶段二：约束审查 (使用指针逻辑)"""
        self.audit_fitness = 0.0
        self.real_delivery_counts = defaultdict(float)
        
        for v_id, seq in self.sol['node_sequences'].items():
            if not seq: 
                self.vehicle_audit_log[v_id] = {'status': 'unused', 'obj': 0.0}
                continue
            
            tasks = self.sol['assignments'].get(v_id, [])
            pickup_ptr = 0
            delivery_ptr = 0
            
            v_cap = self.pd.capacities[v_id]
            v_speed = max(0.1, self.pd.vehicles[v_id]['v'])
            
            # 记录计划送货量
            plan_delivery_counts = defaultdict(float)
            for t in tasks:
                plan_delivery_counts[(t['d'], t['g'])] += t['q']
            
            # 超量检查
            for (node, good), planned_q in plan_delivery_counts.items():
                demand_limit = abs(self.pd.demands[node].get(good, 0))
                if planned_q > demand_limit + self.epsilon:
                    self.violations.append(f"[超量] 节点{node} G{good} 计划{planned_q:.1f} > 需求{demand_limit:.1f}")

            curr_node = self.pd.vehicles[v_id]['L']
            curr_load = defaultdict(float)
            curr_time = 0.0
            
            is_disrupted = False
            vehicle_obj = 0.0
            full_seq = seq + [curr_node]
            choices = self.sol['path_choices'].get(v_id, [0] * len(full_seq))
            
            # 起点处理
            pickup_ptr, delivery_ptr, gained_obj = self._process_node_events_ptr(
                curr_node, tasks, pickup_ptr, delivery_ptr, curr_load, 
                curr_time, is_disrupted, v_id
            )
            vehicle_obj += gained_obj

            for i, next_node in enumerate(full_seq):
                path_idx = choices[i] if i < len(choices) else 0
                ksp = self.pl.get_k_shortest_paths(curr_node, next_node, path_idx + 1)
                
                if not ksp:
                    self.violations.append(f"[连通性] V{v_id} {curr_node}->{next_node} 断路")
                    break
                
                leg_data = ksp[path_idx]
                phys_path = leg_data['path']
                dist = leg_data['dist']
                
                # 1. 检查中断
                if not is_disrupted:
                    for j in range(len(phys_path) - 1):
                        e = (phys_path[j], phys_path[j+1])
                        if e in self.disrupted_edges:
                            is_disrupted = True
                            break
                
                # 2. 检查容量 (Ratio)
                usage = 0.0
                for g, q in curr_load.items():
                    c = v_cap.get(g, 0)
                    if c > 0: usage += q / c
                    elif q > self.epsilon:
                        self.violations.append(f"[容量] V{v_id} 装载非法物资 G{g}:{q}")
                
                if usage > 1.0 + 1e-3: # 稍微放宽一点点误差
                    self.violations.append(f"[容量] V{v_id} 路段 {curr_node}->{next_node} 超载 (Ratio:{usage:.2%})")

                # 3. 更新时间
                curr_time += (dist / v_speed) * 60
                
                # 4. 节点事件
                pickup_ptr, delivery_ptr, step_obj = self._process_node_events_ptr(
                    next_node, tasks, pickup_ptr, delivery_ptr, curr_load, 
                    curr_time, is_disrupted, v_id
                )
                vehicle_obj += step_obj
                
                curr_node = next_node
            
            status = 'disrupted' if is_disrupted else 'active'
            self.vehicle_audit_log[v_id] = {'status': status, 'obj': vehicle_obj}
            self.audit_fitness += vehicle_obj

    # --- 辅助方法: 基于指针的任务消费 ---
    
    def _process_node_load(self, node, tasks, p_ptr, d_ptr, load_dict):
        """仅更新载重，不计算目标 (用于 Phase 1)"""
        # 1. 尝试装货 (Pickups)
        # 必须先装后卸，符合 decode_split 的容量假设
        while p_ptr < len(tasks):
            t = tasks[p_ptr]
            if t['p'] == node:
                load_dict[t['g']] += t['q']
                p_ptr += 1
            else:
                break # 遇到非当前节点的任务，停止，等待到了那个节点再装
        
        # 2. 尝试卸货 (Deliveries)
        while d_ptr < len(tasks):
            t = tasks[d_ptr]
            if t['d'] == node:
                load_dict[t['g']] -= t['q']
                d_ptr += 1
            else:
                break
                
        return p_ptr, d_ptr

    def _process_node_events_ptr(self, node, tasks, p_ptr, d_ptr, load_dict, time, disrupted, v_id):
        """更新载重并计算目标 (用于 Phase 2)"""
        added_obj = 0.0
        
        # 1. Pickups
        while p_ptr < len(tasks):
            t = tasks[p_ptr]
            if t['p'] == node:
                load_dict[t['g']] += t['q']
                p_ptr += 1
            else:
                break
        
        # 2. Deliveries
        while d_ptr < len(tasks):
            t = tasks[d_ptr]
            if t['d'] == node:
                load_dict[t['g']] -= t['q']
                # 结算目标
                if not disrupted and time <= self.pd.H + self.epsilon:
                    w = self.pd.weights[node].get(t['g'], 1.0)
                    score = w * t['q'] * (self.pd.H - time)
                    added_obj += score
                    self.real_delivery_counts[(node, t['g'])] += t['q']
                d_ptr += 1
            else:
                break
                
        return p_ptr, d_ptr, added_obj

    def _phase3_generate_statistics(self):
        # 保持原样
        self.stats['n_total_veh'] = len(self.pd.vehicles)
        self.stats['n_used_veh'] = len([v for v in self.vehicle_audit_log if self.vehicle_audit_log[v]['status'] != 'unused'])
        self.stats['n_disrupted'] = len([v for v in self.vehicle_audit_log if self.vehicle_audit_log[v]['status'] == 'disrupted'])
        
        total_demand = sum(abs(q) for n_d in self.pd.demands.values() for q in n_d.values() if q < 0)
        total_satisfied = sum(self.real_delivery_counts.values())
        self.stats['satisfaction_rate'] = (total_satisfied / max(1, total_demand)) * 100
        self.stats['total_demand'] = total_demand
        self.stats['total_satisfied'] = total_satisfied

    def _print_report(self):
        # 保持原样，略微精简
        print("\n📊 审计结果 (Audit Results):")
        print(f"  - 审计计算的物理目标值 (Physical Fitness): {self.audit_fitness:10.2f}")
        # print(f"  - 算法内部记录的目标值 (Internal Fitness): {self.solver_fitness:10.2f}")
        
        print(f"\n📈 统计指标:")
        print(f"  - 车辆状态: {self.stats['n_used_veh']} 辆已用, 其中 {self.stats['n_disrupted']} 辆遭遇中断")
        print(f"  - 需求满足率: {self.stats['satisfaction_rate']:.2f}% (量: {self.stats['total_satisfied']:.1f}/{self.stats['total_demand']:.1f})")
        
        print(f"\n🌪️ 网络中断审查:")
        print(f"  - 识别中断边数: {len(self.disrupted_edges)} / {self.pd.U}")
        edge, flow = self.stats.get('max_flow_edge', (None, 0))
        if edge:
            print(f"  - 最大流量边: {edge[0]}->{edge[1]} (Flow: {flow:.2f})")

        print("\n⚖️  物理约束与自洽性审查:")
        if not self.violations:
            print("  ✅ [通过] 解方案完全自洽。")
        else:
            print(f"  ❌ [失败] 发现 {len(self.violations)} 项自洽性违规:")
            # 只显示前10条
            unique_violations = sorted(list(set(self.violations)))
            for i, v in enumerate(unique_violations[:10]):
                print(f"     {i+1}. {v}")
            if len(unique_violations) > 10:
                print(f"     ... (共 {len(unique_violations)} 项)")
        print("="*60 + "\n")