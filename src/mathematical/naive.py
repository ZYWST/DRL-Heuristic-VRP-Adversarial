import gurobipy as gp
from gurobipy import GRB
import io
import re # Needed for the reader function
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Optional

# =============================================================================
#  Data Reading Function (read_ampl_dat_to_gurobipy - Updated for H, U and Distance Options)
# =============================================================================
def read_ampl_dat_to_gurobipy(dat_content: str, distance_mode: int = 0):
    """
    Parses a string in AMPL .dat format into a dictionary of Gurobi-friendly
    data structures. This function is specifically tailored to the vehicle
    routing problem's data format.
    
    Args:
        dat_content: AMPL data file content as string
        distance_mode: 0 = use E distances, 1 = calculate from coordinates
    """
    data = {
        'N': [], 'px': {}, 'py': {}, 'E': [], 'd': {},
        'K': [], 'v': {}, 'L': {}, 'G': [], 'c': {},
        'a': {}, 'w': {}, 'H': 460.0, 'U': 1 # Default values
    }
    f = io.StringIO(dat_content)
    lines = f.readlines()
    reading_mode = None
    param_indices = []

    print(f"📊 数据解析模式: {'坐标欧几里得距离' if distance_mode == 1 else 'E中指定距离'}")

    for line_num, line in enumerate(lines, 1):
        original_line = line
        line = line.strip()
        if '#' in line: line = line.split('#', 1)[0].strip()
        if not line: continue
        
        # Match simple param declarations like param H := 200;
        match_simple_param = re.match(r"param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*([0-9.]+)\s*;?", line, re.IGNORECASE)
        if match_simple_param:
            param_name = match_simple_param.group(1).upper()
            param_value = float(match_simple_param.group(2))
            if param_name == 'H':
                data['H'] = param_value
            elif param_name == 'U':
                data['U'] = int(param_value) # U should be an integer
            continue

        if line.endswith(';'):
            declaration = line[:-1].strip()
            if reading_mode and declaration: pass
            reading_mode = None
            if declaration.startswith("set G:="):
                 try:
                     items_str = declaration.split(':=', 1)[1].strip()
                     data['G'] = [int(item) for item in items_str.split(',')]
                     reading_mode = None
                 except Exception as e: print(f"Warning (Line {line_num}): Could not parse set G: {original_line}. Error: {e}")
            continue
            
        match_n_coords = re.match(r"param\s*:N:\s*px\s+py\s*:=", line, re.IGNORECASE)
        match_e_dist = re.match(r"param\s*:E:\s*d\s*=", line, re.IGNORECASE)
        match_k_v_l = re.match(r"param\s*:K:\s*v\s+L\s*:=", line, re.IGNORECASE)
        match_g_set = re.match(r"set\s+G\s*:=", line, re.IGNORECASE)
        match_c_cap = re.match(r"param\s+c\s*:\s*([\d\s]+)\s*:=", line, re.IGNORECASE)
        match_a_amount = re.match(r"param\s+a\s*:\s*([\d\s]+)\s*:=", line, re.IGNORECASE)
        match_w_weight = re.match(r"param\s+w\s*:=", line, re.IGNORECASE)

        if match_n_coords: reading_mode = 'N_coords'; continue
        elif match_e_dist: 
            if distance_mode == 0:
                reading_mode = 'E_d'  # Only read E distances in mode 0
            else:
                reading_mode = 'E_d_skip'  # Skip E distances in mode 1
            continue
        elif match_k_v_l: reading_mode = 'K_v_L'; continue
        elif match_g_set:
            reading_mode = 'G_set'
            try:
                items_str = line.split(':=', 1)[1].strip()
                if items_str: data['G'].extend([int(item) for item in items_str.replace(',', ' ').split()])
            except Exception as e: print(f"Warning (Line {line_num}): Could not parse part of set G: {original_line}. Error: {e}")
            continue
        elif match_c_cap:
            reading_mode = 'c_param'
            try: param_indices = [int(x) for x in match_c_cap.group(1).split()]
            except Exception as e: print(f"Warning (Line {line_num}): Could not parse indices for param c: {original_line}. Error: {e}"); reading_mode = None
            continue
        elif match_a_amount:
            reading_mode = 'a_param'
            try: param_indices = [int(x) for x in match_a_amount.group(1).split()]
            except Exception as e: print(f"Warning (Line {line_num}): Could not parse indices for param a: {original_line}. Error: {e}"); reading_mode = None
            continue
        elif match_w_weight: reading_mode = 'w_param'; continue
        
        parts = line.split()
        if not parts: continue
        try:
            if reading_mode == 'N_coords':
                node_id = int(parts[0]); px_val = float(parts[1]); py_val = float(parts[2])
                if node_id not in data['px']: data['N'].append(node_id); data['px'][node_id] = px_val; data['py'][node_id] = py_val
            elif reading_mode == 'E_d':
                i = int(parts[0]); j = int(parts[1]); dist = float(parts[2])
                data['E'].append((i, j)); data['d'][(i, j)] = dist
            elif reading_mode == 'E_d_skip':
                # Skip reading E distances when using coordinate mode
                continue
            elif reading_mode == 'K_v_L':
                k = int(parts[0]); v_val = float(parts[1]); l_val = int(parts[2])
                if k not in data['v']: data['K'].append(k); data['v'][k] = v_val; data['L'][k] = l_val
            elif reading_mode == 'G_set': data['G'].extend([int(item) for item in line.replace(',', ' ').split()])
            elif reading_mode == 'c_param':
                k = int(parts[0])
                if len(parts) - 1 == len(param_indices):
                    for idx, val_str in enumerate(parts[1:]): data['c'][k, param_indices[idx]] = float(val_str)
                else: print(f"Warning (Line {line_num}): Mismatched values for param c: {original_line}")
            elif reading_mode == 'a_param':
                n_node = int(parts[0])
                if len(parts) - 1 == len(param_indices):
                    for idx, val_str in enumerate(parts[1:]): data['a'][n_node, param_indices[idx]] = float(val_str)
                else: print(f"Warning (Line {line_num}): Mismatched values for param a: {original_line}")
            elif reading_mode == 'w_param':
                 if len(parts) == 3: data['w'][int(parts[0]), int(parts[1])] = float(parts[2])
                 elif len(parts) > 0 and 'default' not in line.lower(): print(f"Warning (Line {line_num}): Unexpected format for param w: {original_line}")
        except (ValueError, IndexError) as e: print(f"Warning (Line {line_num}): Skipping line: '{original_line}'. Error: {e}"); continue

    # Process distance data based on mode
    if distance_mode == 1:
        print("🔄 仅根据坐标更新E中已有边的距离...")
        _update_edge_distances_by_coordinates(data)
    
    data['N'].sort(); data['K'].sort(); data['G'].sort()
    final_data = {}
    final_data['N'] = data['N']; final_data['px'] = data['px']; final_data['py'] = data['py']
    final_data['K'] = data['K']; final_data['v'] = data['v']; final_data['L'] = data['L']; final_data['G'] = data['G']
    final_data['H'] = data['H']; final_data['U'] = data['U'] # Pass H and U to the final data
    
    valid_nodes_set = set(final_data['N'])
    valid_E_list = []
    valid_d_dict = {}
    for i, j in data['E']:
        if i in valid_nodes_set and j in valid_nodes_set:
            if (i, j) in data['d']: valid_E_list.append((i, j)); valid_d_dict[(i, j)] = data['d'][(i, j)]
            else: print(f"Warning: Edge ({i},{j}) in E list, no distance in d. Edge removed.")
    final_data['E'] = gp.tuplelist(valid_E_list); final_data['d'] = gp.tupledict(valid_d_dict)
    final_data['c'] = gp.tupledict(data['c']); raw_a = gp.tupledict(data['a']); raw_w = gp.tupledict(data['w'])
    
    default_w_value = 1.0; temp_w = gp.tupledict()
    for n_node in final_data['N']:
        for g_good in final_data['G']: temp_w[n_node, g_good] = raw_w.get((n_node, g_good), default_w_value)
    final_data['w'] = temp_w
    
    temp_a = gp.tupledict()
    for n_node in final_data['N']:
        for g_good in final_data['G']: temp_a[n_node, g_good] = raw_a.get((n_node, g_good), 0.0)
    final_data['a'] = temp_a
    
    # Print distance statistics
    print(f"✅ 数据解析完成:")
    print(f"  - 节点数量: {len(final_data['N'])}")
    print(f"  - 边数量: {len(final_data['E'])}")
    print(f"  - 距离矩阵大小: {len(final_data['d'])}")
    if distance_mode == 1:
        print(f"  - 注意: 使用完全图，任意两个节点之间都可以直接连通")
    
    return final_data

def _calculate_euclidean_distance(px1: float, py1: float, px2: float, py2: float) -> float:
    """计算两点间的欧几里得距离"""
    return math.sqrt((px2 - px1)**2 + (py2 - py1)**2)

def _update_edge_distances_by_coordinates(data: dict):
    """仅根据坐标更新E中已有边的距离，不改变E结构"""
    updated = 0
    for (i, j) in data['E']:
        if i in data['px'] and j in data['px']:
            px1, py1 = data['px'][i], data['py'][i]
            px2, py2 = data['px'][j], data['py'][j]
            dist = _calculate_euclidean_distance(px1, py1, px2, py2)
            data['d'][(i, j)] = dist
            updated += 1
    print(f"  ➤ 已用坐标更新 {updated} 条边的距离 (仅E中存在的边)")

# =============================================================================
# Visualization Function
# =============================================================================
def visualize_solution(data, model, filename="GRB_viz.png"):
    """
    可视化最优解，包括节点、车辆路径、货物流动、中断边等
    """
    if model is None or model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED] or model.SolCount == 0:
        print("❌ 无可视化内容：模型未求得可行解")
        return
    
    print("🎨 开始生成可视化图像...")
    
    # 提取数据
    N_nodes = data['N']
    E_edges = data['E']
    K_vehicles = data['K']
    L_start_node = data['L']
    px, py = data['px'], data['py']
    a_amount = data['a']
    G_goods = data['G']
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_aspect('equal')
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 绘制所有可用边（灰色细线）
    for i, j in E_edges:
        x1, y1 = px[i], py[i]
        x2, y2 = px[j], py[j]
        ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
    
    # 2. 找出被中断的边
    disrupted_edges = []
    for v in model.getVars():
        if v.VarName.startswith('u[') and v.X > 0.5:
            name = v.VarName
            edge_str = name[name.find('[')+1:name.find(']')]
            edge_tuple = tuple(int(x) for x in edge_str.replace('(','').replace(')','').split(','))
            disrupted_edges.append(edge_tuple)
    
    # 3. 绘制中断边（红色粗线）
    for edge in disrupted_edges:
        i, j = edge
        if i in px and j in px:
            x1, y1 = px[i], py[i]
            x2, y2 = px[j], py[j]
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=4, alpha=0.8, zorder=3)
            # 在边上标记"X"
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, '✕', fontsize=16, color='red', ha='center', va='center', 
                   weight='bold', zorder=4)
    
    # 4. 绘制车辆路径
    vehicle_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow']
    
    for k_idx, k in enumerate(K_vehicles):
        color = vehicle_colors[k_idx % len(vehicle_colors)]
        
        # 重构车辆路径
        path_edges = []
        current_node = L_start_node[k]
        visited_nodes = {current_node}
        
        try:
            for _ in range(len(N_nodes) + 1):
                next_edge = None
                for i_edge, j_edge in E_edges:
                    if abs(i_edge - current_node) < 1e-6:
                        # 查找变量名
                        x_var_name = f'x[{k},{i_edge},{j_edge}]'
                        for v in model.getVars():
                            if v.VarName == x_var_name and v.X > 0.5:
                                next_edge = (i_edge, j_edge)
                                break
                        if next_edge:
                            break
                
                if next_edge:
                    path_edges.append(next_edge)
                    current_node = next_edge[1]
                    if current_node == L_start_node[k]:  # 回到起点
                        break
                    if current_node in visited_nodes:  # 避免无限循环
                        break
                    visited_nodes.add(current_node)
                else:
                    break
            
            # 绘制路径
            for i, j in path_edges:
                x1, y1 = px[i], py[i]
                x2, y2 = px[j], py[j]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.8),
                           zorder=5)
                
        except Exception as e:
            print(f"  警告：车辆{k}路径重构失败: {e}")
    
    # 5. 绘制节点
    for node in N_nodes:
        x, y = px[node], py[node]
        
        # 判断节点类型
        is_depot = node in L_start_node.values()
        is_supply = any(a_amount[node, g] > 0 for g in G_goods)
        is_demand = any(a_amount[node, g] < 0 for g in G_goods)
        
        # 设置节点样式
        if is_depot:
            # 仓库节点 - 大的正方形
            square = patches.Rectangle((x-4, y-4), 8, 8, linewidth=2, 
                                     edgecolor='black', facecolor='gold', zorder=6)
            ax.add_patch(square)
            ax.text(x, y-12, f'仓库{node}', fontsize=9, ha='center', va='top', weight='bold')
        elif is_supply:
            # 供应节点 - 绿色圆圈
            circle = patches.Circle((x, y), 6, linewidth=2, 
                                  edgecolor='darkgreen', facecolor='lightgreen', zorder=6)
            ax.add_patch(circle)
            ax.text(x, y-10, f'供应{node}', fontsize=8, ha='center', va='top')
        elif is_demand:
            # 需求节点 - 红色圆圈
            circle = patches.Circle((x, y), 6, linewidth=2, 
                                  edgecolor='darkred', facecolor='lightcoral', zorder=6)
            ax.add_patch(circle)
            ax.text(x, y-10, f'需求{node}', fontsize=8, ha='center', va='top')
        else:
            # 中转节点 - 蓝色圆圈
            circle = patches.Circle((x, y), 4, linewidth=1, 
                                  edgecolor='darkblue', facecolor='lightblue', zorder=6)
            ax.add_patch(circle)
            ax.text(x, y-8, f'{node}', fontsize=8, ha='center', va='top')
        
        # 在节点内显示编号
        ax.text(x, y, str(node), fontsize=10, ha='center', va='center', 
               weight='bold', color='white', zorder=7)
    
    # 6. 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='lightgray', lw=1, label='可用边'),
        plt.Line2D([0], [0], color='red', lw=4, label='中断边'),
        patches.Rectangle((0, 0), 1, 1, facecolor='gold', edgecolor='black', label='仓库节点'),
        patches.Circle((0, 0), 1, facecolor='lightgreen', edgecolor='darkgreen', label='供应节点'),
        patches.Circle((0, 0), 1, facecolor='lightcoral', edgecolor='darkred', label='需求节点'),
        patches.Circle((0, 0), 1, facecolor='lightblue', edgecolor='darkblue', label='中转节点'),
    ]
    
    # 添加车辆路径图例
    for k_idx, k in enumerate(K_vehicles):
        color = vehicle_colors[k_idx % len(vehicle_colors)]
        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=3, label=f'车辆{k}路径')
        )
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # 7. 设置标题和标签
    ax.set_title('车辆路径优化问题可视化结果', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('X 坐标', fontsize=12)
    ax.set_ylabel('Y 坐标', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 8. 调整布局和保存
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ 可视化图像已保存为: {filename}")
    except Exception as e:
        print(f"❌ 保存图像失败: {e}")
    
    # 显示图像（可选）
    # plt.show()
    plt.close()

# =============================================================================
# Model Building and Solving Function (Enhanced with Visualization)
# =============================================================================
def solve_vehicle_routing_strict(data, gurobi_time_limit=None):
    try:
        # --- 1. Extract Data ---
        N_nodes = data['N']
        E_edges = data['E']
        d_dist = data['d']
        K_vehicles = data['K']
        v_speed = data['v']
        L_start_node = data['L']
        G_goods = data['G']
        c_capacity = data['c']
        a_amount = data['a']
        w_weight = data['w']
        # [NEW] Use H and U from parsed data
        H_time_limit = data['H']
        U_disrupt_count = data['U']

        # --- [NEW] Set Gurobi TimeLimit if provided ---
        m = gp.Model("VehicleRouting_StrictAMPL")
        if gurobi_time_limit is not None:
            print(f"[Gurobi] 设置最大求解时间: {gurobi_time_limit} 秒")
            m.Params.TimeLimit = gurobi_time_limit

        # Big-M values
        M1 = 9999.0
        M_Con8 = 999.0
        M_Con9a = 999.0
        M_Con10 = 99.0
        M_Con11a = 9999.0
        M_Con12 = 9999.0

        # --- 2. Helper Index Sets ---
        DemandIndices = gp.tuplelist([
            (k, i, g) for k in K_vehicles for i in N_nodes for g in G_goods if a_amount[i, g] < 0
        ])
        SupplyIndices = gp.tuplelist([
            (k, i, g) for k in K_vehicles for i in N_nodes for g in G_goods if a_amount[i, g] > 0
        ])
        DemandTimeIndices = gp.tuplelist([
            (k, i_demand, g, j_node_dest)
            for k, i_demand, g in DemandIndices
            for j_node_dest in N_nodes
        ])

        # --- 4. Define Variables ---
        print("Defining variables...")
        x = m.addVars(K_vehicles, E_edges, vtype=GRB.BINARY, name="x")
        t = m.addVars(K_vehicles, N_nodes, vtype=GRB.CONTINUOUS, lb=0.0, name="t")
        y = m.addVars(K_vehicles, E_edges, G_goods, vtype=GRB.CONTINUOUS, lb=0.0, name="y")
        z = m.addVars(DemandIndices, vtype=GRB.CONTINUOUS, lb=0.0, name="z")
        zt = m.addVars(DemandTimeIndices, vtype=GRB.CONTINUOUS, lb=0.0, name="zt")
        b = m.addVars(SupplyIndices, vtype=GRB.CONTINUOUS, lb=0.0, name="b")
        u = m.addVars(E_edges, vtype=GRB.BINARY, name="u")
        q = m.addVars(K_vehicles, N_nodes, vtype=GRB.BINARY, name="q")
        h = m.addVars(DemandIndices, vtype=GRB.BINARY, name="h")
        o = m.addVars(DemandIndices, vtype=GRB.CONTINUOUS, lb=0.0, name="o")

        # --- 5. Define Constraints ---
        print("Adding constraints...")

        # (0) Routing (Unchanged)
        for k_idx in K_vehicles:
            m.addConstr(gp.quicksum(x[k_idx, i_edge, j_edge] for i_edge, j_edge in E_edges if i_edge == L_start_node[k_idx]) == 1)
            m.addConstr(gp.quicksum(x[k_idx, i_edge, j_edge] for i_edge, j_edge in E_edges if j_edge == L_start_node[k_idx]) == 1)
            for i_idx in N_nodes:
                if i_idx != L_start_node[k_idx]:
                    m.addConstr(gp.quicksum(x[k_idx, j_prev, i_curr] for j_prev, i_curr in E_edges if i_curr == i_idx) ==
                                gp.quicksum(x[k_idx, i_curr, j_next] for i_curr, j_next in E_edges if i_curr == i_idx))
        for k_idx in K_vehicles:
            for i_idx in N_nodes:
                m.addConstr(gp.quicksum(x[k_idx, j_prev, i_curr] for j_prev, i_curr in E_edges if i_curr == i_idx) <= 1)

        # (1) - (8) (Unchanged)
        # (1) Time Calculation
        for k_idx in K_vehicles:
            for i_edge, j_edge in E_edges:
                travel_time = 60.0 * d_dist[i_edge, j_edge] / v_speed[k_idx]
                if i_edge == L_start_node[k_idx]:
                    m.addConstr(t[k_idx, j_edge] >= travel_time - M1 * (1 - x[k_idx, i_edge, j_edge]))
                    m.addConstr(t[k_idx, j_edge] <= travel_time + M1 * (1 - x[k_idx, i_edge, j_edge]))
                else:
                    m.addConstr(t[k_idx, j_edge] >= t[k_idx, i_edge] + travel_time - M1 * (1 - x[k_idx, i_edge, j_edge]))
                    m.addConstr(t[k_idx, j_edge] <= t[k_idx, i_edge] + travel_time + M1 * (1 - x[k_idx, i_edge, j_edge]))
        # (2) y <= c*x
        for k_idx in K_vehicles:
            for i_edge, j_edge in E_edges:
                for g_idx in G_goods:
                    m.addConstr(y[k_idx, i_edge, j_edge, g_idx] <= c_capacity[k_idx, g_idx] * x[k_idx, i_edge, j_edge])
        # (3) Vehicle Capacity
        for k_idx in K_vehicles:
            for i_edge, j_edge in E_edges:
                m.addConstr(gp.quicksum(y[k_idx, i_edge, j_edge, g_idx] / c_capacity[k_idx, g_idx] for g_idx in G_goods) <= 1)
        # (4) Flow Balance at Supply Nodes
        for k_idx, i_idx, g_idx in SupplyIndices:
            sum_outgoing_y = gp.quicksum(y[k_idx, i, j, g_idx] for i,j in E_edges.select(i_idx, '*'))
            sum_incoming_y = gp.quicksum(y[k_idx, j, i, g_idx] for j,i in E_edges.select('*', i_idx) if i_idx != L_start_node[k_idx])
            m.addConstr(sum_outgoing_y - sum_incoming_y == b[k_idx, i_idx, g_idx])
        for i_idx in N_nodes:
            for g_idx in G_goods:
                if a_amount[i_idx, g_idx] > 0:
                    m.addConstr(gp.quicksum(b[k,i_idx,g_idx] for k,i,g in SupplyIndices if i==i_idx and g==g_idx) <= a_amount[i_idx, g_idx])
        # (5) Flow Balance at Intermediate Nodes
        for k_idx in K_vehicles:
            for i_idx in N_nodes:
                for g_idx in G_goods:
                    if a_amount[i_idx, g_idx] == 0:
                        lhs = gp.quicksum(y[k_idx,j,i,g_idx] for j,i in E_edges.select('*', i_idx) if i_idx != L_start_node[k_idx])
                        rhs = gp.quicksum(y[k_idx,i,j,g_idx] for i,j in E_edges.select(i_idx, '*'))
                        m.addConstr(lhs == rhs)
        # (6) Flow Balance at Demand Nodes
        for k_idx, i_idx, g_idx in DemandIndices:
            sum_incoming_y = gp.quicksum(y[k_idx, j, i, g_idx] for j, i in E_edges.select('*', i_idx))
            sum_outgoing_y = gp.quicksum(y[k_idx, i, j, g_idx] for i, j in E_edges.select(i_idx, '*') if i_idx != L_start_node[k_idx])
            m.addConstr(sum_incoming_y - sum_outgoing_y == z[k_idx, i_idx, g_idx])
        for i_idx in N_nodes:
            for g_idx in G_goods:
                if a_amount[i_idx, g_idx] < 0:
                    m.addConstr(gp.quicksum(z[k,i_idx,g_idx] for k,i,g in DemandIndices if i==i_idx and g==g_idx) <= -a_amount[i_idx, g_idx])
        for k_idx in K_vehicles:
            sum_b = gp.quicksum(b[k,i,g] for k,i,g in SupplyIndices if k==k_idx)
            sum_z = gp.quicksum(z[k,i,g] for k,i,g in DemandIndices if k==k_idx)
            m.addConstr(sum_b == sum_z)
        # (7) Calculate zt
        for k_idx_zt, i_idx_demand_node, g_idx_zt in DemandIndices:
            for i_edge, j_edge in E_edges:
                if not (k_idx_zt, i_idx_demand_node, g_idx_zt, j_edge) in zt: continue
                travel_time_factor = 60.0 * z[k_idx_zt, i_idx_demand_node, g_idx_zt] * d_dist[i_edge, j_edge] / v_speed[k_idx_zt]
                if i_edge == L_start_node[k_idx_zt]:
                    m.addConstr(zt[k_idx_zt, i_idx_demand_node, g_idx_zt, j_edge] >= travel_time_factor - M1 * (1 - x[k_idx_zt, i_edge, j_edge]))
                    m.addConstr(zt[k_idx_zt, i_idx_demand_node, g_idx_zt, j_edge] <= travel_time_factor + M1 * (1 - x[k_idx_zt, i_edge, j_edge]))
                else:
                    if not (k_idx_zt, i_idx_demand_node, g_idx_zt, i_edge) in zt: continue
                    m.addConstr(zt[k_idx_zt, i_idx_demand_node, g_idx_zt, j_edge] >= zt[k_idx_zt, i_idx_demand_node, g_idx_zt, i_edge] + travel_time_factor - M1 * (1 - x[k_idx_zt, i_edge, j_edge]))
                    m.addConstr(zt[k_idx_zt, i_idx_demand_node, g_idx_zt, j_edge] <= zt[k_idx_zt, i_idx_demand_node, g_idx_zt, i_edge] + travel_time_factor + M1 * (1 - x[k_idx_zt, i_edge, j_edge]))
        # (8) Link zt with z
        for k, ii, g, j in DemandTimeIndices:
             m.addConstr(zt[k, ii, g, j] <= M_Con8 * z[k, ii, g])

        # (9) Edge Disruption
        for i_edge, j_edge in E_edges:
            for i1_edge, j1_edge in E_edges:
                if (i_edge, j_edge) != (i1_edge, j1_edge) :
                    sum_y_ij = gp.quicksum(y[k_idx, i_edge, j_edge, g_idx] for k_idx in K_vehicles for g_idx in G_goods)
                    sum_y_i1j1 = gp.quicksum(y[k_idx, i1_edge, j1_edge, g_idx] for k_idx in K_vehicles for g_idx in G_goods)
                    m.addConstr(sum_y_ij >= sum_y_i1j1 - M_Con9a * (1 - u[i_edge, j_edge] + u[i1_edge, j1_edge]))
        # [NEW] Use U_disrupt_count instead of hardcoded 1
        m.addConstr(gp.quicksum(u[i_edge,j_edge] for i_edge,j_edge in E_edges) == U_disrupt_count, name="C9b")


        # (10) Track Path Disruption Status (q)
        for k_idx in K_vehicles:
            for i_edge, j_edge in E_edges:
                if i_edge == L_start_node[k_idx]:
                    m.addConstr(q[k_idx, j_edge] >= u[i_edge, j_edge] - M_Con10 * (1 - x[k_idx, i_edge, j_edge]))
                else:
                    m.addConstr(q[k_idx, j_edge] >= q[k_idx, i_edge] - M_Con10 * (1 - x[k_idx, i_edge, j_edge]))
                    m.addConstr(q[k_idx, j_edge] >= u[i_edge, j_edge] - M_Con10 * (1 - x[k_idx, i_edge, j_edge]))

        # (11) Determine Delivery Validity (h)
        for k_idx, i_idx, g_idx in DemandIndices:
            if not (k_idx, i_idx, g_idx, i_idx) in zt: continue
            # [NEW] Use H_time_limit from data
            m.addConstr(zt[k_idx, i_idx, g_idx, i_idx] <= z[k_idx, i_idx, g_idx] * H_time_limit + M_Con11a * (1 - h[k_idx, i_idx, g_idx]))
            m.addConstr(h[k_idx, i_idx, g_idx] <= 1 - q[k_idx, i_idx])

        # (12) Calculate Objective Component (o)
        for k_idx, i_idx, g_idx in DemandIndices:
            if not (k_idx, i_idx, g_idx, i_idx) in zt: continue
            # [NEW] Use H_time_limit from data
            m.addConstr(o[k_idx, i_idx, g_idx] <= z[k_idx, i_idx, g_idx] * H_time_limit - zt[k_idx, i_idx, g_idx, i_idx] + M_Con12 * (1 - h[k_idx, i_idx, g_idx]))
            m.addConstr(o[k_idx, i_idx, g_idx] >= z[k_idx, i_idx, g_idx] * H_time_limit - zt[k_idx, i_idx, g_idx, i_idx] - M_Con12 * (1 - h[k_idx, i_idx, g_idx]))
            m.addConstr(o[k_idx, i_idx, g_idx] <= M_Con12 * h[k_idx, i_idx, g_idx])
        
        # --- 6. Define Objective Function ---
        print("Setting objective...")
        objective = gp.quicksum(o[k, i, g] * w_weight[i, g] for k,i,g in DemandIndices)
        m.setObjective(objective, GRB.MAXIMIZE)

        # --- 7. Optimize Model ---
        print("Optimizing model...")
        m.optimize()

        # --- 8. Process Results (Enhanced with corrected objective contributions) ---
        print("\n--- Optimization Results ---")
        if m.Status == GRB.OPTIMAL or (m.Status == GRB.TIME_LIMIT and m.SolCount > 0) or (m.Status == GRB.INTERRUPTED and m.SolCount > 0):
            print(f"Objective Value: {m.ObjVal:.4f}")

            # Total Edge Flows
            print("\nTotal Edge Flows (Sorted Descending):")
            edge_flows = {}
            for i_edge, j_edge in E_edges:
                total_flow = gp.quicksum(y[k, i_edge, j_edge, g].X for k in K_vehicles for g in G_goods).getValue()
                if total_flow > 1e-6:
                    edge_flows[(i_edge, j_edge)] = total_flow
            sorted_flows = sorted(edge_flows.items(), key=lambda item: item[1], reverse=True)
            if not sorted_flows: print("  No goods were transported.")
            else:
                for edge, flow in sorted_flows:
                    print(f"  Edge {edge}: Flow = {flow:.2f}")

            # Vehicle Paths
            print("\nVehicle Paths (x=1):")
            for k_path in K_vehicles:
                print(f"  Vehicle {k_path}:")
                path_edges_k = []; current_n = L_start_node[k_path]; visited_n = {current_n}
                try:
                    for _ in range(len(N_nodes) + 1):
                        next_edge = None
                        for i_edge_p, j_edge_p in E_edges:
                             if abs(i_edge_p - current_n) < 1e-6 and x[k_path, i_edge_p, j_edge_p].X > 0.5:
                                 next_edge = (i_edge_p, j_edge_p)
                                 break
                        if next_edge:
                            path_edges_k.append(next_edge); current_n = next_edge[1]
                            if current_n == L_start_node[k_path]: break
                            if current_n in visited_n: break
                            visited_n.add(current_n)
                        else: break
                    if path_edges_k: print(f"    Path: {L_start_node[k_path]} -> {' -> '.join(map(str, [j for i,j in path_edges_k]))}")
                    else: print(f"    Path: No edges used from depot {L_start_node[k_path]}.")
                except Exception as path_e: print(f"    Error reconstructing path for {k_path}: {path_e}")

            # Vehicle Task Summary
            print("\nVehicle Task Summary:")
            all_tasks_for_file = []
            for k_vehicle in K_vehicles:
                print(f"  --- Vehicle {k_vehicle} ---")
                has_task = False
                pickups = {}
                for k, i, g in SupplyIndices:
                    if k == k_vehicle and b[k, i, g].X > 1e-6:
                        if g not in pickups: pickups[g] = []
                        pickups[g].append([i, b[k, i, g].X])
                        print(f"    Pickup:  {b[k, i, g].X:>6.2f} of Good {g} at Node {i}")
                        has_task = True
                deliveries = {}
                for k, i, g in DemandIndices:
                    if k == k_vehicle and z[k, i, g].X > 1e-6:
                        if g not in deliveries: deliveries[g] = []
                        deliveries[g].append([i, z[k, i, g].X])
                        print(f"    Deliver: {z[k, i, g].X:>6.2f} of Good {g} to Node {i}")
                        has_task = True
                if not has_task:
                    print("    No tasks assigned.")

                # Match pickups to deliveries for file export
                for g_good in G_goods:
                    if g_good in pickups and g_good in deliveries:
                        pickup_list = sorted(pickups[g_good])
                        delivery_list = sorted(deliveries[g_good])
                        while pickup_list and delivery_list:
                            pickup_node, pickup_qty = pickup_list[0]
                            delivery_node, delivery_qty = delivery_list[0]
                            matched_qty = min(pickup_qty, delivery_qty)
                            if matched_qty > 1e-6:
                                all_tasks_for_file.append((k_vehicle, pickup_node, delivery_node, g_good, matched_qty))
                            pickup_list[0][1] -= matched_qty
                            delivery_list[0][1] -= matched_qty
                            if pickup_list[0][1] < 1e-6: pickup_list.pop(0)
                            if delivery_list[0][1] < 1e-6: delivery_list.pop(0)

            # Save tasks.dat
            try:
                with open("tasksCoord.dat", "w") as f:
                    f.write("# Task parameters generated from Gurobi solution\n")
                    f.write("# Format: vehicle_id pickup_node delivery_node good_id quantity\n")
                    f.write("param tasks: V P D G Q :=\n")
                    if not all_tasks_for_file:
                        f.write("# No tasks were generated.\n")
                    else:
                        for task in all_tasks_for_file:
                            f.write(f"{task[0]:<2} {task[1]:<2} {task[2]:<2} {task[3]:<2} {task[4]:<5.1f}\n")
                    f.write(";\n")
                print("\nSuccessfully saved pickup-delivery tasks to tasks.dat")
            except Exception as e:
                print(f"\nError writing to tasks.dat: {e}")

            # ========== 修正版：打印中断边和每个车辆任务的目标函数贡献值 ==========
            try:
                print("\n--- 中断边 (u=1) ---")
                disrupted_edges = []
                for v in m.getVars():
                    if v.VarName.startswith('u[') and v.X > 0.5:
                        name = v.VarName
                        edge_str = name[name.find('[')+1:name.find(']')]
                        edge_tuple = tuple(int(x) for x in edge_str.replace('(','').replace(')','').split(','))
                        disrupted_edges.append(edge_tuple)
                if disrupted_edges:
                    for e in disrupted_edges:
                        print(f"  被中断边: {e}")
                else:
                    print("  无中断边。")
            except Exception as e:
                print(f"  [打印中断边出错]: {e}")

            try:
                print("\n--- 车辆任务目标函数贡献值 (修正版) ---")
                # 直接使用模型中的 o 变量值，而不是自己计算
                total_contribution = 0.0
                task_contributions = []
                
                for v in m.getVars():
                    if v.VarName.startswith('o[') and v.X > 1e-6:
                        # o[k,i,g]
                        o_val = v.X
                        o_name = v.VarName
                        idx_str = o_name[o_name.find('[')+1:o_name.find(']')]
                        idx_tuple = tuple(int(x) for x in idx_str.split(','))
                        k, i, g = idx_tuple
                        
                        # 查找对应的 w[i,g]
                        w_val = w_weight.get((i, g), 1.0)
                        
                        # 查找对应的 z[k,i,g] 和 h[k,i,g]
                        z_val = 0.0
                        h_val = 0.0
                        for v2 in m.getVars():
                            if v2.VarName == f'z[{k},{i},{g}]':
                                z_val = v2.X
                            elif v2.VarName == f'h[{k},{i},{g}]':
                                h_val = v2.X
                        
                        # 计算加权贡献值
                        weighted_contribution = o_val * w_val
                        total_contribution += weighted_contribution
                        
                        task_contributions.append({
                            'vehicle': k,
                            'delivery_node': i,
                            'good': g,
                            'delivery_qty': z_val,
                            'o_value': o_val,
                            'weight': w_val,
                            'contribution': weighted_contribution,
                            'is_valid': h_val > 0.5
                        })
                
                # 按贡献值降序排序
                task_contributions.sort(key=lambda x: x['contribution'], reverse=True)
                
                print(f"  总目标函数值验证: {total_contribution:.4f} (模型目标值: {m.ObjVal:.4f})")
                print("  各任务详细贡献:")
                print("  车辆 | 配送节点 | 商品 | 数量  | o值   | 权重 | 贡献值  | 有效性")
                print("  -----|----------|------|-------|-------|------|---------|-------")
                
                for task in task_contributions:
                    validity = "✓有效" if task['is_valid'] else "✗无效"
                    print(f"  {task['vehicle']:^4} | {task['delivery_node']:^8} | {task['good']:^4} | "
                          f"{task['delivery_qty']:^5.1f} | {task['o_value']:^5.1f} | {task['weight']:^4.1f} | "
                          f"{task['contribution']:^7.2f} | {validity}")
                    
            except Exception as e:
                print(f"  [打印任务贡献值出错]: {e}")
            # =========================================================
            
            # ========== NEW: Generate Visualization ==========
            try:
                visualize_solution(data, m, "GRB_viz.png")
            except Exception as e:
                print(f"❌ 可视化生成失败: {e}")
            # =================================================

        elif m.Status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS..."); m.computeIIS(); m.write("model_iis.ilp"); print("IIS written to model_iis.ilp")
        elif m.Status == GRB.UNBOUNDED:
            print("Model is unbounded.")
        else:
            print(f"Optimization finished with status code {m.Status}")
        return m
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return None
    except Exception as e:
        import traceback
        print(f"Unexpected error in solve_vehicle_routing_strict: {e}")
        traceback.print_exc()
        return None

# =============================================================================
# User Interface Functions
# =============================================================================
def get_user_choice():
    """获取用户的距离计算方式选择"""
    print("\n" + "="*70)
    print("🚚 车辆路径优化 MILP 模型 - 距离计算方式选择")
    print("="*70)
    print("请选择距离计算方式:")
    print("  0 - 使用数据文件E中指定的边距离 (稀疏图)")
    print("  1 - 基于节点坐标计算欧几里得距离 (完全图)")
    print("-"*70)
    print("说明:")
    print("  • 选项0: 保持原有稀疏图结构，只有E中指定的边可以通行")
    print("  • 选项1: 构建完全图，任意两个节点之间都可以直接连通")
    print("-"*70)
    
    while True:
        try:
            choice = input("请输入选择 (0 或 1): ").strip()
            if choice in ['0', '1']:
                return int(choice)
            else:
                print("❌ 输入无效，请输入 0 或 1")
        except (ValueError, KeyboardInterrupt):
            print("❌ 输入无效，请输入 0 或 1")

def print_distance_summary(data: dict, distance_mode: int):
    """打印距离信息摘要"""
    print(f"\n📊 距离数据摘要:")
    print(f"  - 距离计算模式: {'坐标欧几里得距离' if distance_mode == 1 else 'E中指定距离'}")
    print(f"  - 节点数量: {len(data['N'])}")
    print(f"  - 边数量: {len(data['E'])}")
    print(f"  - 距离矩阵大小: {len(data['d'])}")
    
    if distance_mode == 1:
        node_count = len(data['N'])
        max_possible_edges = node_count * (node_count - 1)
        print(f"  - 理论最大边数 (完全图): {max_possible_edges}")
        print(f"  - 图类型: 完全有向图")
    else:
        print(f"  - 图类型: 稀疏有向图 (由E定义)")
    
    # 显示一些示例距离
    if data['d']:
        print(f"  - 距离范围示例:")
        distances = list(data['d'].values())
        min_dist = min(distances)
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        print(f"    * 最小距离: {min_dist:.2f}")
        print(f"    * 最大距离: {max_dist:.2f}")
        print(f"    * 平均距离: {avg_dist:.2f}")

# =============================================================================
# File Reading Functions
# =============================================================================
def read_data_file(filename: str) -> Optional[str]:
    """读取数据文件内容"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ 成功读取数据文件: {filename}")
        return content
    except FileNotFoundError:
        print(f"❌ 错误: 找不到数据文件 '{filename}'")
        print(f"   请确保文件存在于当前目录中")
        return None
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return None

def get_data_file_path():
    """获取数据文件路径"""
    print("\n" + "="*50)
    print("📁 数据文件选择")
    print("="*50)

    default_file = "problem_data_231122.dat"

    print(f"默认数据文件: {default_file}")
    print("请选择:")
    print("  1 - 使用默认文件 (problem_data_231122.dat)")
    print("  2 - 指定其他文件")
    print("-"*50)
    
    while True:
        try:
            choice = input("请输入选择 (1 或 2): ").strip()
            if choice == '1':
                return default_file
            elif choice == '2':
                custom_file = input("请输入数据文件路径: ").strip()
                if custom_file:
                    return custom_file
                else:
                    print("❌ 文件路径不能为空")
            else:
                print("❌ 输入无效，请输入 1 或 2")
        except (ValueError, KeyboardInterrupt):
            print("❌ 输入无效，请输入 1 或 2")



# =============================================================================
#  Independent Auditor (独立审计器)
# =============================================================================
def perform_audit(model, data, real_U):
    """
    对求解结果进行独立审计：
    1. 根据解出的流量 (y变量)，重新计算全图流量。
    2. 找出流量最大的 real_U 条边并断掉。
    3. 重新核算每辆车的路径和得分，计算真实目标值。
    """
    if model is None or model.SolCount == 0: return

    print("\n" + "="*60)
    print(f"🕵️  开始进行 [独立流量审计] (审计设定 U={real_U})...")
    print("="*60)

    # Unpack data
    E = data['E']
    K = data['K']
    L = data['L']
    H = data['H']
    w = data['w']
    
    # ---------------------------------------------------------
    # 1. 智能提取变量 (不再依赖字符串拼接猜测)
    # ---------------------------------------------------------
    y_vals = {}
    x_active_edges = {} # 存储为字典: k -> set of (i,j)
    z_tasks = []        # 存储为列表: (k, i, g, amount)
    zt_vals = {}
    
    for v in model.getVars():
        if v.X <= 1e-6: continue # 忽略 0 值
        
        # 提取 y (流量)
        if v.VarName.startswith('y['):
            y_vals[v.VarName] = v.X
            
        # 提取 x (路径): 解析 x[k,i,j]
        elif v.VarName.startswith('x['):
            # 格式: x[1,0,5]
            content = v.VarName.split('[')[1].split(']')[0]
            k_idx, i_idx, j_idx = map(int, content.split(','))
            if k_idx not in x_active_edges: x_active_edges[k_idx] = set()
            x_active_edges[k_idx].add((i_idx, j_idx))
            
        # 提取 z (卸货任务): 解析 z[k,i,g]
        elif v.VarName.startswith('z['):
            # 格式: z[1,11,1]
            content = v.VarName.split('[')[1].split(']')[0]
            k_idx, i_idx, g_idx = map(int, content.split(','))
            z_tasks.append((k_idx, i_idx, g_idx, v.X))
            
        # 提取 zt (时间成本)
        elif v.VarName.startswith('zt['):
            zt_vals[v.VarName] = v.X

    # ---------------------------------------------------------
    # 2. 计算全图流量 & 识别 Top-U 中断边
    # ---------------------------------------------------------
    edge_flows = {}
    for i, j in E:
        flow = 0.0
        # 这种遍历方式比字符串拼接更慢但更稳，但鉴于y变量较多，
        # 我们这里为了性能，还是遍历 y_vals 字典
        pass
    
    # 更高效的流量统计：直接遍历提取出的 y_vals
    for name, val in y_vals.items():
        # name: y[k,i,j,g]
        content = name.split('[')[1].split(']')[0]
        _, i, j, _ = map(int, content.split(','))
        edge_flows[(i,j)] = edge_flows.get((i,j), 0.0) + val

    # 排序取 Top-U
    sorted_edges = sorted(edge_flows.items(), key=lambda x: x[1], reverse=True)
    disrupted_edges = set([e for e, f in sorted_edges[:real_U]])

    print(f"📉 [审计结果] 识别出的高流量中断边 (Top-{real_U}):")
    if not disrupted_edges:
        print("   (无流量或 U=0)")
    for e in disrupted_edges:
        print(f"   ❌ {e} (流量: {edge_flows.get(e, 0):.2f})")

    # ---------------------------------------------------------
    # 3. 核算真实得分
    # ---------------------------------------------------------
    nominal_obj = model.ObjVal
    actual_obj = 0.0
    
    print("\n📋 详细审计报告:")
    print("  车辆 | 路径状态 | 任务节点 | 商品 | 名义得分 | 实际得分 | 状态")
    print("  " + "-"*65)

    for k in K:
        # --- 重构路径 ---
        path_edges = []
        curr = L[k]
        path_broken = False
        broken_at = None
        
        # 获取该车辆的所有激活边
        active_edges = x_active_edges.get(k, set())
        
        # 链式追踪
        visited = {curr}
        # 最多只有 |N| 步，防止死循环
        for _ in range(len(data['N']) + 2):
            next_node = None
            # 在 active_edges 中找从 curr 出发的边
            for u, v in active_edges:
                if u == curr:
                    next_node = v
                    edge = (u, v)
                    
                    # 检查中断
                    if edge in disrupted_edges and not path_broken:
                        path_broken = True
                        broken_at = edge
                    
                    path_edges.append(edge)
                    break
            
            if next_node is None: break # 终点或断路
            curr = next_node
            if curr in visited: break # 环路保护
            visited.add(curr)
            if curr == L[k]: break # 回到车库
        
        # --- 确定可达性 ---
        # 逻辑：在断点之前的所有节点是“可达”的
        reachable_nodes = set()
        valid_so_far = True
        reachable_nodes.add(L[k]) # 起点总是可达
        
        for u, v in path_edges:
            if (u, v) in disrupted_edges:
                valid_so_far = False
            
            if valid_so_far:
                reachable_nodes.add(v)

        # --- 检查该车辆的任务 ---
        # 从 z_tasks 列表中筛选出属于当前车辆 k 的任务
        veh_tasks_list = [t for t in z_tasks if t[0] == k]
        
        status_str = "正常"
        if path_broken: status_str = f"断于{broken_at}"
        
        if not veh_tasks_list:
            print(f"  {k:^4} | {status_str:^8} | {'-':^8} | {'-':^4} | {'-':^8} | {'-':^8} | 无任务")
            continue

        for _, i_node, g_good, z_val in veh_tasks_list:
            # 获取 zt 值 (时间成本)
            zt_val = 0.0
            # 这里可以用字符串拼接，因为我们确定变量存在
            zt_name = f"zt[{k},{i_node},{g_good},{i_node}]"
            if zt_name in zt_vals: 
                zt_val = zt_vals[zt_name]
            
            # 计算名义得分 (H*z - zt) * w
            nominal_score = (z_val * H - zt_val) * w.get((i_node, g_good), 1.0)
            
            # 计算实际得分
            if i_node in reachable_nodes:
                actual_score = nominal_score
                check_mark = "✅有效"
            else:
                actual_score = 0.0
                check_mark = "❌失效"
            
            actual_obj += actual_score
            print(f"  {k:^4} | {status_str:^8} | {i_node:^8} | {g_good:^4} | {nominal_score:8.2f} | {actual_score:8.2f} | {check_mark}")

    print("  " + "-"*65)
    print(f"📈 最终结论:")
    print(f"   名义目标值 (Solver Guess) : {nominal_obj:.2f}")
    print(f"   实际目标值 (Audit Result) : {actual_obj:.2f}")
    loss = nominal_obj - actual_obj
    loss_pct = (loss / nominal_obj * 100) if nominal_obj > 1e-6 else 0.0
    print(f"   鲁棒性损失 (Gap)          : {loss:.2f} ({loss_pct:.1f}%)")
    print("="*60)
    
    # 重新生成可视化
    visualize_solution(data, model, "Audit_Viz.png", disrupted_edges_override=list(disrupted_edges))
    
# =============================================================================
# Main Execution (Updated Logic)
# =============================================================================
if __name__ == "__main__":
    # 1. Read Data
    fpath = get_data_file_path()
    with open(fpath, 'r', encoding='utf-8') as f: content = f.read()
    
    # 2. Parse
    dist_mode = get_user_choice()
    data = read_ampl_dat_to_gurobipy(content, dist_mode)
    
    # 保存原始的 U，因为我们可能要篡改它
    original_U = data['U']
    
    # 3. Choose Experiment Mode
    print("\n请选择实验模式:")
    print(f"  1 - 鲁棒优化 (Normal Robust): 使用文件中的 U={original_U} 进行求解，考虑中断风险。")
    print(f"  2 - 脆弱性验证 (Naive Check): 强制设 U=0 求解，然后用 U={original_U} 进行审计。")
    mode = input("请输入 1 或 2: ").strip()
    
    # 获取时间限制
    time_limit = input("最大时间限制 (秒, 回车不限): ").strip()
    tl = float(time_limit) if time_limit else None
    
    if mode == '2':
        # === Naive 模式 ===
        print(f"\n[模式2] 正在将模型输入的 U 篡改为 0 (原值 {original_U})...")
        data['U'] = 0 # <--- 关键修改：欺骗求解器
        
        # 调用标准的鲁棒求解器 (它现在变成了 Naive 求解器)
        model = solve_vehicle_routing_strict(data, tl)
        
        # 恢复 U 以便审计
        data['U'] = original_U 
        
        # 执行独立审计
        if model and model.SolCount > 0:
            perform_audit(model, data, original_U)
            
    else:
        # === 鲁棒模式 ===
        # 正常求解
        model = solve_vehicle_routing_strict(data, tl)
        # 也可以跑一下审计验证一下 (预期名义值应接近实际值)
        if model and model.SolCount > 0:
            perform_audit(model, data, original_U)

    print(f"\n🏁 程序执行完毕。")