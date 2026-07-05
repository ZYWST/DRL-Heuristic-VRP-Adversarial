import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import pandas as pd
import folium
from folium import plugins
import re
import json

# ================= 配置区域 =================
NODES_FILE = "data/geo_data/CHINA_filtered_nodes.csv"
REPORT_FILE = "data/solution_report_CHINA_75demands_modified_max_util_weight0223_20260227_090251_seed292024_saved.txt"
OUTPUT_HTML = "interactive_analysis_dashboard_solution_report_CHINA_75demands_modified_max_util_weight0223_20260227_090251_seed292024_saved.html"

# ================= 1. 数据解析 =================
def load_data_analysis():
    # 1. 加载节点
    df_nodes = pd.read_csv(NODES_FILE)
    nodes_dict = {row['NodeID']: [row['Lat'], row['Lon']] for _, row in df_nodes.iterrows()}
    
    # 2. 解析报告
    vehicles = []
    current_v = None
    
    # 正则表达式
    re_vehicle = re.compile(r"--- 车辆 ID:\s*(\d+).*?Depot:\s*(\d+)")
    re_stop = re.compile(r"(?:->|路径详情:)\s*([A-Za-z]+)\((\d+)\)(?:.*\[(.*?)\])?")
    re_phys = re.compile(r"🛣️.*?路径.*?: ([\d\->]+)")
    re_fail = re.compile(r"全车失效")
    # [新增] 解析中断边
    re_break_inline = re.compile(r"-\[.*?断于\s*(\d+)-(\d+).*?\]->")

    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        v_match = re_vehicle.search(line)
        if v_match:
            if current_v: vehicles.append(current_v)
            current_v = {
                'id': v_match.group(1),
                'depot': int(v_match.group(2)),
                'stops': [],
                'path_nodes': [],
                'is_failed': False,
                'broken_edge': None, # [新增]
                'logs': [] 
            }
        
        if not current_v: continue
        
        if re_fail.search(line):
            current_v['is_failed'] = True

        # [新增] 捕获中断边
        b_match = re_break_inline.search(line)
        if b_match:
            current_v['broken_edge'] = (int(b_match.group(1)), int(b_match.group(2)))

        # 解析物理路径
        p_match = re_phys.search(line)
        if p_match:
            try:
                current_v['path_nodes'] = [int(x) for x in p_match.group(1).split('->')]
            except: pass

        # 解析站点和装卸货信息
        s_match = re_stop.search(line)
        if s_match:
            node_type = s_match.group(1)
            node_id = int(s_match.group(2))
            info = s_match.group(3) if s_match.group(3) else ""
            
            delivery_info = "无操作"
            if info:
                delivery_info = info
                
            current_v['stops'].append({
                'id': node_id,
                'type': node_type,
                'info': delivery_info
            })

    if current_v: vehicles.append(current_v)
    return nodes_dict, vehicles

# ================= 2. 交互式地图绘制 =================
def draw_interactive_dashboard(nodes_dict, vehicles):
    print("正在构建交互式分析仪表盘 (启用严格中断截断 & 去除回程环路)...")
    
    # 初始化地图
    center_node = list(nodes_dict.values())[0]
    m = folium.Map(location=center_node, zoom_start=5, tiles='CartoDB positron')

    # 图层控制
    fg_failed = folium.FeatureGroup(name="⚠️ 失效车辆 (Failed)", show=True)
    fg_success = folium.FeatureGroup(name="✅ 正常车辆 (Success)", show=True)
    m.add_child(fg_failed)
    m.add_child(fg_success)

    # --- 绘制车辆路径 ---
    for v in vehicles:
        raw_path = v['path_nodes']
        if not raw_path: continue
        
        # ==========================================
        # 步骤 1: 路径截断 (去除空车回程)
        # 目标：找到最后一个干活(Demand/Supply)的站点，把物理路径里该点之后的部分切掉
        # ==========================================
        last_service_node_id = None
        for stop in reversed(v['stops']):
            if stop['type'] in ['Demand', 'Supply']:
                last_service_node_id = stop['id']
                break
        
        display_path = raw_path
        if last_service_node_id is not None:
            try:
                # 倒序查找，确保找到的是最后一次经过该点的位置
                rev_idx = raw_path[::-1].index(last_service_node_id)
                last_idx = len(raw_path) - 1 - rev_idx
                display_path = raw_path[:last_idx+1]
            except ValueError:
                pass
        
        # ==========================================
        # 步骤 2: 检测中断位置 (无向边匹配)
        # ==========================================
        break_index = -1
        if v['broken_edge']:
            u_brk, v_brk = v['broken_edge']
            broken_set = {u_brk, v_brk}
            
            for i in range(len(display_path) - 1):
                curr, nxt = display_path[i], display_path[i+1]
                if {curr, nxt} == broken_set:
                    break_index = i
                    break

        # ==========================================
        # 步骤 3: 准备绘图数据 (Popup & Layer)
        # ==========================================
        
        # 确定图层和基础颜色
        if v['is_failed']:
            base_color = '#e74c3c' # 红色
            layer = fg_failed
            status_text = "FAILED"
        else:
            base_color = '#2ecc71' # 绿色
            layer = fg_success
            status_text = "SUCCESS"

        # 构建 Popup 表格
        table_rows = []
        for s in v['stops']:
            table_rows.append(f"<tr><td>{s['id']}</td><td>{s['type']}</td><td>{s['info']}</td></tr>")

        table_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 280px;">
            <h4>Vehicle {v['id']} ({status_text})</h4>
            <b>Depot:</b> {v['depot']}<br>
            <i style="color:gray;">*路径已隐藏空车返程段</i>
            <table style="width:100%; border-collapse: collapse; margin-top:5px;" border="1">
                <tr style="background-color:#eee;"><th>Node</th><th>Type</th><th>Action</th></tr>
                {''.join(table_rows)}
            </table>
        </div>
        """
        
        # 辅助函数：绘制一段 GeoJson
        def add_geojson_segment(segment_nodes, color, weight, opacity, dash_array=None, tooltip_extra=""):
            coords = [nodes_dict[n] for n in segment_nodes if n in nodes_dict]
            if len(coords) < 2: return
            
            line_geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in coords] 
                },
                "properties": {
                    "vehicle_id": v['id'],
                    "status": status_text
                }
            }
            
            folium.GeoJson(
                line_geojson,
                name=f"Vehicle {v['id']}",
                style_function=lambda x: {
                    'color': color, 'weight': weight, 'opacity': opacity, 'dashArray': dash_array
                },
                highlight_function=lambda x: {
                    'color': 'blue', 'weight': 6, 'opacity': 1.0
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['vehicle_id', 'status'],
                    aliases=['Vehicle ID:', 'Status:'],
                    style="font-size: 14px; font-weight: bold;"
                ),
                popup=folium.Popup(table_html, max_width=300)
            ).add_to(layer)

        # ==========================================
        # 步骤 4: 执行绘制
        # ==========================================
        
        if break_index != -1:
            # --- 场景 A: 中断车辆 (三段式绘制) ---
            
            # 1. 有效段 (Solid)
            valid_part = display_path[:break_index+1]
            add_geojson_segment(valid_part, base_color, 4, 0.8)
            
            # 2. 中断边 (Red Dashed)
            u_err, v_err = display_path[break_index], display_path[break_index+1]
            if u_err in nodes_dict and v_err in nodes_dict:
                break_coords = [nodes_dict[u_err], nodes_dict[v_err]]
                folium.PolyLine(
                    break_coords, color='red', weight=5, dash_array='5, 5', opacity=1,
                    tooltip=f"Vehicle {v['id']} BREAK at {u_err}-{v_err}",
                    popup=folium.Popup(table_html, max_width=300)
                ).add_to(layer)
                # 红叉标记
                folium.Marker(
                    [(break_coords[0][0]+break_coords[1][0])/2, (break_coords[0][1]+break_coords[1][1])/2],
                    icon=folium.Icon(color='red', icon='remove', prefix='glyphicon')
                ).add_to(layer)
            
            # 3. 幽灵段 (Gray Dashed)
            ghost_part = display_path[break_index+1:]
            add_geojson_segment(ghost_part, 'gray', 3, 0.6, dash_array='5, 10', tooltip_extra=" (Planned)")
            
        else:
            # --- 场景 B: 正常车辆 (整体绘制) ---
            add_geojson_segment(display_path, base_color, 3, 0.6)
            
            # 在终点画一个黑色小圆点，表示"任务结束"
            if display_path:
                end_node = display_path[-1]
                if end_node in nodes_dict:
                    folium.CircleMarker(
                        location=nodes_dict[end_node], radius=4, color='black', fill=True, fill_color='white',
                        fill_opacity=1, tooltip=f"Vehicle {v['id']} End Task"
                    ).add_to(layer)

    # --- 绘制站点详情 (Stops) ---
    fg_stops = folium.FeatureGroup(name="📍 站点详情 (Stops)", show=True)
    m.add_child(fg_stops)

    for v in vehicles:
        # 获取该车辆绘制路径的节点集合，用于过滤不在路径上的站点
        # 注意：对于中断车辆，我们也画了幽灵路径，所以站点可能在幽灵路径上，这也合理（显示原本计划去哪）
        # 但我们仍然要基于 display_path 过滤，以避免显示回程 Depot 附近的无关点
        
        # 这里的截断逻辑复用上面的
        last_service_node_id = None
        for stop in reversed(v['stops']):
            if stop['type'] in ['Demand', 'Supply']:
                last_service_node_id = stop['id']
                break
        path_to_check = v['path_nodes']
        if last_service_node_id is not None:
             try:
                rev_idx = v['path_nodes'][::-1].index(last_service_node_id)
                last_idx = len(v['path_nodes']) - 1 - rev_idx
                path_to_check = v['path_nodes'][:last_idx+1]
             except: pass
        
        path_node_set = set(path_to_check)

        for s in v['stops']:
            if s['id'] not in nodes_dict: continue
            if s['type'] == 'Node': continue 
            
            # 过滤掉不在显示路径上的点（除了Depot）
            if s['id'] not in path_node_set and s['type'] != 'Depot':
                continue

            lat, lon = nodes_dict[s['id']]
            
            if s['type'] == 'Demand':
                icon_color = 'orange'
                if v['is_failed']: icon_color = 'red'
            elif s['type'] == 'Supply':
                icon_color = 'green'
            else:
                icon_color = 'gray'

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='white',
                weight=1,
                fill=True,
                fill_color=icon_color,
                fill_opacity=0.8,
                tooltip=f"<b>{s['type']} {s['id']}</b><br>{s['info']}",
                popup=f"Vehicle {v['id']} @ Node {s['id']}"
            ).add_to(fg_stops)

    # 辅助工具
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    
    m.save(OUTPUT_HTML)
    print(f"分析仪表盘已生成: {OUTPUT_HTML}")

if __name__ == "__main__":
    nodes_dict, vehicles = load_data_analysis()
    draw_interactive_dashboard(nodes_dict, vehicles)