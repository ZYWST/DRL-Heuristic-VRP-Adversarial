import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
#绘制学术论文地图最终版 (支持双微观图自动防交叉)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from shapely.geometry import LineString, box
import geopandas as gpd
import pandas as pd
import matplotlib
import re
import os
import requests
import json 
import math

# ================= 配置区域 =================
NODES_FILE = "data/geo_data/CHINA_filtered_nodes.csv"
EDGES_FILE = "data/geo_data/CHINA_filtered_edges.csv"
REPORT_FILE = "data/solution_report_CHINA_75demands_modified_max_util_weight0223_20260227_090251_seed292024_saved.txt"

FIG_SIZE = (16, 9)
DPI = 300

# 全局字体设置
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ================= 1. 数据解析 =================
def load_data_full():
    print("1. 读取节点数据...")
    df_nodes = pd.read_csv(NODES_FILE)
    nodes_dict = {int(row['NodeID']): (float(row['Lon']), float(row['Lat'])) for _, row in df_nodes.iterrows()}
    
    print("2. 读取背景路网...")
    df_edges_bg = pd.read_csv(EDGES_FILE)
    bg_lines = []
    for _, row in df_edges_bg.iterrows():
        u, v = int(row['FromNode']), int(row['ToNode'])
        if u in nodes_dict and v in nodes_dict:
            bg_lines.append(LineString([nodes_dict[u], nodes_dict[v]]))
    
    print("3. 解析报告流量...")
    edge_flow = {} 
    vehicle_paths = [] 
    broken_edges = []
    
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_v = None
    reading_breaks = False
    for line in lines:
        line = line.strip()
        if "=== ⚠️" in line: reading_breaks = True
        if reading_breaks and "边" in line:
            m = re.search(r"边\s*(\d+)\s*->\s*(\d+)", line)
            if m: broken_edges.append((int(m.group(1)), int(m.group(2))))
        if "---" in line and "===" in line: reading_breaks = False

        if "--- 车辆 ID" in line:
            if current_v: vehicle_paths.append(current_v)
            vid_m = re.search(r"ID:\s*(\d+)", line)
            if vid_m:
                vid = vid_m.group(1)
                current_v = {'id': vid, 'path': [], 'broken': False}
        
        if "完整物理路径" in line and current_v:
            path_str_m = re.search(r"完整物理路径[:：]\s*([\d\->\s]+)", line)
            if path_str_m:
                path_str = path_str_m.group(1)
                p = [int(x.strip()) for x in path_str.split('->') if x.strip().isdigit()]
                current_v['path'] = p
                for i in range(len(p)-1):
                    edge = tuple(sorted((p[i], p[i+1])))
                    edge_flow[edge] = edge_flow.get(edge, 0) + 1
        
        if ("全车失效" in line or "严格模式" in line) and current_v:
            current_v['broken'] = True
            
    if current_v: vehicle_paths.append(current_v)
    
    # 边去重
    unique_broken = set()
    for u, v in broken_edges:
        unique_broken.add(tuple(sorted((u, v))))
    broken_edges = list(unique_broken)

    return nodes_dict, bg_lines, edge_flow, vehicle_paths, broken_edges

# ================= 2. 获取中国矢量底图 =================
def get_china_map_vectors():
    local_file = "data/geo_data/china_map_cache.json"
    if not os.path.exists(local_file):
        url = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            with open(local_file, 'wb') as f:
                f.write(response.content)
        except Exception:
            return None
    try:
        with open(local_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        gdf.set_crs(epsg=4326, inplace=True)
        return gdf
    except Exception:
        return None

# ================= 3. 绘图主程序 =================
def draw_paper_figure(nodes_dict, bg_lines, edge_flow, vehicle_paths, broken_edges):
    
    # --- A. 智能寻找需要放大的目标区域 ---
    target_regions = []
    if broken_edges:
        target_regions.append(broken_edges[0])
        
        if len(broken_edges) > 1:
            u1, v1 = broken_edges[0]
            c_lon1 = (nodes_dict[u1][0] + nodes_dict[v1][0]) / 2
            c_lat1 = (nodes_dict[u1][1] + nodes_dict[v1][1]) / 2
            
            max_dist = -1
            furthest_edge = None
            
            for edge in broken_edges[1:]:
                u2, v2 = edge
                c_lon2 = (nodes_dict[u2][0] + nodes_dict[v2][0]) / 2
                c_lat2 = (nodes_dict[u2][1] + nodes_dict[v2][1]) / 2
                dist = math.hypot(c_lon1 - c_lon2, c_lat1 - c_lat2)
                if dist > max_dist:
                    max_dist = dist
                    furthest_edge = edge
            
            if max_dist > 1.5:
                target_regions.append(furthest_edge)

    # ==== [关键修复] 按纬度从北到南排序目标区域，避免连线交叉 ====
    if len(target_regions) > 1:
        # 提取中心纬度并降序排序
        target_regions.sort(key=lambda edge: (nodes_dict[edge[0]][1] + nodes_dict[edge[1]][1])/2, reverse=True)

    print(f"   -> [智能识别] 识别出 {len(target_regions)} 个需要放大的宏观热点区域，已按南北位置排序。")

    # --- B. 准备画布与坐标轴 ---
    fig = plt.figure(figsize=FIG_SIZE, dpi=330) 
    ax_main = fig.add_axes([0.02, 0.05, 0.60, 0.90]) 
    
    ax_insets = []
    if len(target_regions) == 1:
        ax_insets.append(fig.add_axes([0.65, 0.30, 0.33, 0.40]))
    elif len(target_regions) == 2:
        ax_insets.append(fig.add_axes([0.65, 0.52, 0.33, 0.40])) # Top Inset (北方区域)
        ax_insets.append(fig.add_axes([0.65, 0.08, 0.33, 0.40])) # Bottom Inset (南方区域)

    # --- C. 绘制主图 ---
    gdf_china = get_china_map_vectors()
    if gdf_china is not None:
        gdf_china = gdf_china.to_crs(epsg=3857)
        gdf_china.plot(ax=ax_main, color='#F2F2F2', edgecolor='#DCDCDC', linewidth=1, zorder=0)

    if bg_lines:
        gdf_bg = gpd.GeoDataFrame(geometry=bg_lines, crs="EPSG:4326").to_crs(epsg=3857)
        gdf_bg.plot(ax=ax_main, color="#C3BEBE", linewidth=0.5, alpha=0.6, zorder=1)

    flow_lines, flow_counts = [], []
    for (u, v), flow in edge_flow.items():
        if u in nodes_dict and v in nodes_dict:
            flow_lines.append(LineString([nodes_dict[u], nodes_dict[v]]))
            flow_counts.append(flow)
            
    if flow_lines:
        gdf_flow = gpd.GeoDataFrame({'flow': flow_counts, 'geometry': flow_lines}, crs="EPSG:4326").to_crs(epsg=3857)
        max_flow = max(flow_counts) if flow_counts else 1
        try: cmap = matplotlib.colormaps['YlOrRd']
        except: cmap = plt.cm.get_cmap('YlOrRd')
        
        colors = [cmap(f / max_flow) for f in gdf_flow['flow']]
        widths = [1.0 + (f / max_flow) * 4.0 for f in gdf_flow['flow']]
        
        gdf_flow.plot(ax=ax_main, color=colors, linewidth=widths, alpha=0.9, zorder=2, 
                     capstyle='round', joinstyle='round')

    for u, v in broken_edges:
        if u in nodes_dict and v in nodes_dict:
            line = LineString([nodes_dict[u], nodes_dict[v]])
            g_break = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=3857)
            g_break.plot(ax=ax_main, color='white', linewidth=5, zorder=3)
            g_break.plot(ax=ax_main, color='#D32F2F', linewidth=3, zorder=4)
            mid = g_break.iloc[0].interpolate(0.5, normalized=True)
            ax_main.plot(mid.x, mid.y, marker='X', markersize=12, color='#B71C1C', 
                         markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    if bg_lines:
        bounds = gdf_bg.total_bounds
        margin = 100000
        ax_main.set_xlim(bounds[0]-margin, bounds[2]+margin)
        ax_main.set_ylim(bounds[1]-margin, bounds[3]+margin)

    # --- D. 循环绘制微观图 (Insets) ---
    for idx, (target_edge, ax_in) in enumerate(zip(target_regions, ax_insets)):
        u, v = target_edge
        center_lon = (nodes_dict[u][0] + nodes_dict[v][0]) / 2
        center_lat = (nodes_dict[u][1] + nodes_dict[v][1]) / 2
        
        center_pt = gpd.GeoSeries([LineString([(center_lon, center_lat), (center_lon, center_lat)])], crs="EPSG:4326").to_crs(epsg=3857).iloc[0].centroid
        radius = 200000 
        minx, maxx = center_pt.x - radius, center_pt.x + radius
        miny, maxy = center_pt.y - radius, center_pt.y + radius
        
        ax_in.set_xlim(minx, maxx)
        ax_in.set_ylim(miny, maxy)
        
        if gdf_china is not None:
            gdf_china.plot(ax=ax_in, color='#F5F5F5', edgecolor='#DCDCDC', linewidth=1, zorder=0)
        if bg_lines:
            try:
                gdf_local = gdf_bg.cx[minx:maxx, miny:maxy]
                if not gdf_local.empty:
                    gdf_local.plot(ax=ax_in, color='#999999', linewidth=0.5, alpha=0.4, zorder=1)
            except: pass

        for bu, bv in broken_edges:
            line = LineString([nodes_dict[bu], nodes_dict[bv]])
            g_break = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(epsg=3857)
            if g_break.iloc[0].intersects(box(minx, miny, maxx, maxy)):
                 g_break.plot(ax=ax_in, color='#D32F2F', linewidth=4, zorder=10)
                 mid = g_break.iloc[0].interpolate(0.5, normalized=True)
                 ax_in.plot(mid.x, mid.y, marker='X', markersize=10, color='#D32F2F', 
                            markeredgecolor='white', markeredgewidth=1, zorder=11)

        count = 0
        for veh in vehicle_paths:
            if u in veh['path'] or v in veh['path']:
                p_coords = [nodes_dict[n] for n in veh['path'] if n in nodes_dict]
                if len(p_coords) < 2: continue
                ls = LineString(p_coords)
                gs = gpd.GeoSeries([ls], crs="EPSG:4326").to_crs(epsg=3857)
                
                if veh['broken']:
                    gs.plot(ax=ax_in, color='white', linewidth=2, zorder=5)
                    gs.plot(ax=ax_in, color='#D32F2F', linestyle='--', linewidth=1.5, zorder=6)
                else:
                    gs.plot(ax=ax_in, color='white', linewidth=3, zorder=7)
                    gs.plot(ax=ax_in, color='#1B5E20', linewidth=2, zorder=8)
                count += 1
                if count >= 10: break

        ax_in.set_title(f"局部视图 {idx+1}: 路段 {u}-{v} 附近", fontsize=14, fontweight='bold', pad=10)
        ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_in.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(1.2)

        rect = plt.Rectangle((minx, miny), maxx-minx, maxy-miny, transform=ax_main.transData, 
                             fill=False, edgecolor='black', linewidth=1.2, linestyle='--')
        ax_main.add_patch(rect)
        
        # 微微调淡连接线的颜色和粗细，使其不至于太抢眼
        con1 = ConnectionPatch(xyA=(maxx, maxy), coordsA=ax_main.transData, xyB=(0, 1), coordsB=ax_in.transAxes, color="#555555", linestyle="--", linewidth=0.6)
        con2 = ConnectionPatch(xyA=(maxx, miny), coordsA=ax_main.transData, xyB=(0, 0), coordsB=ax_in.transAxes, color="#555555", linestyle="--", linewidth=0.6)
        fig.add_artist(con1)
        fig.add_artist(con2)

    ax_main.set_axis_off()
    
    legend_elements = [
        Line2D([0], [0], color='#F2F2F2', marker='s', markersize=10, markeredgecolor='#DCDCDC', label='省界'),
        Line2D([0], [0], color=matplotlib.colormaps['YlOrRd'](0.8), lw=4, label='流量（高）'),
        Line2D([0], [0], color='#D32F2F', lw=0, marker='X', markersize=8, label='被中断路段'),
        Line2D([0], [0], color='#1B5E20', lw=2, label='迂回路径'),
        Line2D([0], [0], color='#D32F2F', lw=2, linestyle='--', label='中断路径')
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=16)
    
    output_filename = "Final_Academic_Map_Parallel_Lines.svg"
    plt.savefig(output_filename, dpi=330, bbox_inches='tight')
    # plt.savefig("Final_Academic_Map_Parallel_Lines.png", dpi=300, bbox_inches='tight')
    print(f"✅ 绘图完成！已保存为 {output_filename} 和 png 格式")

if __name__ == "__main__":
    nodes, bg, flow, paths, breaks = load_data_full()
    draw_paper_figure(nodes, bg, flow, paths, breaks)