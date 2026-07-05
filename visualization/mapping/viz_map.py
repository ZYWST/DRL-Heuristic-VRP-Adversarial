import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import pandas as pd
import folium

# ================= 配置区域 =================
INPUT_NODES = "CHINA_filtered_nodes.csv"
INPUT_EDGES = "CHINA_filtered_edges.csv"
OUTPUT_HTML = "network_visualization.html"

# 可视化样式设置
NODE_COLOR = "blue"
NODE_RADIUS = 3
EDGE_COLOR = "gray"
EDGE_WEIGHT = 1.5
EDGE_OPACITY = 0.5

def visualize_network():
    print("1. 正在读取数据...")
    try:
        nodes = pd.read_csv(INPUT_NODES)
        edges = pd.read_csv(INPUT_EDGES)
    except FileNotFoundError:
        print("错误：未找到CSV文件，请检查路径。")
        return

    # 数据量检查与警告
    if len(nodes) > 5000:
        print(f"警告：节点数量较多 ({len(nodes)}个)，生成地图可能会变慢或导致浏览器卡顿。")

    print("2. 正在处理坐标映射...")
    # 将节点ID映射到坐标 (Lat, Lon)，提高画边时的查找速度
    # 假设CSV列名为 NodeID, Lat, Lon
    node_map = {
        row['NodeID']: (row['Lat'], row['Lon']) 
        for _, row in nodes.iterrows()
    }

    print("3. 初始化地图...")
    # 计算中心点
    center_lat = nodes['Lat'].mean()
    center_lon = nodes['Lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')

    # --- 绘制边 (Roads) ---
    print(f"4. 正在绘制 {len(edges)} 条边...")
    # 使用 FeatureGroup 可以作为一个图层整体控制
    edges_layer = folium.FeatureGroup(name="Edges (Roads)")
    
    for _, row in edges.iterrows():
        u, v = row['FromNode'], row['ToNode']
        
        # 确保边的两个端点都在节点列表中
        if u in node_map and v in node_map:
            p1 = node_map[u]
            p2 = node_map[v]
            
            folium.PolyLine(
                locations=[p1, p2],
                color=EDGE_COLOR,
                weight=EDGE_WEIGHT,
                opacity=EDGE_OPACITY
            ).add_to(edges_layer)
    
    edges_layer.add_to(m)

    # --- 绘制节点 (Nodes) ---
    print(f"5. 正在绘制 {len(nodes)} 个节点...")
    nodes_layer = folium.FeatureGroup(name="Nodes")
    
    for _, row in nodes.iterrows():
        nid = int(row['NodeID'])
        lat = row['Lat']
        lon = row['Lon']
        
        # 构造标签信息
        # Tooltip: 鼠标悬停显示
        # Popup: 鼠标点击显示
        label_info = f"ID: {nid}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
        
        folium.CircleMarker(
            location=(lat, lon),
            radius=NODE_RADIUS,
            color=NODE_COLOR,
            fill=True,
            fill_color=NODE_COLOR,
            fill_opacity=0.7,
            popup=folium.Popup(label_info, max_width=200),
            tooltip=label_info  # 这里实现了你的需求：标签含有经纬度
        ).add_to(nodes_layer)

    nodes_layer.add_to(m)

    # 添加图层控制 (可以在地图右上角开关节点或边)
    folium.LayerControl().add_to(m)

    print(f"6. 保存地图至 {OUTPUT_HTML}...")
    m.save(OUTPUT_HTML)
    print("完成！请在浏览器中打开生成的 HTML 文件。")

if __name__ == "__main__":
    visualize_network()