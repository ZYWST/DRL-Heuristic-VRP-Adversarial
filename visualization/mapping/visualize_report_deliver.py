import sys, os
_d = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.append(_d)
import pandas as pd
import folium
import re

# ================= 配置区域 =================
NODES_FILE = "data/geo_data/CHINA_filtered_nodes.csv"
EDGES_FILE = "data/geo_data/CHINA_filtered_edges.csv"
REPORT_FILE = "data/solution_report_CHINA_75demands_modified_max_util_weight0223_20260227_090251_seed292024_saved.txt"
OUTPUT_HTML = "深色路网full_network_route_map_solution_report_CHINA_75demands_modified_max_util_weight0223_20260227_090251_seed292024_saved.html"

# 重点城市字典 (将被标记为红点)
TARGET_CITIES = {
    # ================= 直辖市 =================
    "北京市": (39.725414, 116.335442), "天津市": (39.224274, 117.132111),
    "上海市": (31.407391, 121.48484), "重庆市": (29.5630, 106.5516),

    # ================= 华北地区 (河北、山西) =================
    "石家庄市": (38.30722, 114.368057), "唐山市": (39.6301, 118.1802), "秦皇岛市": (39.9354, 119.6005),
    "邯郸市": (36.6256, 114.5391), "邢台市": (37.0708, 114.5048), "保定市": (38.8738, 115.4648),
    "张家口市": (40.8244, 114.8858), "承德市": (40.9644, 117.9624), "沧州市": (38.3045, 116.8386),
    "廊坊市": (39.5380, 116.6838), "衡水市": (37.7390, 115.6987),
    "太原市": (37.8706, 112.5489), "大同市": (40.0768, 113.3001), "阳泉市": (37.8570, 113.5808),
    "长治市": (36.1954, 113.1163), "晋城市": (35.4902, 112.8521), "朔州市": (39.3317, 112.4333),
    "晋中市": (37.6957, 112.7245), "运城市": (35.0264, 110.9982), "忻州市": (38.4165, 112.7342),
    "临汾市": (36.0880, 111.5190), "吕梁市": (37.5243, 111.1343),

    # ================= 华东地区 (江苏、浙江、安徽、福建、江西、山东) =================
    "南京市": (32.0603, 118.7969), "无锡市": (31.568, 120.299), "徐州市": (34.2044, 117.2841),
    "常州市": (31.8112, 119.9741), "苏州市": (31.2990, 120.5853), "南通市": (31.9802, 120.8943),
    "连云港市": (34.5967, 119.2216), "淮安市": (33.6104, 119.0153), "盐城市": (33.3474, 120.1636),
    "扬州市": (32.3942, 119.4129), "镇江市": (32.2052, 119.4528), "泰州市": (32.4555, 119.9234),
    "宿迁市": (33.9630, 118.2748),
    "杭州市": (30.853918, 120.415451), "宁波市": (29.8683, 121.5440), "温州市": (27.9943, 120.6994),
    "嘉兴市": (30.7539, 120.7555), "湖州市": (30.8930, 120.0868), "绍兴市": (30.0024, 120.5861),
    "金华市": (29.1029, 119.6474), "衢州市": (28.9358, 118.8891), "舟山市": (29.9855, 122.2072),
    "台州市": (28.6564, 121.4208), "丽水市": (28.4676, 119.9228),
    "合肥市": (31.8206, 117.2272), "芜湖市": (31.352, 118.375), "蚌埠市": (32.9163, 117.3897),
    "淮南市": (32.6255, 116.9969), "马鞍山市": (31.6700, 118.5067), "淮北市": (33.9558, 116.7983),
    "铜陵市": (30.9455, 117.8115), "安庆市": (30.5248, 117.0429), "黄山市": (29.7147, 118.3375),
    "滁州市": (32.3017, 118.3169), "阜阳市": (32.8901, 115.8142), "宿州市": (33.652, 116.962),
    "六安市": (31.7445, 116.5077), "亳州市": (33.8446, 115.7787), "池州市": (30.6648, 117.4916),
    "宣城市": (30.9407, 118.7588),
    "福州市": (26.0745, 119.2965), "厦门市": (24.4798, 118.0894), "莆田市": (25.4541, 119.0078),
    "三明市": (26.2634, 117.6386), "泉州市": (24.8741, 118.6757), "漳州市": (24.5133, 117.6482),
    "南平市": (26.6420, 118.1777), "龙岩市": (25.0752, 117.0179), "宁德市": (26.6656, 119.5479),
    "南昌市": (28.6820, 115.8579), "景德镇市": (29.2917, 117.2152), "萍乡市": (27.6253, 113.8413),
    "九江市": (29.7051, 115.9928), "新余市": (27.8179, 114.9167), "鹰潭市": (28.2758, 117.0272),
    "赣州市": (25.8311, 114.9347), "吉安市": (27.1117, 114.9793), "宜春市": (27.8155, 114.4166),
    "抚州市": (27.9482, 116.3582), "上饶市": (28.4548, 117.9436),
    "济南市": (36.6512, 117.1201), "青岛市": (36.0671, 120.3826), "淄博市": (36.8132, 118.0550),
    "枣庄市": (34.8716, 117.5681), "东营市": (37.4341, 118.6747), "烟台市": (37.464, 121.448),
    "潍坊市": (36.707, 119.161), "济宁市": (35.415, 116.587), "泰安市": (36.2002, 117.0855),
    "威海市": (37.5131, 122.1204), "日照市": (35.4164, 119.5269), "临沂市": (35.056, 118.352),
    "德州市": (37.4354, 116.3575), "聊城市": (36.4560, 115.9855), "滨州市": (37.3820, 117.9701),
    "菏泽市": (35.2338, 115.4807),

    # ================= 华中地区 (河南、湖北、湖南) =================
    "郑州市": (34.7466, 113.6253), "开封市": (34.797, 114.307), "洛阳市": (34.6181, 112.4540),
    "平顶山市": (33.7662, 113.1928), "安阳市": (36.0968, 114.3925), "鹤壁市": (35.7472, 114.2973),
    "新乡市": (35.303, 113.926), "焦作市": (35.2158, 113.2418), "濮阳市": (35.7618, 115.0292),
    "许昌市": (34.0355, 113.8526), "漯河市": (33.5804, 114.0165), "三门峡市": (34.7726, 111.1997),
    "南阳市": (32.9908, 112.5283), "商丘市": (34.4140, 115.6564), "信阳市": (32.147, 114.091),
    "周口市": (33.6190, 114.6968), "驻马店市": (32.9906, 114.0294),
    "武汉市": (30.5928, 114.3055), "黄石市": (30.2005, 115.0385), "十堰市": (32.6188, 110.7981),
    "宜昌市": (30.6920, 111.2865), "襄阳市": (32.008, 112.122), "鄂州市": (30.3919, 114.8949),
    "荆门市": (31.0354, 112.1994), "孝感市": (30.9168, 113.9556), "荆州市": (30.3323, 112.2393),
    "黄冈市": (30.4354, 114.8724), "咸宁市": (29.8414, 114.3225), "随州市": (31.6905, 113.3824),
    "恩施土家族苗族自治州": (30.2949, 109.4883),
    "长沙市": (28.2282, 112.9388), "株洲市": (27.827, 113.133), "湘潭市": (27.8297, 112.9251),
    "衡阳市": (26.893, 112.572), "邵阳市": (27.2368, 111.4693), "岳阳市": (29.356, 113.129),
    "常德市": (29.0317, 111.6985), "张家界市": (29.1170, 110.4792), "益阳市": (28.5880, 112.3550),
    "郴州市": (25.7705, 113.0147), "永州市": (26.4204, 111.6135), "怀化市": (27.5501, 109.9985),
    "娄底市": (27.7017, 111.9961), "湘西土家族苗族自治州": (28.3128, 109.7388),

    # ================= 华南地区 (广东、广西、海南) =================
    "广州市": (23.1291, 113.2644), "韶关市": (24.8105, 113.5975), "深圳市": (22.5431, 114.0579),
    "珠海市": (22.2707, 113.5767), "汕头市": (23.3541, 116.6819), "佛山市": (23.0215, 113.1214),
    "江门市": (22.5788, 113.0816), "湛江市": (21.2707, 110.3594), "茂名市": (21.6630, 110.9254),
    "肇庆市": (23.0472, 112.4651), "惠州市": (23.111, 114.416), "梅州市": (24.2885, 116.1225),
    "汕尾市": (22.7875, 115.3753), "河源市": (23.7437, 114.7006), "阳江市": (21.8569, 111.9825),
    "清远市": (23.6818, 113.0560), "东莞市": (23.0207, 113.7518), "中山市": (22.5160, 113.3920),
    "潮州市": (23.6569, 116.6226), "揭阳市": (23.5253, 116.3725), "云浮市": (22.9150, 112.0445),
    "南宁市": (22.8170, 108.3665), "柳州市": (24.3255, 109.4126), "桂林市": (25.2736, 110.2902),
    "梧州市": (23.4769, 111.2791), "北海市": (21.4812, 109.1192), "防城港市": (21.6862, 108.3539),
    "钦州市": (21.9810, 108.6538), "贵港市": (23.0936, 109.6105), "玉林市": (22.6366, 110.1653),
    "百色市": (23.9040, 106.6163), "贺州市": (24.4036, 111.5670), "河池市": (24.6939, 108.0851),
    "来宾市": (23.7612, 109.2298), "崇左市": (22.3734, 107.3650),
    "海口市": (20.0174, 110.3492), "三亚市": (18.2528, 109.5120), "三沙市": (16.8310, 112.3386),
    "儋州市": (19.5134, 109.5705),

    # ================= 西南地区 (重庆、四川、贵州、云南) =================
    "成都市": (30.5728, 104.0668), "自贡市": (29.3516, 104.7784), "攀枝花市": (26.5823, 101.7186),
    "泸州市": (28.8724, 105.4405), "德阳市": (31.1268, 104.3980), "绵阳市": (31.4678, 104.7326),
    "广元市": (32.4417, 105.8297), "遂宁市": (30.5328, 105.5929), "内江市": (29.5802, 105.0584),
    "乐山市": (29.552, 103.765), "南充市": (30.799, 106.082), "眉山市": (30.0762, 103.8486),
    "宜宾市": (28.752, 104.643), "广安市": (30.4561, 106.6328), "达州市": (31.2096, 107.4648),
    "雅安市": (29.9802, 102.9976), "巴中市": (31.8588, 106.7537), "资阳市": (30.1293, 104.6349),
    "阿坝藏族羌族自治州": (31.8997, 102.2214), "甘孜藏族自治州": (30.0512, 101.9603),
    "凉山彝族自治州": (27.8916, 102.2678),
    "贵阳市": (26.842445, 106.588608), "六盘水市": (26.5926, 104.8302), "遵义市": (27.7263, 106.9274),
    "安顺市": (26.2530, 105.9283), "毕节市": (27.3017, 105.2863), "铜仁市": (27.7172, 109.1895),
    "黔西南布依族苗族自治州": (25.0881, 104.9062), "黔东南苗族侗族自治州": (26.5835, 107.9826),
    "黔南布依族苗族自治州": (26.2596, 107.5172),
    "昆明市": (24.8801, 102.8329), "曲靖市": (25.490, 103.796), "玉溪市": (24.3520, 102.5427),
    "保山市": (25.1118, 99.1671), "昭通市": (27.3366, 103.7172), "丽江市": (26.8721, 100.2297),
    "普洱市": (22.7851, 100.9723), "临沧市": (23.8776, 100.0869), "楚雄彝族自治州": (25.0389, 101.5401),
    "红河哈尼族彝族自治州": (23.3668, 103.3841), "文山壮族苗族自治州": (23.3767, 104.2440),
    "西双版纳傣族自治州": (22.0017, 100.7979), "大理白族自治州": (25.6065, 100.2676),
    "德宏傣族景颇族自治州": (24.4366, 98.5783), "怒江傈僳族自治州": (25.8509, 98.8543),
    "迪庆藏族自治州": (27.8188, 99.7064),

    # ================= 西北地区 (陕西、青海) =================
    "西安市": (34.620052, 108.927063), "铜川市": (34.8977, 108.9451), "宝鸡市": (34.3615, 107.2375),
    "咸阳市": (34.3296, 108.7090), "渭南市": (34.4994, 109.5089), "延安市": (36.585, 109.489),
    "汉中市": (33.0676, 107.0236), "榆林市": (38.285, 109.734), "安康市": (32.6849, 109.0293),
    "商洛市": (33.8683, 109.9418),
}

# 区域配色配置
REGION_CONFIG = {
    'NorthChina (华北)': {
        'start_nodes': {2469, 2602, 2564, 2640, 2757, 2669},
        'colors': ['#e6194b', '#f58231', '#800000', '#9a6324', '#fabebe']
    },
    'Guanzhong (关中)': {
        'start_nodes': {2976, 3058, 3140},
        'colors': ['#911eb4', '#f032e6', '#e6beff', '#dcbeff', '#4b0082']
    },
    'ChengYu (成渝)': {
        'start_nodes': {51, 734, 263, 253, 969, 971, 197},
        'colors': ['#3cb44b', '#808000', '#aaffc3', '#bcf60c', '#228B22']
    },
    'SouthChina (华南)': {
        'start_nodes': {1063, 1384 ,1437, 1449, 1386, 1419},
        'colors': ['#000075', '#4363d8', '#4169E1', '#000080', '#1E90FF']
    },
    'EastChina (华东)': {
        'start_nodes': {2194, 1831, 1983, 2142, 2132, 1907},
        'colors': ['#46f0f0', '#008080', '#42d4f4', '#00CED1', '#20B2AA']
    }
}

DEFAULT_COLORS = ['#808080', '#A9A9A9', '#000000'] 

def get_region_color(start_node_id, idx):
    for _, config in REGION_CONFIG.items():
        if start_node_id in config['start_nodes']:
            return config['colors'][idx % len(config['colors'])]
    return DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

def load_nodes_and_edges(nodes_path, edges_path):
    print("1. 加载路网数据...")
    df_nodes = pd.read_csv(nodes_path)
    nodes_dict = {row['NodeID']: (row['Lat'], row['Lon']) for _, row in df_nodes.iterrows()}
    
    df_edges = pd.read_csv(edges_path)
    edges_list = []
    for _, row in df_edges.iterrows():
        u, v = int(row['FromNode']), int(row['ToNode'])
        if u in nodes_dict and v in nodes_dict:
            edges_list.append((nodes_dict[u], nodes_dict[v]))
    return nodes_dict, edges_list

def parse_full_report(report_path):
    print("2. 解析报告...")
    vehicles = []
    current_v = None
    
    # 关键正则
    re_vehicle = re.compile(r"--- 车辆 ID:\s*(\d+).*?Depot:\s*(\d+)")
    re_stop = re.compile(r"(?:->|路径详情:)\s*([A-Za-z]+)\((\d+)\)(?:.*@([\d\.]+)min)?(?:\s*\[(.*?)\])?")
    # 匹配报告中的物理路径
    re_phys_path = re.compile(r"🛣️.*?路径.*?: ([\d\->]+)")
    # 匹配中断边 (无论顺序)
    re_break_inline = re.compile(r"-\[.*?断于\s*(\d+)-(\d+).*?\]->")
    # 匹配全车失效标记
    re_fail = re.compile(r"全车失效")

    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    idx_counter = 0
    for line in lines:
        line = line.strip()
        v_match = re_vehicle.search(line)
        if v_match:
            if current_v: vehicles.append(current_v)
            vid = v_match.group(1)
            depot = int(v_match.group(2))
            current_v = {
                'id': vid, 'depot': depot, 
                'stops': [], 'physical_path': [], 
                'broken_edge': None, 'is_failed': False,
                'color': get_region_color(depot, idx_counter)
            }
            idx_counter += 1
            continue
        
        if not current_v: continue

        # 捕获失效标记
        if re_fail.search(line): current_v['is_failed'] = True

        # 捕获逻辑站点
        s_match = re_stop.search(line)
        if s_match:
            stype, sid = s_match.group(1), int(s_match.group(2))
            if stype in ['Depot', 'Supply', 'Demand']:
                info = s_match.group(4) or ""
                # 记录站点信息
                current_v['stops'].append({
                    'id': sid, 'type': stype, 
                    'info': info, 'is_valid': '❌' not in info
                })
        
        # 捕获报告中明确指出的中断边 (作为参考)
        b_match = re_break_inline.search(line)
        if b_match:
            current_v['broken_edge'] = (int(b_match.group(1)), int(b_match.group(2)))

        # 捕获完整物理路径
        p_match = re_phys_path.search(line)
        if p_match:
            path_str = p_match.group(1)
            current_v['physical_path'] = [int(x) for x in path_str.split('->') if x.strip()]

    if current_v: vehicles.append(current_v)
    return vehicles

def draw_enhanced_map(nodes_dict, edges_list, vehicles):
    print("3. 绘制地图 (启用严格中断截断 & 去除回程环路)...")
    if not nodes_dict: return
    
    center = list(nodes_dict.values())[0]
    m = folium.Map(location=center, zoom_start=6, tiles='CartoDB positron')

    # 底图路网
    fg_net = folium.FeatureGroup(name="基础路网", show=True)
    folium.PolyLine(edges_list, color="#7d7c7c", weight=1, opacity=0.3).add_to(fg_net)
    m.add_child(fg_net)

    for v in vehicles:
        raw_path = v['physical_path']
        if not raw_path: continue

        # 先计算截断和中断检测，再构造图层名称，以便在图层右侧显示中断标识

        # ==========================================
        # 步骤 1: 路径截断 (去除空车回程)
        # 目标：找到最后一个干活(Demand/Supply)的站点，把物理路径里该点之后的部分切掉
        # ==========================================
        last_task_node = None
        for stop in reversed(v['stops']):
            if stop['type'] in ['Demand', 'Supply']:
                last_task_node = stop['id']
                break
        
        display_path = raw_path
        if last_task_node is not None:
            try:
                # 在物理路径中找到该站点最后一次出现的索引
                # 使用倒序查找，防止路径中有环路导致切太早
                # list[::-1].index(x) 返回的是倒序的索引
                rev_idx = raw_path[::-1].index(last_task_node)
                last_idx = len(raw_path) - 1 - rev_idx
                # 截断：保留到该点为止
                display_path = raw_path[:last_idx+1]
            except ValueError:
                pass # 逻辑站点不在物理路径中(可能是数据异常)，不做截断

        # ==========================================
        # 步骤 2: 严格中断检测 (Strict Break Detection)
        # 目标：无论报告里的方向如何，只要遇到该边，立即视为中断
        # ==========================================
        valid_coords = []
        break_segment_coords = []
        ghost_coords = []
        
        break_index = -1

        if v['broken_edge']:
            u_brk, v_brk = v['broken_edge']
            broken_set = {u_brk, v_brk}
            
            # 扫描截断后的路径
            for i in range(len(display_path) - 1):
                curr, nxt = display_path[i], display_path[i+1]
                # 检查是否存在无向边匹配
                if {curr, nxt} == broken_set:
                    break_index = i
                    break # ！！！找到第一次出现中断的地方，立即停止扫描！！！

        # 构造 FeatureGroup 名称：如果检测到中断则在括号内加上“被中断”字样
        label_inner = f"Depot {v['depot']}"
        if break_index != -1:
            label_inner = f"{label_inner}，被中断"
        fg_name = f"车辆 {v['id']} ({label_inner})"
        fg = folium.FeatureGroup(name=fg_name, show=False)
        
        # ==========================================
        # 步骤 3: 构建可视化分段
        # ==========================================
        path_nodes_to_draw = [] # 用于最后判定Marker是否在路径上
        
        if break_index != -1:
            # --- 场景 A: 车辆遭遇中断 ---
            # 1. 有效段 (Depot -> 断点前)
            valid_part = display_path[:break_index+1]
            path_nodes_to_draw.extend(valid_part)
            if len(valid_part) > 1:
                folium.PolyLine(
                    [nodes_dict[n] for n in valid_part], 
                    color=v['color'], weight=3, opacity=0.8
                ).add_to(fg)
            
            # 2. 中断段 (断点本身) - 红色加粗
            u_err, v_err = display_path[break_index], display_path[break_index+1]
            coords_err = [nodes_dict[u_err], nodes_dict[v_err]]
            folium.PolyLine(
                coords_err, color='red', weight=5, opacity=1, dash_array='5, 5',
                popup=f"❌ 中断发生于 {u_err}-{v_err}"
            ).add_to(fg)
            folium.Marker(
                [(coords_err[0][0]+coords_err[1][0])/2, (coords_err[0][1]+coords_err[1][1])/2],
                icon=folium.Icon(color='red', icon='remove', prefix='glyphicon')
            ).add_to(fg)
            
            # 3. 幽灵段 (断点后 -> 计划的终点) - 灰色虚线
            # 这部分展示了如果不断路，车辆原本想去哪里
            ghost_part = display_path[break_index+1:]
            path_nodes_to_draw.extend(ghost_part)
            if len(ghost_part) > 1:
                folium.PolyLine(
                    [nodes_dict[n] for n in ghost_part], 
                    color='gray', weight=2, dash_array='3, 6', opacity=0.6,
                    tooltip="中断后未执行的计划路径"
                ).add_to(fg)

        else:
            # --- 场景 B: 车辆正常行驶 ---
            path_nodes_to_draw.extend(display_path)
            coords = [nodes_dict[n] for n in display_path if n in nodes_dict]
            if len(coords) > 1:
                folium.PolyLine(
                    coords, color=v['color'], weight=3, opacity=0.8
                ).add_to(fg)
                
            # 在正常车辆的终点画一个黑色小圆点，表示"任务结束"
            if coords:
                folium.CircleMarker(
                    coords[-1], radius=4, color='black', fill=True, fill_color='white'
                ).add_to(fg)

        # ==========================================
        # 步骤 4: 绘制站点 (仅绘制路径上的点)
        # ==========================================
        # 为了避免地图太乱，我们只画那些在 display_path 里的点
        # 或者是中断位置之前的点
        
        path_node_set = set(path_nodes_to_draw)
        
        for stop in v['stops']:
            sid = stop['id']
            if sid not in nodes_dict: continue
            
            # 如果这个站点不在我们画的线路上，就不标了（除了Depot）
            # 这样可以避免看到那些被截断的回程路上的无意义站点
            if sid not in path_node_set and stop['type'] != 'Depot':
                continue

            icon_color = 'gray'
            icon = 'info-sign'
            if stop['type'] == 'Depot': 
                icon_color = 'black'; icon = 'home'
            elif stop['type'] == 'Supply': 
                icon_color = 'green'; icon = 'arrow-up'
            elif stop['type'] == 'Demand': 
                icon_color = 'blue'; icon = 'arrow-down'
            
            # 如果是无效站点（在中断后），标红
            status_html = ""
            if not stop['is_valid']:
                icon_color = 'red'
                icon = 'ban-circle'
                status_html = "<br><b style='color:red'>[未送达/失效]</b>"
            
            folium.Marker(
                nodes_dict[sid],
                icon=folium.Icon(color=icon_color, icon=icon, prefix='glyphicon'),
                popup=folium.Popup(f"<b>车{v['id']} {stop['type']}</b><br>ID: {sid}<br>{stop['info']}{status_html}", max_width=200),
                tooltip=f"车{v['id']} {sid}"
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUTPUT_HTML)
    print(f"完成！已生成: {OUTPUT_HTML}")

if __name__ == "__main__":
    nodes, edges = load_nodes_and_edges(NODES_FILE, EDGES_FILE)
    vehicles = parse_full_report(REPORT_FILE)
    draw_enhanced_map(nodes, edges, vehicles)