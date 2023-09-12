# 导入必要的库
import numpy as np

# 定义节点信息
nodes = ["O", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "D"]
start_node = "O"
end_node = "D"
middle_nodes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
transport_modes = ["公路", "铁路", "水路"]
# 三角模糊数表示的货物重量
q_fuzzy = (8, 18, 22)  # 单位: t
q_l, q_m, q_u = q_fuzzy

def get_fuzzy_components(fuzzy_number):
    """
    提取三角模糊数的各个组成部分
    """
    l, m, u = fuzzy_number
    return l, m, u

# 提前到达和延迟到达的成本
Pe = 30  # 元/(t·h)
Pl = 50  # 元/( t·h)

# 运输过程中的最大运输成本和碳排放量限制
max_cost = 70000  # 元
max_emission = 6000  # kg

# 节点间距离 (单位: km)
# 格式: {(起点, 终点): {"公路": 距离, "铁路": 距离, "水路": 距离}}
distances = {
    ("O", "1"): {"公路": 632, "铁路": 757, "水路": 515},
    ("O", "3"): {"公路": 670, "铁路": 707, "水路": 562},
    ("O", "4"): {"公路": 780, "铁路": 1049, "水路": 667},
    ("1", "2"): {"公路": 259, "铁路": 316, "水路": 0},
    ("1", "4"): {"公路": 636, "铁路": 688, "水路": 0},
    ("2", "6"): {"公路": 868, "铁路": 937, "水路": 665},
    ("2", "7"): {"公路": 770, "铁路": 883, "水路": 609},
    ("3", "4"): {"公路": 336, "铁路": 342, "水路": 0},
    ("3", "5"): {"公路": 340, "铁路": 362, "水路": 0},
    ("4", "5"): {"公路": 342, "铁路": 0, "水路": 0},
    ("4", "6"): {"公路": 587, "铁路": 838, "水路": 0},
    ("5", "8"): {"公路": 513, "铁路": 536, "水路": 0},
    ("5", "9"): {"公路": 855, "铁路": 1226, "水路": 729},
    ("6", "7"): {"公路": 297, "铁路": 0, "水路": 0},
    ("6", "9"): {"公路": 622, "铁路": 667, "水路": 579},
    ("7", "9"): {"公路": 813, "铁路": 912, "水路": 739},
    ("8", "9"): {"公路": 446, "铁路": 0, "水路": 0},
    ("8", "10"): {"公路": 419, "铁路": 408, "水路": 0},
    ("9", "10"): {"公路": 316, "铁路": 0, "水路": 0},
    ("9", "11"): {"公路": 320, "铁路": 301, "水路": 0},
    ("10", "11"): {"公路": 311, "铁路": 292, "水路": 0},
    ("10", "D"): {"公路": 292, "铁路": 281, "水路": 0},
    ("11", "D"): {"公路": 134, "铁路": 0, "水路": 0},
}

#提取dij(k)的值
def get_distance(i, j, k):
    return distances.get((i, j), {}).get(k, 0)

# 三种运输方式单位碳排放量和单位运费
unit_emissions = {"公路": 0.12, "铁路": 0.025, "水路": 0.049}
unit_fees = {"公路": 0.68, "铁路": 0.41, "水路": 0.27}

#提取Cij(k)的值
def get_unit_fee(k):
    return unit_fees.get(k, 0)

#提取Eij(k)的值
def get_unit_emission(k):
    return unit_emissions.get(k, 0)

# 各运输方式之间的换装碳排放量、换装成本和模糊中转时间
transfer_info = {
    ("铁路", "公路"): {"碳排放": 1.56, "换装成本": 5.13, "换装时间": (1, 1.5, 2)},
    ("铁路", "水路"): {"碳排放": 3.12, "换装成本": 7.31, "换装时间": (2, 2.5, 3)},
    ("公路", "水路"): {"碳排放": 6, "换装成本": 6.46, "换装时间": (1.5, 2, 2.5)},
}
#提取模糊中转时间
def get_fuzzy_transfer_time(i, k, l):
    return transfer_info.get((k, l), {}).get("换装时间", (0, 0, 0))

#提取Ci(kl)的值
def get_transfer_fee(i, k, l):
    return transfer_info.get((k, l), {}).get("换装成本", 0)

#提取Ei(kl)的值
def get_transfer_emission(i, k, l):
    return transfer_info.get((k, l), {}).get("碳排放", 0)

# 两两节点间的模糊运输时间
fuzzy_transport_time = {
    ("O", "1"): {"公路": (6, 7, 8), "铁路": (4, 5, 6), "水路": (10, 16, 20)},
    ("O", "3"): {"公路": (5, 6, 7), "铁路": (9, 10, 13), "水路": (18, 22, 28)},
    ("O", "4"): {"公路": (7, 8, 9), "铁路": (9, 10, 13), "水路": (20, 24, 30)},
    ("1", "2"): {"公路": (11, 12, 13), "铁路": (16, 20, 24), "水路": (0, 0, 0)},
    ("1", "4"): {"公路": (17, 18, 19), "铁路": (17, 20, 25), "水路": (0, 0, 0)},
    ("2", "6"): {"公路": (18, 19, 20), "铁路": (16, 20, 24), "水路": (24, 28, 35)},
    ("2", "7"): {"公路": (14, 15, 16), "铁路": (17, 20, 25), "水路": (25, 28, 40)},
    ("3", "4"): {"公路": (8, 10, 12), "铁路": (8, 10, 12),"水路": (0, 0, 0)},
    ("3", "5"): {"公路": (13, 14, 15), "铁路": (7, 9, 11), "水路": (0, 0, 0)},
    ("4", "5"): {"公路": (9, 10, 11), "铁路": (0, 0, 0), "水路": (0, 0, 0)},
    ("4", "6"): {"公路": (4, 5, 6), "铁路": (6, 8, 9), "水路": (0, 0, 0)},
    ("5", "8"): {"公路": (11, 12, 13), "铁路": (14, 17, 21), "水路": (0, 0, 0)},
    ("5", "9"): {"公路": (9, 10, 11), "铁路": (6, 8, 9), "水路": (16, 19, 24)},
    ("6", "7"): {"公路": (8, 9, 10), "铁路": (7, 9, 11), "水路": (0, 0, 0)},
    ("6", "9"): {"公路": (16, 16, 17), "铁路": (15, 18, 20), "水路": (27, 32, 40)},
    ("7", "9"): {"公路": (13, 14, 15), "铁路": (8, 10, 12), "水路": (23, 28, 35)},
    ("8", "9"): {"公路": (7, 8, 9), "铁路": (0, 0, 0), "水路": (0, 0, 0)},
    ("8", "10"): {"公路": (17, 18, 19), "铁路": (15, 18, 23), "水路": (0, 0, 0)},
    ("9", "10"): {"公路": (18, 19, 20), "铁路": (0, 0, 0), "水路": (0, 0, 0)},
    ("9", "11"): {"公路": (14, 15, 16), "铁路": (17, 20, 25), "水路": (0, 0, 0)},
    ("1O", "11"): {"公路": (15, 16, 17), "铁路": (14, 16, 20), "水路": (0, 0, 0)},
    ("1O", "D"): {"公路": (9, 10, 11), "铁路": (6, 8, 9), "水路": (0, 0, 0)},
    ("11", "D"): {"公路": (11, 12, 13), "铁路": (0, 0, 0), "水路": (0, 0, 0)},
}

#提取模糊运输时间
def get_fuzzy_transport_time(i, j, k):
    return fuzzy_transport_time.get((i, j), {}).get(k, (0, 0, 0))

#提取到达节点的模糊时间ti
def calculate_ti(path):
    ti_values = []
    total_time = (0, 0, 0)
    for idx in range(len(path) - 1):
        i, j = path[idx], path[idx + 1]
        for k in transport_modes:
            t_ijk = get_fuzzy_transport_time(i, j, k)
            total_time = tuple(map(sum, zip(total_time, t_ijk)))
        if idx < len(path) - 2:
            for k, l in transfer_info:
                t_ikl = get_fuzzy_transfer_time(i, k, l)
                total_time = tuple(map(sum, zip(total_time, t_ikl)))
        ti_values.append(total_time)
    return ti_values


# 两两节点最大运输能力
max_transport_capacity = {
    ("O", "1"): {"公路": 20, "铁路": 25, "水路": 15},
    ("O", "3"): {"公路": 28, "铁路": 22, "水路": 25},
    ("O", "4"): {"公路": 19, "铁路": 21, "水路": 20},
    ("1", "2"): {"公路": 20, "铁路": 25, "水路": 0},
    ("1", "4"): {"公路": 32, "铁路": 26, "水路": 0},
    ("2", "6"): {"公路": 18, "铁路": 24, "水路": 20},
    ("2", "7"): {"公路": 22, "铁路": 20, "水路": 26},
    ("3", "4"): {"公路": 20, "铁路": 15, "水路": 0},
    ("3", "5"): {"公路": 24, "铁路": 28, "水路": 0},
    ("4", "5"): {"公路": 30, "铁路": 0, "水路": 0},
    ("4", "6"): {"公路": 21, "铁路": 24, "水路": 0},
    ("5", "8"): {"公路": 24, "铁路": 26, "水路": 0},
    ("5", "9"): {"公路": 28, "铁路": 24, "水路": 22},
    ("6", "7"): {"公路": 25, "铁路": 0, "水路": 0},
    ("6", "9"): {"公路": 24, "铁路": 24, "水路": 20},
    ("7", "9"): {"公路": 20, "铁路": 25, "水路": 26},
    ("8", "9"): {"公路": 20, "铁路": 0, "水路": 0},
    ("8", "10"): {"公路": 30, "铁路": 24, "水路": 0},
    ("9", "10"): {"公路": 26, "铁路": 0, "水路": 0},
    ("9", "11"): {"公路": 28, "铁路": 25, "水路": 0},
    ("10", "11"): {"公路": 24, "铁路": 26, "水路": 0},
    ("10", "D"): {"公路": 19, "铁路": 24, "水路": 0},
    ("11", "D"): {"公路": 22, "铁路": 0, "水路": 0},
}
#提取qij(k)的值
def get_max_transport_capacity(i, j, k):
    return max_transport_capacity.get((i, j), {}).get(k, 0)


# 节点处最大中转运输能力
max_transfer_capacity = {
    "1": {"公-铁": 20, "铁-水": 25, "公-水": 21},
    "2": {"公-铁": 25, "铁-水": 20, "公-水": 18},
    "3": {"公-铁": 24, "铁-水": 30, "公-水": 22},
    "4": {"公-铁": 30, "铁-水": 25, "公-水": 15},
    "5": {"公-铁": 24, "铁-水": 22, "公-水": 25},
    "6": {"公-铁": 18, "铁-水": 21, "公-水": 20},
    "7": {"公-铁": 20, "铁-水": 25, "公-水": 22},
    "8": {"公-铁": 28, "铁-水": 0, "公-水": 0},
    "9": {"公-铁": 30, "铁-水": 25, "公-水": 26},
    "10": {"公-铁": 24, "铁-水": 0, "公-水": 0},
    "11": {"公-铁": 22, "铁-水": 0, "公-水": 0},
    "D": {"公-铁": 0, "铁-水": 0, "公-水": 0},
}

#提取qi(kl)的值
def get_max_transfer_capacity(i, k, l):
    return max_transfer_capacity.get(i, {}).get(f"{k}-{l}", 0)


# 各节点时间窗设置
time_window = {
    "0-5": {"类型": "软时间窗", "范围": (5, 25)},
    "6-11": {"类型": "软时间窗", "范围": (25, 50)},
    "D": {"类型": "硬时间窗", "范围": (35, 72)},
}
# 提取软时间窗范围
def get_time_window(node):
    if 0 <= node <= 5:
        return time_window["0-5"]["范围"]
    elif 6 <= node <= 11:
        return time_window["6-11"]["范围"]
    elif str(node) in time_window:
        return time_window[str(node)]["范围"]
    else:
        # 返回一个默认的时间窗范围，或者可以引发一个异常
        return (0, 0)

# 提取硬时间窗范围
T_min, T_max = time_window["D"]["范围"]


def total_transport_cost(X, Y, distances, unit_fees, transfer_info, Pe, Pl, time_window):
    """
    计算总运输成本
    """
    # 节点间运输成本
    inter_node_cost = sum(
        X[(i, j, k)] * distances[(i, j)][k] * unit_fees[k]
        for i in nodes for j in nodes for k in transport_modes
        if (i, j) in distances and k in distances[(i, j)]
    )

    # 节点转运成本
    transfer_cost = sum(
        Y[(i, k, l)] * transfer_info[(k, l)]["换装成本"]
        for i in middle_nodes for k, l in transfer_info
    )

    # 货物提前到达的单位仓储成本
    early_arrival_cost = sum(
        Pe * max(time_window[i]["范围"][0] - X[(start_node, i, k)], 0)
        for i in middle_nodes for k in transport_modes
        if i in time_window and time_window[i]["类型"] == "软时间窗"
    )

    # 货物延迟到达的单位惩罚成本
    late_arrival_cost = sum(
        Pl * max(X[(start_node, i, k)] - time_window[i]["范围"][1], 0)
        for i in middle_nodes for k in transport_modes
        if i in time_window and time_window[i]["类型"] == "软时间窗"
    )

    return inter_node_cost + transfer_cost + early_arrival_cost + late_arrival_cost


def total_carbon_emission(X, Y, distances, unit_emissions, transfer_info):
    """
    计算总碳排放量
    """
    # 在途运输碳排放量
    inter_node_emission = sum(
        X[(i, j, k)] * distances[(i, j)][k] * unit_emissions[k]
        for i in nodes for j in nodes for k in transport_modes
        if (i, j) in distances and k in distances[(i, j)]
    )

    # 转换碳排放量
    transfer_emission = sum(
        Y[(i, k, l)] * transfer_info[(k, l)]["碳排放"]
        for i in middle_nodes for k, l in transfer_info
    )

    return inter_node_emission + transfer_emission


def total_transport_time(X, Y, fuzzy_transport_time, transfer_info):
    """
    计算总运输时间，是个三角模糊数，后面需要根据此来定义清晰化后的约束条件
    """
    # 总节点间运输时间
    inter_node_time_l = sum(
        X[(i, j, k)] * fuzzy_transport_time[(i, j)][k][0]
        for i in nodes for j in nodes for k in transport_modes
        if (i, j) in fuzzy_transport_time and k in fuzzy_transport_time[(i, j)]
    )
    inter_node_time_m = sum(
        X[(i, j, k)] * fuzzy_transport_time[(i, j)][k][1]
        for i in nodes for j in nodes for k in transport_modes
        if (i, j) in fuzzy_transport_time and k in fuzzy_transport_time[(i, j)]
    )
    inter_node_time_u = sum(
        X[(i, j, k)] * fuzzy_transport_time[(i, j)][k][2]
        for i in nodes for j in nodes for k in transport_modes
        if (i, j) in fuzzy_transport_time and k in fuzzy_transport_time[(i, j)]
    )

    # 总节点转运时间
    transfer_time_l = sum(
        Y[(i, k, l)] * transfer_info[(k, l)]["换装时间"][0]
        for i in middle_nodes for k, l in transfer_info
    )
    transfer_time_m = sum(
        Y[(i, k, l)] * transfer_info[(k, l)]["换装时间"][1]
        for i in middle_nodes for k, l in transfer_info
    )
    transfer_time_u = sum(
        Y[(i, k, l)] * transfer_info[(k, l)]["换装时间"][2]
        for i in middle_nodes for k, l in transfer_info
    )

    return (
    inter_node_time_l + transfer_time_l, inter_node_time_m + transfer_time_m, inter_node_time_u + transfer_time_u)

def calculate_fuzzy_ti_for_each_node(paths, fuzzy_transport_time, transfer_info):
    """
    计算每个节点的模糊时间变量ti
    """
    # 初始化模糊时间为(0, 0, 0)
    ti = (0, 0, 0)
    ti_values = {paths[0]: ti}

    # 遍历路径中的每一段
    for i in range(len(paths) - 1):
        current_node = paths[i]
        next_node = paths[i + 1]

        # 获取当前段的运输方式
        transport_mode = None
        for mode in fuzzy_transport_time[(current_node, next_node)]:
            transport_mode = mode
            break

        # 获取当前段的模糊运输时间
        segment_fuzzy_time = fuzzy_transport_time[(current_node, next_node)][transport_mode]

        # 更新模糊时间
        ti = tuple(map(sum, zip(ti, segment_fuzzy_time)))

        # 如果存在中转，获取模糊中转时间
        if i < len(paths) - 2:
            next_next_node = paths[i + 2]
            if (transport_mode, next_next_node) in transfer_info:
                transfer_fuzzy_time = transfer_info[(transport_mode, next_next_node)]["换装时间"]
                ti = tuple(map(sum, zip(ti, transfer_fuzzy_time)))

        ti_values[next_node] = ti

    return ti_values


import pulp

# 定义模型
model = pulp.LpProblem("Fuzzy_Opportunity_Constrained_Programming", pulp.LpMinimize)

# 约束10. 定义决策变量
X = pulp.LpVariable.dicts("X", ((i, j, k) for i in nodes for j in nodes for k in transport_modes if (i, j) in distances and k in distances[(i, j)]), 0, 1, pulp.LpBinary)
Y = pulp.LpVariable.dicts("Y", ((i, k, l) for i in middle_nodes for k, l in transfer_info), 0, 1, pulp.LpBinary)

# 定义目标函数

alpha = 0.5  # 权重，可以根据需要调整
model += alpha * total_transport_cost(X, Y, distances, unit_fees, transfer_info, Pe, Pl, time_window) + (1 - alpha) * total_carbon_emission(X, Y, distances, unit_emissions, transfer_info), "Objective"

# 添加约束条件
# 约束7. 流量守恒约束
for node in middle_nodes:
    for k in transport_modes:
        inflow = pulp.lpSum([X[(i, node, k)] for i in nodes if (i, node) in distances and k in distances[(i, node)]])
        outflow = pulp.lpSum([X[(node, j, k)] for j in nodes if (node, j) in distances and k in distances[(node, j)]])
        model += inflow == outflow

# 约束3. 总运输成本约束
model += total_transport_cost(X, Y, distances, unit_fees, transfer_info, Pe, Pl, time_window) <= max_cost

# 约束4. 总碳排放量的约束
model += total_carbon_emission(X, Y, distances, unit_emissions, transfer_info) <= max_emission

# 确保所有货物都被运送到目的地
model += pulp.lpSum([X[(i, end_node, k)] for i in nodes if (i, end_node) in distances and k in distances[(i, end_node)]]) == sum(q_fuzzy)

# 约束5. 货物在两个相邻节点间运输时只能采用一种运输方式，不可拆分
def constraint8(X, i, j):
    return sum([X[i][j][k] for k in transport_modes]) <= 1

# 约束6. 货物在每个节点最多转换一次不同的运输方式
def constraint9(Y, i):
    return sum([sum([Y[i][k][l] for l in transport_modes]) for k in transport_modes]) <= 1

# 约束10. 确保货物运输方式转换的连续性和准确性
def constraint11(X, i, j, k, l):
    return X[i][j][k] + X[i][j][l] >= 2 * Y[i][k][l]

# 解决模型
model.solve()

# 输出结果
for (i, j, k) in X:
    if X[(i, j, k)].varValue > 0:
        print(f"从节点 {i} 到节点 {j} 使用运输方式 {k} 的货物量为: {X[(i, j, k)].varValue}")

for (i, k, l) in Y:
    if Y[(i, k, l)].varValue > 0:
        print(f"在节点 {i} 进行从运输方式 {k} 到 {l} 的换装的货物量为: {Y[(i, k, l)].varValue}")


# 定义模糊软时间窗Ti(E)和Ti(L)
def get_fuzzy_time_window(node):
    time_range = get_time_window(node)
    Ti_E = time_range[0]
    Ti_L = time_range[1]
    return Ti_E, Ti_L

print(get_fuzzy_time_window(7))


# 获取指定节点的模糊时间变量ti[0]、ti[1]、ti[2]
def get_fuzzy_ti_for_node(node, paths, fuzzy_transport_time, transfer_info):
    ti_values = calculate_fuzzy_ti_for_each_node(paths, fuzzy_transport_time, transfer_info)
    ti = ti_values.get(node, (0, 0, 0))
    return ti[0], ti[1], ti[2]

# 定义三角模糊变量T[0]、T[1]、T[2]
def get_total_fuzzy_time(X, Y, fuzzy_transport_time, transfer_info):
    T = total_transport_time(X, Y, fuzzy_transport_time, transfer_info)
    return T[0], T[1], T[2]


# 定义常量和参数
alpha = 0.9  # 置信水平α
beta1 = 0.9  # 置信水平β1
beta2 = 0.9  # 置信水平β2
gamma = 0.9  # 置信水平γ
omega = 0.5  # 权重ω
q_l = ...    # q(l)
q_m = ...    # q(m)
T_min = 35  # 最小时间
T_max = 72  # 最大时间

from scipy.optimize import linprog

# 约束11. f-的定义
def objective_function(X, Y):
    term1 = sum([sum([sum([get_unit_fee(k) * get_distance(i, j, k) * X[i][j][k] for k in transport_modes]) for j in nodes]) for i in nodes])
    term2 = sum([sum([get_transfer_fee(i, k, l)[0] * Y[i][k][l] for l in transport_modes for k in transport_modes]) for i in nodes])
    term3= sum([sum([sum([get_unit_emission(k) * get_distance(i, j, k) * X[i][j][k] for k in transport_modes]) for j in nodes]) for i in nodes])
    term4 = sum([sum([get_transfer_emission(i, k, l)[0] * Y[i][k][l] for l in transport_modes for k in transport_modes]) for i in nodes])
    term5 = sum([max(Ti_E - ti[0], 0) for i in nodes])
    term6 = sum([max(ti[1] - Ti_L, 0) for i in nodes])

    f_minus = omega * ((1 - alpha) * q_l + alpha * q_m) * (term1 + term2 + Pe * term5 * (1 - alpha)  + Pl * term6 * alpha) + (1 - omega) * ((1 - alpha) * q_l + alpha * q_m) * (term3 + term4)

    return f_minus

# 约束12. f-的上界
def constraint_12(f_minus):
    return f_minus <= omega * max_cost + (1 - omega) * max_emission

# 约束13. qij(k)的下界
def constraint_13(i, j, k, X):
    return get_max_transport_capacity(i, j, k) >= X[i][j][k] * (1 - beta1) * q_l + X[i][j][k] * beta1 * q_m

# 约束14. qi(kl)的下界
def constraint_14(i, k, l, Y):
    return get_max_transfer_capacity(i, k, l) >= Y[i][k][l] * (1 - beta2) * q_l + Y[i][k][l] * beta2 * q_m

# 约束15. T的范围
def constraint_15(T):
    return T_min <= (1 - gamma) * T[0] + gamma * T[1] <= T_max

# 约束16. alpha, beta1, beta2, gamma的范围
def constraint_16(alpha, beta1, beta2, gamma):
    return 0 < alpha < 1 and 0 < beta1 < 1 and 0 < beta2 < 1 and 0 < gamma < 1

# 约束17. omega的范围
def constraint_17(omega):
    return 0 <= omega <= 1


