import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import copy


def norm2(a):  # 2范数
    return np.linalg.norm(a)


def rad(a):  # 弧度
    if a[0] >= 0:
        return np.arccos(np.dot(a, np.array([1, 0])) / (norm2(a)))
    else:
        return 2 * np.pi - np.arccos(np.dot(a, np.array([1, 0])) / (norm2(a)))


def dst(a, b):  # 计算另一直角边长度
    return math.fabs(a ** 2 - b ** 2) ** 0.5


def sim_agent(G_T, num, N_i):  # 选取最相似的智能体（距离最近）
    min_ = num  # 若无则返回自身
    temp = 1
    for n in N_i:
        if norm2(G_T[num] - G_T[n]) <= temp:
            min_ = n
            temp = norm2(G_T[num] - G_T[n])
    return min_


def TPfig(G, number, start, end):  # 拓扑绘画(number输入非0值将显示拓扑图),输出邻接矩阵
    plt.figure(figsize=(5, 5))
    plt.xlim(start, end)
    plt.ylim(start, end)
    num = 0
    result = []
    for i in G[0]:
        num += 1
        result.append(copy.deepcopy([]))
    for i in range(num):
        for j in range(num):
            if j == i:
                result[i].append(0)
                continue
            else:
                if ((G[0][i] - G[0][j]) ** 2 + (G[1][i] - G[1][j]) ** 2) <= 1:
                    result[i].append(1)
                    Gpic = [[G[0][i], G[0][j]], [G[1][i], G[1][j]]]
                    plt.plot(Gpic[0], Gpic[1], c='k', linewidth=0.1, zorder=1)
                else:
                    result[i].append(0)
    plt.scatter(G[0], G[1], c='r', zorder=2)
    if number:
        plt.show()
    return np.array(result)


def D(A):   # 求度矩阵
    result = np.zeros(np.shape(A))
    for i in range(len(A)):
        result[i][i] = np.sum(A[i])
    return result


def MAS(G, A, alpha):   # 一致性算法
    return np.array([np.dot((alpha * (A - D(A)) + np.eye(np.shape(A)[0])), G[0]),
                     np.dot((alpha * (A - D(A)) + np.eye(np.shape(A)[0])), G[1])])


def RSRSP(G_T, A_trad):  # 基于RSRSP的邻接矩阵
    result = np.zeros((np.shape(G_T)[0], np.shape(G_T)[0]))
    beta = [n * 0.5 * np.pi / 90 for n in range(0, 90)]
    for num in range(np.shape(G_T)[0]):
        N_num = []  # 存放分扇区后智能体标号
        var_beta = []
        I = [index for index in range(len(A_trad[num])) if A_trad[num][index] == 1]
        for b in beta:
            temp = {'N_1': [], 'N_2': [], 'N_3': [], 'N_4': []}
            for index in I:
                rad_temp = rad(G_T[index] - G_T[num]) - b
                if np.pi / 2 * 0 <= rad_temp < np.pi / 2 * 1:
                    temp['N_1'].append(index)
                if np.pi / 2 * 1 <= rad_temp < np.pi / 2 * 2:
                    temp['N_2'].append(index)
                if np.pi / 2 * 2 <= rad_temp < np.pi / 2 * 3:
                    temp['N_3'].append(index)
                if np.pi / 2 * 3 <= rad_temp < np.pi / 2 * 4:
                    temp['N_4'].append(index)
            N_num.append(temp)
            var_beta.append(np.var([len(N_num[-1]['N_1']), len(N_num[-1]['N_2']),
                            len(N_num[-1]['N_3']), len(N_num[-1]['N_4'])]))
        min_index = var_beta.index(min(var_beta))
        cooperation = [sim_agent(G_T, num, N_num[min_index]['N_1']),
                       sim_agent(G_T, num, N_num[min_index]['N_2']),
                       sim_agent(G_T, num, N_num[min_index]['N_3']),
                       sim_agent(G_T, num, N_num[min_index]['N_4'])]  # 请求合作
        for index in cooperation:
            if index != num:
                result[num][index] = 1
                result[index][num] = 1
    return result


#基本参数
alpha1 = 0.1
alpha2 = 0.01
R = 3
# 拓扑
G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(120):
    G_x.append(np.random.uniform(-R, R))
for i in G_x:
    G_y.append(np.random.uniform(-dst(R, i), dst(R, i)))
G = np.array([G_x, G_y])
while 1:
    A_trad = TPfig(G, 1, -4, 4)
    num = 0
    for i in A_trad:
        if sum(i) == len(A_trad) - 1:
            num += 1
    if num == len(A_trad):  # SAN算法
        G = MAS(G, A_trad, alpha2)
    else:  # RSRSP算法
        A_rsrsp = RSRSP(G.T, A_trad)
        G = MAS(G, A_rsrsp, alpha1)