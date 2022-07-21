import random

import numpy as np
import matplotlib.pyplot as plt
import random as rd

a = np.array([float(n) / 10 for n in range(0, 11)])


def TPfig(G):  # 拓扑绘画函数
    plt.figure(figsize=(5, 5))
    num = 0
    for i in G[0]:
        num += 1
    for i in range(num):
        for j in range(num):
            if j == i:
                continue
            else:
                if (G[0][i] - G[0][j]) ** 2 + (G[1][i] - G[1][j]) ** 2 <= 1:
                    Gpic = [[G[0][i], G[0][j]], [G[1][i], G[1][j]]]
                    plt.plot(Gpic[0], Gpic[1], c='k', linewidth=0.1, zorder=1)
    plt.scatter(G[0], G[1], c='r', zorder=2)
    plt.show()



L = 10.0
p = 2

#  top-1
G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(p * int(L ** 2)):
    G_x.append(np.random.randint(-L / 2 * 1000, L / 2 * 1000) / 1000)
for i in G_x:
    G_y.append(np.random.randint(- float(int((((L / 2) ** 2 - i ** 2) ** 0.5) * 1000)),
                                 float(int((((L / 2) ** 2 - i ** 2) ** 0.5) * 1000))) / 1000)
G1 = [G_x, G_y]
# TPfig(G1)

#  top-2
G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(10):
    for j in range(10):
        for n in range(2):
            G_x.append(np.random.randint(1 * i * 1000, 1 * (i + 1) * 1000) / 1000)
            G_y.append(np.random.randint(1 * j * 1000, 1 * (j + 1) * 1000) / 1000)
G2 = [G_x, G_y]
# TPfig(G2)

#  top-3
G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
r = L/(2 + 2 ** 0.5)  # 半径
l_set = -L/(2 + 2 * (2 ** 0.5))  #左圆心坐标
r_set = L/(2 + 2 * (2 ** 0.5))  #右圆心坐标
for i in range(int(0.5 * 2 * L ** 2)):
    G_x.append(np.random.randint((l_set - r) * 10000, (l_set + r) * 10000) / 10000)
for i in G_x:
    G_y.append(np.random.randint((int(-(r ** 2 - (i - l_set) ** 2) ** 0.5 + l_set) * 10000),
                                 (int(-(r ** 2 - (i - l_set) ** 2) ** 0.5 - l_set) * 10000)) / 10000)
for i in range(int(0.5 * 2 * L ** 2)):
    G_x.append(np.random.randint((r_set - r) * 100, (r_set + r) * 10000) / 10000)
for i in G_x[100:]:
    G_y.append(np.random.randint((int((r ** 2 - (i - r_set) ** 2) ** 0.5 - r_set) * 10000),
                                 (int((r ** 2 - (i - r_set) ** 2) ** 0.5 + r_set) * 10000)) / 10000)

G3 = [G_x, G_y]
TPfig(G3)
#  top-4