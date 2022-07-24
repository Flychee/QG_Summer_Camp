import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd

alpha = np.array([float(n) / 10 for n in range(0, 11)])


def TPfig(G, A, number):  # 拓扑绘画(number输入非0值将显示拓扑图),输出邻接矩阵函数
    plt.figure(figsize=(5, 5))
    num = 0
    for i in G[0]:
        num += 1
        A.append([])
    for i in range(num):
        for j in range(num):
            if j == i:
                A[i].append(0)
                continue
            else:
                if (G[0][i] - G[0][j]) ** 2 + (G[1][i] - G[1][j]) ** 2 <= 1:
                    A[i].append(1)
                    Gpic = [[G[0][i], G[0][j]], [G[1][i], G[1][j]]]
                    plt.plot(Gpic[0], Gpic[1], c='k', linewidth=0.1, zorder=1)
                else:
                    A[i].append(0)
    plt.scatter(G[0], G[1], c='r', zorder=2)
    if number:
        plt.show()

def motif(A, M):#得到(3, 3)模体矩阵
    num = 0
    for i in A[0]:
        num += 1
    for n in range(num):
        M.append([])
        for m in range(num):
            M[n].append(0)
    for i in range(num):
        for j in range(num):
            if A[i][j] == 1:
                for k in range(num):
                    if A[i][k] == 1 and k != j:
                        if A[j][k] == 1:
                            M[i][j] += 1


def dst(a, b):  # 计算距离
    return math.fabs(a ** 2 - b ** 2) ** 0.5


L = 10.0
p = 2

# motif函数验证
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(5):
    G_x.append(np.random.uniform(1, 3))
for i in G_x:
    G_y.append(np.random.uniform(1, 3))
G_mt = [G_x, G_y]
A_mt = []
M_mt = []
TPfig(G_mt, A_mt, 1)
motif(A_mt, M_mt)
print(M_mt)"""

#  top-1
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(5):
    G_x.append(np.random.uniform(-L / 2, L / 2))
for i in G_x:
    G_y.append(np.random.uniform(-dst(L/2, i), dst(L/2, i)))
G1 = [G_x, G_y]
A1 = []
M1 = []
TPfig(G1, A1, 0)
motif(A1, M1)"""

#  top-2
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(10):
    for j in range(10):
        for n in range(2):
            G_x.append(np.random.uniform(i, i + 1))
            G_y.append(np.random.uniform(j, j + 1))
G2 = [G_x, G_y]
A2 = []
M2 = []
TPfig(G2, A2, 0)
motif(A2, M2)"""

#  top-3
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
r = L / (2 + 2 ** 0.5)  # 半径
l_set = -L / (2 + 2 * (2 ** 0.5))  # 左圆心坐标
r_set = L / (2 + 2 * (2 ** 0.5))  # 右圆心坐标
for i in range(int(0.5 * 2 * L ** 2)):
    G_x.append(np.random.uniform(l_set - r, l_set + r))
for i in G_x:
    G_y.append(np.random.uniform(- dst(r, i - l_set) + l_set,  dst(r, i - l_set) + l_set))
for i in range(int(0.5 * 2 * L ** 2)):
    G_x.append(np.random.uniform(r_set - r, r_set + r))
for i in G_x[100:]:
    G_y.append(np.random.uniform(- dst(r, i - r_set) + r_set, dst(r, i - r_set) + r_set))
G3 = [G_x, G_y]
A3 = []
M3 = []
TPfig(G3, A3, 0)
motif(A3, M3)"""

#  top-4
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
r_out = L / 2  # 外半径
r_in = 0.6 * r_out  # 内半径
for i in range(int(p * L ** 2)):
    G_x.append(np.random.uniform(-L / 2, L / 2))
for i in G_x:
    if i > r_in or i < -r_in:
        G_y.append(np.random.uniform(-dst(r_out, i), dst(r_out, i)))
    else:
        n = rd.uniform(0, 1)
        if n > 0.5:
            G_y.append(np.random.uniform(dst(r_in, i), dst(r_out, i)))
        else:
            G_y.append(np.random.uniform(-dst(r_out, i), -dst(r_in, i)))
G4 = [G_x, G_y]
A4 = []
M4 = []
TPfig(G4, A4, 0)
motif(A4, M4)"""

#  top-5
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(int(p * L ** 2)):
    n = np.random.uniform(0, 1)
    if n < 0.125:
        G_x.append(np.random.uniform(0, 0.25 * L))
    elif n > 0.875:
        G_x.append(np.random.uniform(0.75 * L, L))
    else:
        G_x.append(np.random.uniform(0.25 * L, 0.75 * L))
for i in G_x:
    if i < L / 2.0:
        G_y.append(np.random.uniform(0, 2 * i))
    else:
        G_y.append(np.random.uniform(0, 2 * (L - i)))
G5 = [G_x, G_y]
A5 = []
M5 = []
TPfig(G5, A5, 0)
motif(A5, M5)"""

#  top-6
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(int(p * L ** 2)):
    n = np.random.uniform(0, 1)
    if n < 0.2:
        G_x.append(np.random.uniform(0, L / 3.0))
    elif n > 0.8:
        G_x.append(np.random.uniform(2.0 * L / 3.0, L))
    else:
        G_x.append(np.random.uniform(L / 3.0, 2.0 * L / 3.0))
for i in G_x:
    if i < L / 3.0 or i > 2.0 * L / 3.0:
        G_y.append(np.random.uniform(L / 3.0, 2.0 * L / 3.0))
    else:
        G_y.append(np.random.uniform(0, L))
G6 = [G_x, G_y]
A6 = []
M6 = []
TPfig(G6, A6, 0)
motif(A6, M6)"""

#  top-7
"""G_x = [0]  # 拓扑图 x 坐标
G_y = [0]  # 拓扑图 y 坐标
for i in range(1, 11):
    a = i * 0.1 * L / 2
    for n in range(0, 4 * i):
        G_x.append(a * np.cos(n / (4 * i) * 2 * np.pi))
        G_y.append(a * np.sin(n / (4 * i) * 2 * np.pi))
G7 = [G_x, G_y]
A7 = []
M7 = []
TPfig(G7, A7, 0)
motif(A7, M7)"""

#  top-8
"""G_x = [0]  # 拓扑图 x 坐标
G_y = [0]  # 拓扑图 y 坐标
a = L / 2 * 1 / 10
for i in range(1, 11):
    for n in range(0, 20):
        G_x.append(a * i * np.cos(n / 20 * 2 * np.pi))
        G_y.append(a * i * np.sin(n / 20 * 2 * np.pi))
G8 = [G_x, G_y]
A8 = []
M8 = []
TPfig(G8, A8, 0)
motif(A8, M8)"""

#  top-9
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(0, 15):
    a = i / 15 * L
    for n in range(0, 15):
        b = n / 15 * L
        G_x.append(a)
        G_y.append(b)
G9 = [G_x, G_y]
A9 = []
M9 = []
TPfig(G9, A9, 0)
motif(A9, M9)"""

#  top-10
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
set_dst = L / 27
h = 3 ** 0.5 * set_dst
for i in range(0, 27):
    for n in range(0, 8):
        if i % 2 == 0:
            G_x.append(i * set_dst)
            G_y.append(2 * n * h)
        if i % 2 == 1:
            G_x.append(i * set_dst)
            G_y.append((2 * n + 1) * h)
G10 = [G_x, G_y]
A10 = []
M10 = []
TPfig(G10, A10, 0)
motif(A10, M10)"""
