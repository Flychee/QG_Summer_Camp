import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import copy


def TPfig(G, number, start, end):  # 拓扑绘画(number输入非0值将显示拓扑图),输出邻接矩阵函数
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

def I(shape):  # 对角1矩阵
    result = []
    for i in range(shape):
        result.append(copy.deepcopy([]))
        for j in range(shape):
            result[-1].append(0)
    for i in range(shape):
        result[i][i] = 1
    return np.array(result)


def motif(A):  # (3, 3)模体矩阵
    num = 0
    result = []
    for i in A[0]:
        num += 1
    for n in range(num):
        result.append(copy.deepcopy([]))
        for m in range(num):
            result[n].append(0)
    for i in range(num):
        for j in range(num):
            if A[i][j] == 1:
                for k in range(num):
                    if A[i][k] == 1 and k != j:
                        if A[j][k] == 1:
                            result[i][j] += 1
    return np.array(result)


def W(alpha, A, M):  # 加权混合矩阵
    return (1 - alpha) * M + alpha * A


def W_r(W):  # 加权矩阵非0元素皆化为倒数
    result = []
    for a in W:
        result.append(copy.deepcopy([]))
        for b in a:
            if b == 0:
                result[-1].append(0)
            else:
                result[-1].append(1 / b)
    return np.array(result)

def D(M):   # 求度矩阵
    result = np.zeros(np.shape(M))
    for i in range(len(M)):
        result[i][i] = np.sum(M[i])
    return result


def L_r(A, W_r):  # 基于W_r的拉普拉斯矩阵
    return D(A) - np.dot(np.dot(D(A), np.linalg.pinv(D(W_r))), W_r)


def epsilon(A):  # 参数计算
    return 1 / np.shape(A)[0]


def MWMS_S(G, L_):
    return [np.dot(I(np.shape(L_)[0]) - epsilon(L_) * L_, G[0]),
            np.dot(I(np.shape(L_)[0]) - epsilon(L_) * L_, G[1])]


def MWMS_J(G, W_r):
    return [np.dot(np.dot(np.linalg.pinv(I(np.shape(W_r)[0]) + D(W_r)), I(np.shape(W_r)[0]) + W_r), G[0]),
            np.dot(np.dot(np.linalg.pinv(I(np.shape(W_r)[0]) + D(W_r)), I(np.shape(W_r)[0]) + W_r), G[1])]


def dst(a, b):  # 计算距离
    return math.fabs(a ** 2 - b ** 2) ** 0.5


def func(G, num, alpha, start, end):
    G_temp = G
    while 1:
        A_temp = TPfig(G_temp, 1, start, end)
        M_temp = motif(A_temp)
        W_temp = W(alpha, A_temp, M_temp)
        W_r_temp = W_r(W_temp)
        L_r_temp = L_r(A_temp, W_r_temp)
        if num != 0:
            G_temp = MWMS_J(G_temp, W_r_temp)
        if num == 0:
            G_temp = MWMS_S(G_temp, L_r_temp)


L = 10.0
p = 2
alpha = np.array([float(n) / 10 for n in range(0, 11)])

# motif函数验证(论文figure3)
"""A_mt = [[0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0]]
M_mt = motif(A_mt)
print(M_mt)"""

#  top-1
"""G_x = []  # 拓扑图 x 坐标
G_y = []  # 拓扑图 y 坐标
for i in range(200):
    G_x.append(np.random.uniform(-L / 2, L / 2))
for i in G_x:
    G_y.append(np.random.uniform(-dst(L/2, i), dst(L/2, i)))
G1 = [np.array(G_x), np.array(G_y)]
while 1:
    A1 = TPfig(G1, 1, -5, 5)
    M1 = motif(A1)
    W1 = W(alpha[1], A1, M1)
    W_r1 = W_r(W1)
    L_r1 = L_r(A1, W_r1)
    G1 = MWMS_J(G1, W_r1)"""


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
G_x = [0]  # 拓扑图 x 坐标
G_y = [0]  # 拓扑图 y 坐标
for i in range(1, 11):
    a = i * 0.1 * L / 2
    for n in range(0, 4 * i):
        G_x.append(a * np.cos(n / (4 * i) * 2 * np.pi))
        G_y.append(a * np.sin(n / (4 * i) * 2 * np.pi))
G7 = [G_x, G_y]
func(G7, 0, alpha[9], -5, 5)


"""#  top-8
G_x = [0]  # 拓扑图 x 坐标
G_y = [0]  # 拓扑图 y 坐标
a = L / 2 * 1 / 10
for i in range(1, 11):
    for n in range(0, 20):
        G_x.append(a * i * np.cos(n / 20 * 2 * np.pi))
        G_y.append(a * i * np.sin(n / 20 * 2 * np.pi))
G8 = [G_x, G_y]
func(G8, 1, alpha[4], -5, 5)"""

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
func(G9, 1, alpha[5], 0, 10)"""

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
