import numpy as np
from matplotlib import pyplot as plt
import copy
import matplotlib.animation as ani

# 追随者定义
A = [[0.0, 10.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 0.0]]  # 追随者邻接矩阵

k = [0.0, 10.0, 10.0]  # 与领导者连接关系

x = [[6.0, 60.0], [10.0, 40.0], [16.0, 70.0]]  # 位置信息
v = [[10.0, 5.0], [8.0, 4.0], [9.0, 3.0]]  # 速度信息
a = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  # 加速度信息
r = [[-15.0, 0.0], [-10.0, 0.0], [-5.0, 0.0]]  # 与领导者期望相对位置信息
X = [[[6.0, 60.0]], [[10.0, 40.0]], [[16.0, 70.0]]]  # 记录位置变化
V = [ [[10.0, 5.0]], [[8.0, 4.0]], [[9.0, 3.0]] ]  # 记录速度变化
# 领导者定义
x_l = [20.0, 50.0]
v_l = [6.0, 0.0]
X_l = [[20.0, 50.0]]  # 记录位置变化


def update(A, k, x, v,a, X, V,  x_l, v_l, X_l, r, time):
    for i in range(0, 3):
        x[i][0] = x[i][0] + v[i][0] * time + 0.5 * a[i][0] * time ** 2.0
        x[i][1] = x[i][1] + v[i][1] * time + 0.5 * a[i][1] * time ** 2.0
        v[i][0] = v[i][0] + a[i][0] * time
        v[i][1] = v[i][1] + a[i][1] * time
        X[i].append(copy.deepcopy(x[i]))
        V[i].append(copy.deepcopy(v[i]))
        a[i] = [0, 0]  # 加速度信息清零
        for j in range(0, 3):
            a[i][0] -= A[i][j] * (x[i][0] - x[j][0] - (r[i][0] - r[j][0]) + v[i][0] - v[j][0])
            a[i][1] -= A[i][j] * (x[i][1] - x[j][1] - (r[i][1] - r[j][1]) + v[i][1] - v[j][1])
        a[i][0] += 0.0 - k[i] * (x[i][0] - x_l[0] - r[i][0] + v[i][0] - v_l[0])
        a[i][1] += 0.0 - k[i] * (x[i][1] - x_l[1] - r[i][1] + v[i][1] - v_l[1])

    x_l[0] = x_l[0] + v_l[0] * time
    x_l[1] = x_l[1] + v_l[1] * time
    X_l.append(copy.deepcopy(x_l))  # 领导者更新信息

for i in range(15000):
    update(A, k, x, v, a, X, V, x_l, v_l, X_l, r, 0.001)

# 画图准备
x_l_pic = []
y_l_pic = []
x_pic = [[], [], []]
y_pic = [[], [], []]
for i in range(0, 3):
    for n in X[i]:
        x_pic[i].append(n[0])
        y_pic[i].append(n[1])

for i in X_l:
    x_l_pic.append(i[0])
    y_l_pic.append(i[1])

def draw(n):
    c = ['g', 'y', 'c']
    plt.cla()
    plt.plot(x_l_pic[:n*50], y_l_pic[:n*50], c="red")
    plt.annotate('', xy=[xx + 0.2*vv for xx, vv in zip(X_l[n*50], v_l)], xytext=(X_l[n * 50]),
                 arrowprops=dict(connectionstyle="arc3", facecolor="red"))

    for i in range(0, 3):
        plt.plot(x_pic[i][:n * 50], y_pic[i][:n * 50], c=c[i])
        plt.annotate('', xy=[xx + 0.2*vv for xx, vv in zip(X[i][n * 50], V[i][n * 50])], xytext=(X[i][n * 50]),
                     arrowprops=dict(connectionstyle="arc3", facecolor=c[i]))
    plt.xlim(0, x_l_pic[n * 50]+10)
    plt.legend(["Leader", "Vehicle i", "Vehicle i+1", "Vehicle i+2"])
    plt.tight_layout()

fig = plt.figure()
an = ani.FuncAnimation(fig, draw, 100)
an.save("fly.gif")
plt.show()

