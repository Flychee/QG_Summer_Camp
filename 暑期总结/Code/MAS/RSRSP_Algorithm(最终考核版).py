import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import copy
import sqlalchemy
import pandas as pd
from sqlalchemy import create_engine


class RSRSP:
    def __init__(self, DP_flag, rc=0.5, alpha1=0.1, alpha2=0.01):
        self.DP_flag = DP_flag
        self.rc = rc
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.engine = create_engine("mysql+pymysql://root:3751ueoxjwgixjw3913@39.98.41.126:3306/qg_final")
        self.location_address = 'RSRSP_10_1000_location'  # the name is RSRSP_x_xxx_location
        self.adjacency_address = 'RSRSP_10_1000_adjacency'  # the name is RSRSP_x_xxx_adjacency
        self.npy_path = r'C:\Users\86177\Desktop\QG_Summer_Camp\Last Assessment\Algorithm\数据集\Agents-1000\010log_neat_radiation_1000\010log_neat_radiation_1000.npy'  # the address of the .npy used to create the topology matrix
        self.file = ""
        self.G_original = None
        self.G = None

    def G_create(self):
        """
        Create the topology
        """
        self.G_original = np.load(self.npy_path, allow_pickle=True)

    def G_merge(self, A_cnt):
        """
        merge the topology matrix
        :param A_cnt: the adjacency matrix with 0.05 * rc, DS: array
        :return: a more simple matrix , DS: array
        """
        self.G = [[], []]
        delete = []
        for index1 in range(len(A_cnt)):
            if index1 not in delete:
                temp = [index1]
                delete.append(index1)
                for index2 in range(len(A_cnt)):
                    if index2 not in delete and A_cnt[index1][index2] == 1:
                        temp.append(index2)
                        delete.append(index2)
                self.G[0].append(temp)
        for element in self.G[0]:
            self.G[1].append(self.G_original[element[0]])
        self.G[1] = np.array(self.G[1])

    def norm2(self, a):  # 2范数
        """
        Calculate the 2-norm
        :param a: A vector, DS:array
        :return: the 2-norm of the vector, DS:float/double
        """
        return np.linalg.norm(a)

    def rad(self, a):  # 弧度
        """
        Calculate the radian between the vector and x_axis
        :param a: A vector, DS:array
        :return: the radian between the vector and x_axis, DS:float/double
        """
        if RSRSP.norm2(self, a) == 0:  # 重合
            return 0
        else:
            if a[1] >= 0:  # 纵坐标大于0
                return np.arccos(np.dot(a, np.array([1, 0])) / (RSRSP.norm2(self, a)))
            else:
                return 2 * np.pi - np.arccos(np.dot(a, np.array([1, 0])) / (RSRSP.norm2(self, a)))

    def dst(self, a, b):  # 计算另一直角边长度
        """
        Use the pythagorean theorem to calculate the square edge
        :param a:The hypotenuse, DS: float/double
        :param b:The square, DS: float/double
        :return:Another square, DS: float/double
        """
        return math.fabs(a ** 2 - b ** 2) ** 0.5

    def sim_agent(self, num, N_i):  # 选取最远的智能体
        """
        Choose the nearest agent in num_th agent's neighbor sets
        :param num:The index of the agent, DS: int
        :param N_i:one of the num_th agent's neighbor sets, DS: list
        :return:the farthest neighbor's index, DS: int
        """
        min_ = num  # 若无则返回自身
        temp = self.rc
        for n in N_i:
            if RSRSP.norm2(self, self.G[1][num] - self.G[1][n]) <= temp:
                min_ = n
                temp = RSRSP.norm2(self, self.G[1][num] - self.G[1][n])
        return min_

    def D(self, A):  # 求度矩阵
        """
        Create the degree matrix according to adjacency matrix
        :param A:topology's adjacency matrix, DS:array
        :return:the degree matrix, DS: array
        """
        result = np.zeros(np.shape(A))
        for i in range(len(A)):
            result[i][i] = np.sum(A[i])
        return result

    def A(self, G, r):  # 根据通信距离构建邻接矩阵
        """
        Create the adjacency matrix according to topology and the communication distance
        :param G: the topology matrix
        :param r:the communication distance, DS:float/double
        :return:the adjacency matrix, DS: array
        """
        result = np.zeros((np.shape(G)[0], np.shape(G)[0]))
        for i in range(len(G)):
            for j in range(len(G)):
                if ((G[i][0] - G[j][0]) ** 2 + (G[i][1] - G[j][1]) ** 2) <= r ** 2 \
                        and i != j:
                    result[i][j] = 1
        return np.array(result)

    def TPfig(self, A, start, end):  # 拓扑绘画
        """
        draw  the figure of the topology
        :param A:the adjacency of the topology, DS:array
        :param start:the start point of the axis, DS:int
        :param end:the end point of the axis, DS:int
        """
        plt.figure(figsize=(5, 5))
        plt.xlim(start, end)
        plt.ylim(start, end)
        for i in range(len(self.G[1])):
            for j in range(len(self.G[1])):
                if A[i][j] == 1:
                    Gpic = [[self.G[1][i][0], self.G[1][j][0]], [self.G[1][i][1], self.G[1][j][1]]]
                    plt.plot(Gpic[0], Gpic[1], c='k', linewidth=0.1, zorder=1)
        plt.scatter(self.G[1].T[0], self.G[1].T[1], c='r', zorder=2)
        # plt.savefig('time_{}.jpg'.format(time))
        plt.show()

    def MAS(self, A, alpha):  # 一致性算法
        """
        The consensus algorithm
        :param alpha: the parameter that controls the rate of convergence, DS: array
        :param A:the adjacency, DS: array
        :return: the location of the topology at next time, DS : array
        """
        return np.dot(alpha * (A - RSRSP.D(self, A)) + np.eye(np.shape(A)[0]), self.G[1])

    def RSRSP(self, A_trad):  # 基于RSRSP的邻接矩阵
        """
        Create the adjacency matrix according to the RSRSP algorithm
        :param A_trad:the adjacency matrix with rc, DS : array
        :return: the RSRSP adjacency matrix, DS : array
        """
        result = np.zeros((np.shape(self.G[1])[0], np.shape(self.G[1])[0]))
        beta = [n * 0.5 * np.pi / 360 for n in range(0, 360, 8)]
        for num in range(np.shape(self.G[1])[0]):
            N_num = []  # 存放分扇区后智能体标号
            var_beta = []
            I = [index for index in range(len(A_trad[num])) if A_trad[num][index] == 1]
            for b in beta:
                temp = {'N_1': [], 'N_2': [], 'N_3': [], 'N_4': []}
                for index in I:
                    rad_temp = RSRSP.rad(self, self.G[1][index] - self.G[1][num]) - b
                    if np.pi / 2 * 0 <= rad_temp < np.pi / 2 * 1:
                        temp['N_1'].append(index)
                    if np.pi / 2 * 1 <= rad_temp < np.pi / 2 * 2:
                        temp['N_2'].append(index)
                    if np.pi / 2 * 2 <= rad_temp < np.pi / 2 * 3:
                        temp['N_3'].append(index)
                    if np.pi / 2 * 3 <= rad_temp < np.pi / 2 * 4 \
                            or np.pi / 2 * (-1) <= rad_temp < np.pi / 2 * 0:  # 注意第一象限减去beta后存在负数的情况
                        temp['N_4'].append(index)
                N_num.append(temp)
                var_beta.append(np.var([len(N_num[-1]['N_1']), len(N_num[-1]['N_2']),
                                        len(N_num[-1]['N_3']), len(N_num[-1]['N_4'])]))
            min_index = var_beta.index(min(var_beta))
            cooperation = [RSRSP.sim_agent(self, num, N_num[min_index]['N_1']),
                           RSRSP.sim_agent(self, num, N_num[min_index]['N_2']),
                           RSRSP.sim_agent(self, num, N_num[min_index]['N_3']),
                           RSRSP.sim_agent(self, num, N_num[min_index]['N_4'])]  # 请求合作
            for index in cooperation:
                if index != num:
                    result[num][index] = 1
                    result[index][num] = 1
        return result

    def component(self, A):  # 生成连通分支
        """
        find all connected components in the topology
        :param A:the adjacency matrix, DS:array
        :return:the list of connected components, DS: list (of lists)
        """
        A_sp = A + np.eye(np.shape(A)[0])
        result = []
        delete = copy.deepcopy(set([]))
        for row in range(len(A_sp)):
            if row not in delete:
                temp = [num for num in range(len(A_sp)) if A_sp[row][num] == 1]
                result.append(copy.deepcopy(set(temp)))
                delete.add(row)
                for index in range(len(A_sp)):
                    if index not in delete:
                        temp = [num for num in range(len(A_sp)) if A_sp[index][num] == 1]
                        if not result[-1].isdisjoint(set(temp)):
                            result[-1] = result[-1] | set(temp)
                            delete.add(index)
        end_result = []  # 再次清洗
        for element in result:
            if len(end_result) == 0:
                end_result.append(element)
            else:
                for index in range(len(end_result)):
                    if not end_result[index].isdisjoint(element):
                        end_result[index] = end_result[index] | element
                        break
                    elif index == len(end_result) - 1:
                        end_result.append(element)
        return end_result

    def lct_list(self, location_list, time):
        """
        append the data in the list which will be written into xxx_x_xxx_location
        :param location_list:Used to save the data of the topology, DS:list (of dict)
        :param time: The number of iterations, DS: int
        """
        id = time * (len(self.G_original) + 1)
        for element in self.G_original:
            temp = {'id': id, 'x': element[0], 'y': element[1]}
            location_list.append(temp)
            id += 1
        location_list.append({'id': id, 'x': None, 'y': None})


    def ajc_list(self, A, adjacency_list, time):
        """
        append the data in the list which will be written into xxx_x_xxx_adjacency
        :param A: the adjacency matrix, DS: array
        :param adjacency_list: Used to save the adjacency matrix of the topology, DS:list (of dict)
        :param time: The number of iterations, DS: int
        """
        id = time * (len(A) + 1)
        for element in A:
            index_temp = [index for index in range(len(element)) if element[index] == 1]
            temp = {'id': id, 'ad_index': str(index_temp)[1: -1]}
            adjacency_list.append(temp)
            id += 1
        adjacency_list.append({'id': id, 'ad_index': None})

    def file_write(self, location_list, adjacency_list):
        """
        write the data into xxx_x_xxx_adjacency
        :param location_list: Used to save the location matrix of the topology, DS:list (of dict)
        :param adjacency_list:Used to save the adjacency matrix of the topology, DS:list (of dict)
        """
        path = r"RSRSP_Data"

        lct_csv_name = self.location_address + '.csv'
        location = pd.DataFrame(location_list)
        location.to_csv(path + '/' + lct_csv_name, encoding='utf-8')

        ajc_csv_name = self.adjacency_address + '.csv'
        adjacency = pd.DataFrame(adjacency_list)
        adjacency.to_csv(path + '/' + ajc_csv_name, encoding='utf-8')

        conn = self.engine.connect()

        location.to_sql(name=self.location_address, con=conn, if_exists='replace', index=False)
        print("成功写入" + self.location_address)

        adjacency.to_sql(name=self.adjacency_address, con=conn, if_exists='replace', index=False)
        print("成功写入" + self.adjacency_address)

    def main(self, num=3):
        """
        The main function
        :param num: Choose the algorithm, DS: int
        :return: the successful signal, DS: string
        """
        location_list = []
        adjacency_list = []
        time = 0
        RSRSP.G_create(self)
        while 1:
            RSRSP.G_merge(self, RSRSP.A(self, self.G_original, 0.05 * self.rc))  # 收敛判定精度
            test1 = self.G
            RSRSP.G_merge(self, RSRSP.A(self, self.G_original, self.rc))  # 通信距离
            test2 = self.G
            if len(test1[0]) == len(test2[0]):
                RSRSP.lct_list(self, location_list, time)
                A_rc = RSRSP.A(self, self.G_original, self.rc)
                RSRSP.ajc_list(self, A_rc, adjacency_list, time)
                print(self.location_address[: -9] + "聚类完毕")
                break
            A_cnt = RSRSP.A(self, self.G_original, 0.000001 * self.rc)  # 去重精度
            A_rc = RSRSP.A(self, self.G_original, self.rc)
            RSRSP.G_merge(self, A_cnt)
            A_trad = RSRSP.A(self, self.G[1], self.rc)
            if time % 150 == 0 and time != 0:
                RSRSP.TPfig(self, A_trad, start=-6, end=6)
            RSRSP.lct_list(self, location_list, time)
            RSRSP.ajc_list(self, A_rc, adjacency_list, time)

            if num == 1:  # 使用SAN算法
                self.G[1] = RSRSP.MAS(self, A_trad, self.alpha2)

            elif num == 2:  # 使用RSRSP算法
                A_rsrsp = RSRSP.RSRSP(self, A_trad)
                self.G[1] = RSRSP.MAS(self, A_rsrsp, self.alpha1)

            elif num == 3:  # 改良算法
                if len(RSRSP.component(self, A_rc)) == len(test2[0]):  # SAN算法
                    print('SAN')
                    self.G[1] = RSRSP.MAS(self, A_trad, self.alpha2)

                else:  # RSRSP算法
                    print('RSRSP')
                    A_rsrsp = RSRSP.RSRSP(self, A_trad)
                    self.G[1] = RSRSP.MAS(self, A_rsrsp, self.alpha1)

            for index in range(len(self.G[0])):
                for element in self.G[0][index]:
                    self.G_original[element] = self.G[1][index]
            print(time)
            time += 1
        RSRSP.file_write(self, location_list, adjacency_list)
        return "SUCCESS"


rsrsp = RSRSP(False)
rsrsp.main()
exit(1)