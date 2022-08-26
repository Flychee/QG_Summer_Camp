import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import copy
import sqlalchemy
import pandas as pd
from sqlalchemy import create_engine


class DSG:
    def __init__(self, DP_flag, rc=0.5):
        self.DP_flag = DP_flag
        self.rc = rc
        self.engine = create_engine("mysql+pymysql://root:3751ueoxjwgixjw3913@39.98.41.126:3306/qg_final")
        self.location_address = 'DSG_10_1000_location'  # the name is DSG_x_xxx_location
        self.adjacency_address = 'DSG_10_1000_adjacency'  # the name is DSG_x_xxx_adjacency
        self.npy_path = r'C:\Users\86177\Desktop\QG_Summer_Camp\Last Assessment\数据集\Agents-1000\010log_neat_radiation_1000\010log_neat_radiation_1000.npy'  # the address of the .npy used to create the topology matrix
        self.file = ""
        self.d = 0.3 * rc
        self.G_original = None
        self.G = None

    def G_create(self):
        """
        Create the topology
        :return: A topology, DS: array
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
        if DSG.norm2(self, a) == 0:  # 重合
            return 0
        else:
            if a[1] >= 0:  # 纵坐标大于0
                return np.arccos(np.dot(a, np.array([1, 0])) / (DSG.norm2(self, a)))
            else:
                return 2 * np.pi - np.arccos(np.dot(a, np.array([1, 0])) / (DSG.norm2(self, a)))

    def dst(self, a, b):  # 计算另一直角边长度
        """
        Use the pythagorean theorem to calculate the square edge
        :param a:The hypotenuse, DS: float/double
        :param b:The square, DS: float/double
        :return:Another square, DS: float/double
        """
        return math.fabs(a ** 2 - b ** 2) ** 0.5

    def far_agent(self, num, N_i):  # 选取最远的智能体
        """
        Choose the farthest agent in num_th agent's neighbor sets
        :param num:The index of the agent, DS: int
        :param N_i:one of the num_th agent's neighbor sets, DS: list
        :return:the farthest neighbor's index, DS: int
        """
        max_ = num  # 若无则返回自身
        temp = 0
        for n in N_i:
            if DSG.norm2(self, self.G[1][num] - self.G[1][n]) >= temp:
                max_ = n
                temp = DSG.norm2(self, self.G[1][num] - self.G[1][n])
        return max_

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

    def SDB(self, A):  # 基于SDB的邻接矩阵
        """
        Create the adjacency matrix according to SDB algorithm
        :param A:the adjacency matrix according to the communication distance, DS:array
        :return:the SDB-adjacency matrix, DS: array
        """
        result = np.zeros((np.shape(self.G[1])[0], np.shape(self.G[1])[0]))
        N_num = []  # 存放分扇区后智能体标号
        for num in range(np.shape(self.G[1])[0]):
            temp = {'N_1': [], 'N_2': [], 'N_3': [], 'N_4': []}
            if sum(A[num]) == len(A[num]) - 1:  # 与其他所有智能体连接
                for index in range(np.shape(self.G[1])[0]):
                    if index != num:
                        result[num][index] = 1
            else:
                I = [index for index in range(len(A[num])) if A[num][index] == 1]
                for index in I:
                    rad_temp = DSG.rad(self, a=self.G[1][index] - self.G[1][num])
                    if np.pi / 2 * 0 <= rad_temp < np.pi / 2 * 1:
                        temp['N_1'].append(index)
                    elif np.pi / 2 * 1 <= rad_temp < np.pi / 2 * 2:
                        temp['N_2'].append(index)
                    elif np.pi / 2 * 2 <= rad_temp < np.pi / 2 * 3:
                        temp['N_3'].append(index)
                    elif np.pi / 2 * 3 <= rad_temp < np.pi / 2 * 4:
                        temp['N_4'].append(index)
            N_num.append(temp)
        for num in range(np.shape(self.G[1])[0]):
            T = [0 if len(N_num[num]['N_1']) == 0 else 1,
                 0 if len(N_num[num]['N_2']) == 0 else 1,
                 0 if len(N_num[num]['N_3']) == 0 else 1,
                 0 if len(N_num[num]['N_4']) == 0 else 1]
            if np.dot(np.array(T), np.array([1, -1, 1, -1])) == 0 and sum(T) == 2:
                index_list = [index for index in range(len(T)) if T[index] == 1]
                choose_agent1 = num
                choose_agent2 = num
                max_rad = 0
                num_dict = {'0': 'N_1', '1': 'N_2', '2': 'N_3', '3': 'N_4'}
                for agent1 in N_num[num][num_dict[str(index_list[0])]]:
                    for agent2 in N_num[num][num_dict[str(index_list[1])]]:

                        rad_temp = np.abs(DSG.rad(self, self.G[1][agent1] - self.G[1][num]) -
                                          DSG.rad(self, self.G[1][agent2] - self.G[1][num])) \
                            if np.abs(DSG.rad(self, self.G[1][agent1] - self.G[1][num]) -
                                      DSG.rad(self, self.G[1][agent2] - self.G[1][num])) <= np.pi \
                            else 2 * np.pi - np.abs(DSG.rad(self, self.G[1][agent1] - self.G[1][num]) -
                                                    DSG.rad(self, self.G[1][agent2] - self.G[1][num]))

                        # 排除相减后大于180度的情况
                        if rad_temp >= max_rad:
                            max_rad = rad_temp
                            choose_agent1 = agent1
                            choose_agent2 = agent2
                cooperation = [choose_agent1, choose_agent2]  # 请求合作
            else:
                cooperation = [DSG.far_agent(self, num, N_num[num]['N_1']),
                               DSG.far_agent(self, num, N_num[num]['N_2']),
                               DSG.far_agent(self, num, N_num[num]['N_3']),
                               DSG.far_agent(self, num, N_num[num]['N_4'])]  # 请求合作
            for index in cooperation:
                if index != num:
                    result[num][index] = 1
        return result

    def DSG(self, A, i):  # 基于期望控制输入的子图邻接矩阵
        """
        Create the adjacency matrix of the subgraph of the i according to DSG algorithm
        :param A:the adjacency matrix according to the communication distance, DS:array
        :param i:the index of the agent, DS:int
        :return:the list of the list of i_th neighbors' indexes and the DSG-adjacency matrix,
                DS: list (of list and array)
        """
        I = [index for index in range(len(A)) if A[i][index] == 1]
        result = [I, np.zeros((len(I), len(I)))]  # 最后一行为i
        for index1 in range(len(result[1])):
            for index2 in range(len(result[1])):
                if index1 != index2 \
                        and DSG.norm2(self, self.G[1][result[0][index1]] - self.G[1][result[0][index2]]) <= self.d:
                    result[1][index1][index2] = 1
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

    def CS(self, cpt, i_DSG, i):  # 找出用于约束的邻居集
        """
        find the nearest neighbor in each connected component of the i_th agent
        :param cpt:the connected component sets, DS:list (of lists)
        :param i_DSG:the list of neighbors' indexes and the DSG adjacency matrix, DS: list (of list and matrix)
        :param i: the i_th agent's index, DS: int
        :return:the list of the nearest neighbors, DS: list
        """
        result = []
        temp = []
        for index in range(len(i_DSG[1])):
            for set_list in cpt:
                if index in set_list and set_list not in temp:
                    temp.append(set_list)
        for set_list in temp:
            min_dst = self.rc
            nearest_agent = []
            for element in set_list:
                cal = DSG.norm2(self, self.G[1][i] - self.G[1][i_DSG[0][element]])
                if cal < min_dst:
                    min_dst = cal
                    nearest_agent.clear()
                    nearest_agent.append(element)
                elif cal == min_dst:
                    nearest_agent.append(element)
                for index in nearest_agent:
                    result.append(index)
        return result

    def check_location(self, location, i, j):  # 判断位置在圆内还是圆外
        """
        Check if the location is in the circle
        :param location: the target's location, DS:array
        :param i:i_th agent's location, DS:array
        :param j:j_th agent's location, DS:array
        :return:the numbers of 0 and 1 which use to judge if location is in the circle, DS: array
        """
        x = (i[0] + j[0]) / 2
        y = (i[1] + j[1]) / 2
        if (location[0] - x) ** 2 + (location[1] - y) ** 2 <= (self.rc * 0.5) ** 2:
            return 0
        else:
            return 1

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

    def lct_list(self, location_list, time):
        """
        append the data in the list which will be written into xxx_x_xxx_location
        :param G_original: the original topology, DS :array
        :param location_list:Used to save the data of the topology, DS:list (of dict)
        :param time: The number of iterations, DS: int
        """
        id = time * (len(self.G_original) + 1) + 1
        for element in self.G_original:
            temp = {'id': id, 'x': element[0], 'y': element[1]}
            location_list.append(temp)
            id += 1
        location_list.append({'id': id, 'x': None, 'y': None})

    def lct_write(self, location_list):
        """
        write the data into xxx_x_xxx_location
        :param location_list:Used to save the location of the topology, DS:list (of dict)
        """
        path = r"DSG_Data"
        csv_name = self.location_address + '.csv'
        location = pd.DataFrame(location_list)
        location.to_csv(path + '/' + csv_name, encoding='utf-8')
        conn = self.engine.connect()
        location.to_sql(name=self.location_address, con=conn, if_exists='replace', index=False)
        print("成功写入" + self.location_address)

    def ajc_list(self, A, adjacency_list, time):
        """
        append the data in the list which will be written into xxx_x_xxx_adjacency
        :param A: the adjacency matrix, DS: array
        :param adjacency_list: Used to save the adjacency matrix of the topology, DS:list (of dict)
        :param time: The number of iterations, DS: int
        """
        id = time * (len(A) + 1) + 1
        for element in A:
            index_temp = [index for index in range(len(element)) if element[index] == 1]
            temp = {'id': id, 'ad_index': str(index_temp)[1: -1]}
            adjacency_list.append(temp)
            id += 1
        adjacency_list.append({'id': id, 'ad_index': None})

    def ajc_write(self, adjacency_list):
        """
        write the data into xxx_x_xxx_adjacency
        :param adjacency_list:Used to save the adjacency matrix of the topology, DS:list (of dict)
        """
        path = r"DSG_Data"
        csv_name = self.adjacency_address + '.csv'
        adjacency = pd.DataFrame(adjacency_list)
        adjacency.to_csv(path + '/' + csv_name, encoding='utf-8')
        conn = self.engine.connect()
        adjacency.to_sql(name=self.adjacency_address, con=conn, if_exists='replace', index=False)
        print("成功写入" + self.adjacency_address)

    def main(self):
        """
        main function
        :return: the successful signal, DS: string
        """
        location_list = []
        adjacency_list = []
        DSG.G_create(self)
        time = 0
        while 1:
            DSG.G_merge(self, DSG.A(self, self.G_original, 0.002 * self.rc))
            if len(self.G[0]) == 1:
                DSG.lct_list(self, location_list, time)
                A_rc = DSG.A(self, self.G_original, self.rc)
                DSG.ajc_list(self, A_rc, adjacency_list, time)
                print(self.location_address[: -9] + "聚类完毕")
                break
            A_cnt = DSG.A(self, self.G_original, 0.000001 * self.rc)
            A_rc = DSG.A(self, self.G_original, self.rc)
            DSG.G_merge(self, A_cnt)
            A_trad = DSG.A(self, self.G[1], self.rc)
            DSG.TPfig(self, A_trad, start=-6, end=6)
            DSG.lct_list(self, location_list, time)
            DSG.ajc_list(self, A_rc, adjacency_list, time)
            # SDB算法
            A_SDB = DSG.SDB(self, A_trad)
            # 控制输入U
            U = np.dot(np.dot(np.linalg.pinv(DSG.D(self, A_SDB)), A_SDB) - np.eye(np.shape(A_SDB)[0]), self.G[1])

            # DSG算法
            U_hat = []
            for element in U:
                if DSG.norm2(self, element) == 0:
                    U_hat.append([0, 0])
                else:
                    U_hat.append(min((self.rc - self.d) / (2 * DSG.norm2(self, element)), 1) * element)
            U_hat = np.array(U_hat)  # 期望输入
            lbd = [1 - 0.01 * n for n in range(0, 101)]  # 控制参数
            result = []
            for index in range(len(U_hat)):
                check_bool = False
                i_DSG = DSG.DSG(self, A_trad, index)
                i_cpt = DSG.component(self, i_DSG[1])
                i_CS = DSG.CS(self, i_cpt, i_DSG, index)
                for parameter in lbd:
                    target = self.G[1][index] + parameter * U_hat[index]
                    check = [DSG.check_location(self, target, self.G[1][index], self.G[1][i_DSG[0][neighbor]])
                             for neighbor in i_CS]
                    if sum(check) == 0 and len(check) != 0:
                        result.append(target)
                        check_bool = True
                        break
                if not check_bool:
                    result.append(self.G[1][index])
            self.G[1] = np.array(result)
            for index in range(len(self.G[0])):
                for element in self.G[0][index]:
                    self.G_original[element] = self.G[1][index]
            print(time)
            time += 1
        DSG.lct_write(self, location_list)
        DSG.ajc_write(self, adjacency_list)
        return "SUCCESS"


dsg = DSG(False)
dsg.main()
exit(1)