import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import copy
import sqlalchemy
import pandas as pd
from sqlalchemy import create_engine


class MWMS_S:
    def __init__(self, DP_flag, rc=0.5, alpha=0.5):
        self.DP_flag = DP_flag
        self.rc = rc
        self.alpha = alpha
        self.engine = create_engine("mysql+pymysql://root:3751ueoxjwgixjw3913@39.98.41.126:3306/qg_final")
        self.location_address = 'MWMS_S_10_200_location'  # the name is MWMS_S_x_xxx_location
        self.adjacency_address = 'MWMS_S_10_200_adjacency'  # the name is MWMS_S_x_xxx_adjacency
        self.npy_path = \
            r'C:\Users\86177\Desktop\QG_Summer_Camp\Last Assessment\Algorithm\数据集\Agents-200\010log_neat_radiation_200\010log_neat_radiation_200.npy'  # the address of the .npy used to create the topology matrix
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

    def TPfig(self, A, start, end, time):  # 拓扑绘画
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
        plt.savefig('time_{}.jpg'.format(time))
        plt.show()

    def motif(self, A):  # (3, 3)模体矩阵
        num = 0
        result = np.zeros((np.shape(A)[0], np.shape(A)[0]))
        for i in range(num):
            for j in range(num):
                if A[i][j] == 1:
                    for k in range(num):
                        if A[i][k] == 1 and k != j:
                            if A[j][k] == 1:
                                result[i][j] += 1
        return np.array(result)

    def W_J(self, A, M):  # MWMS-J加权混合矩阵
        return (1 - self.alpha) * A + self.alpha * M

    def W_r(self, W):  # 加权矩阵非0元素皆化为倒数
        result = []
        for a in W:
            result.append(copy.deepcopy([]))
            for b in a:
                if b == 0:
                    result[-1].append(0)
                else:
                    result[-1].append(1 / b)
        return np.array(result)

    def L_r(self, A, W_r):  # 基于W_r的拉普拉斯矩阵
        return MWMS_S.D(self, A) - np.dot(np.dot(MWMS_S.D(self, A), np.linalg.pinv(MWMS_S.D(self, W_r))), W_r)

    def MWMS_S(self, L_):
        self.G[1] = np.dot(np.eye(np.shape(L_)[0]) - (1 / np.shape(L_)[0]) * L_, self.G[1])

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
        id = time * (len(self.G_original) + 1) + 1
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
        id = time * (len(A) + 1) + 1
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
        path = r"MWMS_S_Data"

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

    def main(self):
        """
        The main function
        :param num: Choose the algorithm, DS: int
        :return: the successful signal, DS: string
        """
        location_list = []
        adjacency_list = []
        time = 0
        MWMS_S.G_create(self)
        while 1:
            MWMS_S.G_merge(self, MWMS_S.A(self, self.G_original, 0.05 * self.rc))  # 收敛判定精度
            test1 = self.G
            MWMS_S.G_merge(self, MWMS_S.A(self, self.G_original, self.rc))  # 通信距离
            test2 = self.G
            if len(test1[0]) == len(test2[0]):
                MWMS_S.lct_list(self, location_list, time)
                A_rc = MWMS_S.A(self, self.G_original, self.rc)
                MWMS_S.ajc_list(self, A_rc, adjacency_list, time)
                print(self.location_address[: -9] + "聚类完毕")
                break
            A_cnt = MWMS_S.A(self, self.G_original, 0.000001 * self.rc)  # 去重精度
            A_rc = MWMS_S.A(self, self.G_original, self.rc)
            MWMS_S.G_merge(self, A_cnt)
            A_trad = MWMS_S.A(self, self.G[1], self.rc)
            # if time % 150 == 0 and time != 0:
                # MWMS_S.TPfig(self, A_trad, start=-6, end=6)
            MWMS_S.TPfig(self, A_trad, start=-6, end=6, time=time)
            MWMS_S.lct_list(self, location_list, time)
            MWMS_S.ajc_list(self, A_rc, adjacency_list, time)

            M = MWMS_S.motif(self, A_trad)  # MWMS-S
            W_temp = MWMS_S.W_J(self, A_trad, M)
            W_r_temp = MWMS_S.W_r(self, W_temp)
            L_r_temp = MWMS_S.L_r(self, A_trad, W_r_temp)
            MWMS_S.MWMS_S(self, L_r_temp)

            for index in range(len(self.G[0])):
                for element in self.G[0][index]:
                    self.G_original[element] = self.G[1][index]
            print(time)
            time += 1
        # MWMS_S.file_write(self, location_list, adjacency_list)
        return "SUCCESS"


m_s = MWMS_S(False)
m_s.main()
exit(1)