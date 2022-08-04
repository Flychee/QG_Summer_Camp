import numpy as np
import copy
import time
from scipy.stats import pearsonr
import sqlalchemy
import pandas as pd
from pandas.io import sql
from sqlalchemy import create_engine


class Algorithm:
    def __init__(self):
        self.epsilon = 1
        self.rho = 0.5
        self.beta = 1
        self.R = []
        self.I_before_num = 1000
        self.R_T = []

    def norm2(self, a):  # 2范数
        return np.linalg.norm(np.array(a))

    def PCC(self, a, b):  # 皮尔逊相关系数
        if Algorithm.norm2(self, a=a) == 0 or Algorithm.norm2(self, a=b) == 0:
            return 0
        else:
            return pearsonr(a, b)[0]

    def COS(slef, a, b):  # 基于余弦的相似度
        if Algorithm.norm2(self, a=a) == 0 or Algorithm.norm2(self, a=b) == 0:
            return 0
        else:
            return np.dot(a, b) / (Algorithm.norm2(self, a=a) * Algorithm.norm2(self, a=b))

    def create(self):
        engine = sqlalchemy.create_engine('mysql+pymysql://root:3751ueoxjwgixjw3913@39.98.41.126:3306/book_management')
        sql1 = ''' select kind_num from library '''
        sql2 = ''' select * from rating '''
        df1 = pd.read_sql(sql1, engine)
        df2 = pd.read_sql(sql2, engine)
        kind_num = df1.kind_num.values
        user = df2.user.values
        item = df2.item.values
        praise = df2.praise.values
        collection = df2.collection.values
        for user_ in range(max(user)):
            self.R.append(copy.deepcopy([]))
            for item_ in range(max(item)):
                self.R[-1].append(0)
        for user_, item_, praise_, collection_, in zip(user, item, praise, collection):
            num = 0
            if user_ != kind_num[item_]:
                if praise_ == 1:
                    num += np.random.normal(20, 5)
                    if collection_ == 1:
                        num += 3
                elif collection_ == 1 and praise_ == 0:
                    num += np.random.normal(30, 10)
            if user_ == kind_num[item_]:
                if praise_ == 1:
                    num += np.random.normal(75, 5)
                    if collection_ == 1:
                        num += 10
                elif collection_ == 1 and praise_ == 0:
                    num += np.random.normal(80, 5)
            self.R[user_ - 1][item_ - 1] = num
        self.R = np.array(self.R)
        self.R_T = self.R.T

    def clean(self):  # 清洗数据
        R_temp = []
        for r in self.R:
            R_temp.append(copy.deepcopy([]))
            for r_element in r:
                if np.isnan(r_element):
                    R_temp[-1].append(0)
                else:
                    R_temp[-1].append(r_element)
        self.R = np.array(R_temp)
        self.R_T = self.R.T

    def I_before(self, i):  # 初步选取相关性较高的项目
        S_i_temp = np.array([Algorithm.PCC(self, a=self.R_T[i], b=element) for element in self.R_T])
        S_i_temp_sort = np.sort(-S_i_temp)
        num_temp = self.I_before_num
        while 1:
            if S_i_temp_sort[num_temp] >= 0:
                num_temp -= 1
            else:
                return np.argsort(-S_i_temp)[1:num_temp]

    def I(self, I_before, a):  # 在相关性高的项目里选取用户评分过的项目
        return [index for index in I_before if self.R[a][index] != 0]

    def max_len(self, i, I):  # 向量对最大长度
        max = 0
        for element in self.R_T[i]:
            if element != 0:
                max += 1
        for num in I:
            result = 0
            for element in self.R_T[num]:
                if element != 0:
                    result += 1
            if result >= max:
                max = result
        return max

    def U(self, i, j):
        return [user for user in range(len(self.R_T[0])) if self.R_T[i][user] != 0 and self.R_T[j][user] != 0]

    def RS(self, U, i, j):  # 计算敏感度
        if len(U) == 1:
            return np.exp(-1 * self.beta)
        R_i = [self.R[u][i] for u in U]
        R_j = [self.R[u][j] for u in U]
        R_i_simi = copy.deepcopy(R_i)
        R_j_simi = copy.deepcopy(R_j)
        R_i_simi.pop(int(np.random.randint(0, len(R_i))))  # 邻近数据集
        R_j_simi.pop(int(np.random.randint(0, len(R_j))))  # 邻近数据集
        max_r = 0
        max_r_simi = 0
        for u in range(len(U)):
            r = R_i[u] * R_j[u] / (Algorithm.norm2(self, R_i_simi) * Algorithm.norm2(self, R_j_simi))
            if r >= max_r:
                max_r = r
            r_simi = R_i[u] * R_j[u] * np.abs(
                Algorithm.norm2(self, R_j_simi) * Algorithm.norm2(self, R_i) - Algorithm.norm2(self,
                                                                                               R_i_simi) * Algorithm.norm2(
                    self, R_j)) \
                     / (Algorithm.norm2(self, R_i_simi) * Algorithm.norm2(self, R_j_simi) * Algorithm.norm2(self,
                                                                                                            R_i) * Algorithm.norm2(
                self, R_j))
            if r_simi >= max_r_simi:
                max_r_simi = r_simi
        result = max(max_r, max_r_simi)
        return result * np.exp(-1 * self.beta)

    def s_k(self, i, k):  # 第k高相似度
        S_i_temp = [Algorithm.PCC(self, a=self.R_T[i], b=element) for element in self.R_T]
        S_i_temp.sort()
        return S_i_temp[-k]

    def w(self, s_k, k, RS, max_len):  # 截断参数
        return min(s_k, 4 * k * max(RS) * np.log(k * (max_len - k) / self.rho) / self.epsilon)

    def PNC(self, RS, I, k, s_k, w, i):  # Private Neighbor Selection
        C1 = []
        C0 = []
        result = [[], []]  # 选取邻居的索引和敏感度
        S_tru = [[], []]  # 截断矩阵
        Prob = [[], []]  # 概率矩阵
        Rd = [[], []]  # 概率累积
        S_i_temp = [Algorithm.PCC(self, a=self.R_T[i], b=element) for element in self.R_T]
        for element in I:
            S_tru[0].append(element)
            if S_i_temp[element] > s_k - w:
                S_tru[1].append(S_i_temp[element])
                if i != element:
                    C1.append(element)
            else:
                S_tru[1].append(s_k - w)
                if i != element:
                    C0.append(element)
        part_of_denominator = 0
        for index in range(len(C1)):
            part_of_denominator += np.exp(self.epsilon * S_tru[1][index] / (4 * k * RS[index]))
        for index in range(len(S_tru[1])):
            Prob[0].append(S_tru[0][index])
            Prob[1].append(np.exp(self.epsilon * S_tru[1][index]) / (4 * k * RS[index]) /
                           (part_of_denominator + len(C0) * np.exp(
                               self.epsilon * S_tru[1][index] / (4 * k * RS[index]))))
        for index in range(len(Prob[0])):  # 将概率转为概率分布
            Rd[0].append(Prob[0][index])
            Rd[1].append(copy.deepcopy([]))
            if index == 0:
                Rd[1][-1].append(0)
            else:
                Rd[1][-1].append(Rd[1][-2][1])
            Rd[1][-1].append(Rd[1][-1][0] + Prob[1][index])
        num = 0
        while num < k:
            rd = np.random.uniform(0, np.sum(Prob[1]))
            for index_j in range(len(Rd[1])):
                if Rd[1][index_j][0] <= rd < Rd[1][index_j][1]:
                    result[0].append(Rd[0][index_j])
                    result[1].append(RS[index_j])  # 对应RS敏感度
                    Rd[1][index_j][0] = -1
                    Rd[1][index_j][1] = -1
                    num += 1
                    break
        return result

    def ave(self, a):
        num = 0
        for element in self[a]:
            if element != 0:
                num += 1
        return sum(self[a]) / num

    def PNCF(self, N, a, i):  # Private Neighbor Collaborative Filtering
        S_N = []  # 差分隐私处理后相似性
        S_i_temp = [Algorithm.PCC(self, a=self.R_T[i], b=element) for element in self.R_T]
        for index in range(len(N[0])):
            s_i = S_i_temp[N[0][index]] + np.random.laplace(0, 2 * N[1][index] / self.epsilon)
            S_N.append(s_i)
        result = 0  # 预测的r_ai
        for index in range(len(N[0])):
            result += S_N[index] * (self.R_T[N[0][index]][a] - Algorithm.ave(self.R_T, N[0][index]))
        result /= np.sum(S_N)
        result += Algorithm.ave(self.R_T, i)
        return result

    def choose_book(self, R_pre_a, a, return_book):  # 返回推荐书籍
        temp = []
        for index, element in zip(R_pre_a[0], R_pre_a[1]):
            if self.R[a][index] == 0:
                temp.append((element, index))
        temp.sort(key=lambda tup: tup[0])
        book1 = [n[1] for n in temp]
        book2 = []  # 消除重复元素
        for element in book1:
            if not element in book2:
                book2.append(element)
        return book2[:return_book]

#    def MAE(self, R_pre_a, a):  # 效能分析
#        result = 0
#        num = 0
#        for element, element_j in zip(R_pre_a, self.R[a]):
#           if element_j != 0:
#                num += 1
#                result += np.abs(element - element_j)
#        return result / num

    def main_func(self, a, num_func, return_book):
        R_pre_a = [[], []]
        time = 0
        for element in range(num_func):
            time += 1
            i = np.random.randint(0, len(self.R[a]))
            R_pre_a[0].append(i)
            S_i_temp = [Algorithm.PCC(self, a=self.R_T[i], b=element) for element in self.R_T]
            if Algorithm.norm2(self, a=self.R_T[i]) == 0 \
                    or Algorithm.norm2(self, a=S_i_temp) == 0:
                R_pre_a[1].append(0)
            else:
                I_before_i = Algorithm.I_before(self, i=i)
                I_i = Algorithm.I(self, I_before=I_before_i, a=a)
                if len(I_i) < 10:
                    R_pre_a[1].append(0)
                else:
                    max_len_i = len(I_i)  # 可选邻居最大数
                    k_i = int(max_len_i / 2)
                    RS_i = []  # 敏感度
                    for element in I_i:
                        U_i = Algorithm.U(self, i=i, j=element)
                        rs_i = Algorithm.RS(self, U=U_i, i=i, j=element)
                        RS_i.append(rs_i)
                    if len(RS_i) == 0:
                        R_pre_a[1].append(0)
                    else:
                        s_k_i = Algorithm.s_k(self, i=i, k=k_i)
                        w_i = Algorithm.w(self, s_k=s_k_i, k=k_i, RS=RS_i, max_len=max_len_i)
                        N_i = Algorithm.PNC(self, RS=RS_i, I=I_i, k=k_i, s_k=s_k_i, w=w_i, i=i)
                        R_pre_a[1].append(Algorithm.PNCF(self, N=N_i, a=a, i=i))
            print('time = {}'.format(time))
        return Algorithm.choose_book(self, R_pre_a=R_pre_a, a=a, return_book=return_book)


agr = Algorithm()
agr.create()
agr.clean()
for a in range(5):
    book = agr.main_func(a, 200, 50)
    with open('result{}.txt'.format(a), 'w') as file:
       for i in book:
            file.write(str(i)+',')

