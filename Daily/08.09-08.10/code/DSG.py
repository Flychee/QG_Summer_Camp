import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import copy


def component(A):  # 生成连通分支
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
                    """if len((result[-1] | set(temp))) != len([*result[-1], *temp]):"""
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