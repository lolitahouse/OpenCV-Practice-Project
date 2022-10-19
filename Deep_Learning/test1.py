# -*- coding = utf-8 -*-
# @Time :  11:01
# @Author : lolita
# @File : test1.py
# @Software: PyCharm


# 初识感知机
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(AND(0, 0))  # 输出0
print(AND(1, 0))  # 输出0
print(AND(0, 1))  # 输出0
print(AND(1, 1))  # 输出1
