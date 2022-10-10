# -*- coding = utf-8 -*-
# @Time :  22:31
# @Author : lolita
# @File : KNN_demo.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------k近邻算法基本使用--------------------------------
# 用于训练的数据
rand1 = np.random.randint(0, 30, (20, 2)).astype(np.float32)
rand2 = np.random.randint(70, 100, (20, 2)).astype(np.float32)
# 将rand1和rand2拼接为训练数据
trainData = np.vstack((rand1, rand2))
# 数据标签，共两类，0和1
r1Label = np.zeros((20, 1)).astype(np.float32)
r2Label = np.ones((20, 1)).astype(np.float32)
tdLabel = np.vstack((r1Label, r2Label))
# 使用绿色标注类型0
g = trainData[tdLabel.ravel() == 0]
plt.scatter(g[:, 0], g[:, 1], 80, 'g', 'o')
# 使用蓝色标注类型1
b = trainData[tdLabel.ravel() == 1]
plt.scatter(b[:, 0], b[:, 1], 80, 'b', 'o')
# plt.show()
# 用于测试的随机数，值为0~100
test = np.random.randint(0, 100, (1, 2)).astype(np.float32)
plt.scatter(test[:, 0], test[:, 1], 80, 'r', '*')
# 调用k近邻模块进行训练
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, tdLabel)
# 使用k近邻算法分类
ret, result, neighbours, dist = knn.findNearest(test, 5)
# 显示处理结果
print(result)
print(neighbours)
print(dist)
plt.show()

