# -*- coding = utf-8 -*-
# @Time :  23:07
# @Author : lolita
# @File : SVM_demo.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------SVM基本使用--------------------------------
# 准备用于训练的数据
a = np.random.randint(95, 100, (20, 2)).astype(np.float32)
b = np.random.randint(90, 95, (20, 2)).astype(np.float32)
# 将rand1和rand2拼接为训练数据
data = np.vstack((a, b))
# 数据标签，共两类，0和1
aLabel = np.zeros((20, 1))
bLabel = np.ones((20, 1))
label = np.vstack((aLabel, bLabel))
label = np.array(label, dtype='int32')
# 创建SVM
svm = cv2.ml.SVM_create()
# 属性设置
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setC(0.01)
# 训练
result = svm.train(data, cv2.ml.ROW_SAMPLE, label)
# 预测
test = np.vstack([[98, 90], [90, 99]])
test = np.array(test, dtype='float32')
(p1, p2) = svm.predict(test)
# 观察结果
plt.scatter(a[:, 0], a[:, 1], 80, 'g', 'o')
plt.scatter(b[:, 0], b[:, 1], 80, 'b', 's')
plt.scatter(test[:, 0], test[:, 1], 80, 'r', '*')
plt.show()
print(test)
print(p2)

