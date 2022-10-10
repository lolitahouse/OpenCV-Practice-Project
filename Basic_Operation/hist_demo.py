# -*- coding = utf-8 -*-
# @Time :  19:19
# @Author : lolita
# @File : hist_demo.py
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt

'''
# 绘制直方图
img = cv2.imread("./material/cat.jfif", 0)
cv2.imshow("original", img)
plt.hist(img.ravel(), 256)
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
'''

# 均衡化
img = cv2.imread("./material/cat.jfif", 0)
equ = cv2.equalizeHist(img)
cv2.imshow("original", img)
cv2.imshow("result", equ)
plt.figure("原始")
plt.hist(img.ravel(), 256)
plt.figure("均衡化")
plt.hist(equ.ravel(), 256)
cv2.waitKey()
cv2.destroyAllWindows()

# subplot()



