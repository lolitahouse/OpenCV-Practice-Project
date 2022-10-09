# -*- coding = utf-8 -*-
# @Time :  9:21
# @Author : lolita
# @File : pyramid_demo.py
# @Software: PyCharm

import cv2
import numpy as np

'''
img = cv2.imread("./material/cat.jfif", 0)
# 向下采样
rst1 = cv2.pyrDown(img)
cv2.imshow("result1", rst1)
# 向下采样
rst2 = cv2.pyrUp(img)
cv2.imshow("result2", rst2)
cv2.waitKey()
cv2.destroyAllWindows()
'''

# 向上，向下不可逆
img = cv2.imread("./material/cat.jfif", 0)
down = cv2.pyrDown(img)
up = cv2.pyrUp(down)
diff = up-img
cv2.imshow("result", diff)
cv2.waitKey()
cv2.destroyAllWindows()
# 可以用拉普拉斯金字塔储存丢弃的部分，l1 = g0-pyrup(g1)































