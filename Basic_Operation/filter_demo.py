# -*- coding = utf-8 -*-
# @Time :  23:57
# @Author : lolita
# @File : filter_demo.py
# @Software: PyCharm

import cv2
import numpy as np

'''
# 均值滤波
img = cv2.imread("./material/noise.png")
rst = cv2.blur(img, (5, 5))
cv2.imshow("original", img)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 方框滤波
img = cv2.imread("./material/noise.png")
rst1 = cv2.boxFilter(img, -1, (2, 3))
rst2 = cv2.boxFilter(img, -1, (2, 2), normalize=0)
cv2.imshow("original", img)
cv2.imshow("result1", rst1)
cv2.imshow("result2", rst2)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 高斯滤波
img = cv2.imread("./material/noise.png")
rst = cv2.GaussianBlur(img, (5, 5), 0, 0)
cv2.imshow("original", img)
cv2.imshow("result1", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 中值滤波
img = cv2.imread("./material/noise.png")
rst = cv2.medianBlur(img, 3)
cv2.imshow("original", img)
cv2.imshow("result1", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 双边滤波
img = cv2.imread("./material/noise.png")
rst = cv2.bilateralFilter(img, 25, 100, 100)
cv2.imshow("original", img)
cv2.imshow("result1", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''

# 自定义卷积核
img = cv2.imread("./material/noise.png")
kernel = np.ones((9, 9), np.float32)/81
rst = cv2.filter2D(img, -1, kernel)
cv2.imshow("original", img)
cv2.imshow("result1", rst)
cv2.waitKey()
cv2.destroyAllWindows()





