# -*- coding = utf-8 -*-
# @Time :  19:33
# @Author : lolita
# @File : threshold_demo.py
# @Software: PyCharm

import cv2

'''
# 二值化
img = cv2.imread("./material/cat.jfif", 0)
t, rst = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 反二值化
img = cv2.imread("./material/cat.jfif", 0)
t, rst = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 截断
img = cv2.imread("./material/cat.jfif", 0)
t, rst = cv2.threshold(img, 180, 255, cv2.THRESH_TRUNC)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 超阈值零处理
img = cv2.imread("./material/cat.jfif", 0)
t, rst = cv2.threshold(img, 180, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 低阈值零处理
img = cv2.imread("./material/cat.jfif", 0)
t, rst = cv2.threshold(img, 180, 255, cv2.THRESH_TOZERO)
cv2.imshow("img", img)
cv2.imshow("rst", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 自适应
img = cv2.imread("./material/cat.jfif", 0)
MEAN = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
GAUS = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
cv2.imshow("mean", MEAN)
cv2.imshow("gaus", GAUS)
cv2.waitKey()
cv2.destroyAllWindows()
'''

# otsu阈值处理
img = cv2.imread("./material/cat.jfif", 0)
t, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
cv2.imshow("img", img)
cv2.imshow("rst", otsu)
print(t)
cv2.waitKey()
cv2.destroyAllWindows()








