# -*- coding = utf-8 -*-
# @Time :  0:31
# @Author : lolita
# @File : morphology.py
# @Software: PyCharm

import cv2

'''
# 腐蚀
img = cv2.imread("./material/erode.png")
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=5)
# 膨胀
dilation = cv2.dilate(erosion, kernel)
cv2.imshow("original", img)
cv2.imshow("erosion", erosion)
cv2.imshow("delation", dilation)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
#  通用形态学函数
img = cv2.imread("./material/erode.png")
kernel = np.ones((10, 10), np.uint8)
kernel1 = np.ones((5, 5), np.uint8)
# 开运算
rst1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# 闭运算
rst2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# 梯度运算
rst3 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel1)
# 礼帽运算
rst4 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel1)
# 黑帽运算
rst5 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel1)
cv2.imshow("original", img)
cv2.imshow("open", rst1)
cv2.imshow("close", rst2)
cv2.imshow("gradient", rst3)
cv2.imshow("tophat", rst4)
cv2.imshow("blackhat", rst5)
cv2.waitKey()
cv2.destroyAllWindows()
'''

# 核函数
img = cv2.imread("./material/erode.png")
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (59, 59))
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (59, 59))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (59, 59))
rst1 = cv2.dilate(img, kernel1)
rst2 = cv2.dilate(img, kernel2)
rst3 = cv2.dilate(img, kernel3)
cv2.imshow("original", img)
cv2.imshow("rst1", rst1)
cv2.imshow("rst2", rst2)
cv2.imshow("rst3", rst3)
cv2.waitKey()
cv2.destroyAllWindows()






