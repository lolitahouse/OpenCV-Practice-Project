# -*- coding = utf-8 -*-
# @Time :  10:09
# @Author : lolita
# @File : contour_demo.py
# @Software: PyCharm

import cv2
import numpy as np

'''
#  绘制轮廓
img = cv2.imread("./material/contour.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, contours, -1, (0, 0, 255), 5)
cv2.imshow("result", img)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
#  逐一绘制轮廓
img = cv2.imread("./material/contour.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n = len(contours)
contoursImg = []
for i in range(n):
    temp = np.zeros(img.shape, np.uint8)
    contoursImg.append(temp)
    contoursImg[i] = cv2.drawContours(contoursImg[i], contours, i, (0, 0, 255), 5)
    cv2.imshow("contours[" + str(i) + "]", contoursImg[i])
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 计算矩：moment(contours[i])
# 计算面积：contourArea(contours[i])
# 计算长度：arcLength(contours[i],True)
# Hu矩函数进行形状匹配
'''

# 矩形包围框
img = cv2.imread("./material/boundary.png")
# -----------提取轮廓----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# ------------构造矩形边界-----------------------
x, y, w, h = cv2.boundingRect(contours[0])
brcnt = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
cv2.drawContours(img, [brcnt], -1, (255, 255, 255), 2)
# ------------显示矩形边界--------------------------
cv2.imshow("result", img)
cv2.waitKey()
cv2.destroyAllWindows()


