# -*- coding = utf-8 -*-
# @Time :  0:05
# @Author : lolita
# @File : conversion_demo.py
# @Software: PyCharm

import cv2
import numpy as np

'''
# 缩放
img = np.ones([2, 4, 3], dtype=np.uint8)
size = img.shape[:2]
rst = cv2.resize(img, size)
print(img)
print(rst)
'''
'''
# 翻转
img = cv2.imread("./material/cat.jfif")
x = cv2.flip(img, 0)
y = cv2.flip(img, 1)
xy = cv2.flip(img, -1)
cv2.imshow("x", x)
cv2.imshow("y", y)
cv2.imshow("xy", xy)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 仿射平移
img = cv2.imread("./material/cat.jfif")
height, width = img.shape[:2]
x = 100
y = 200
M = np.float32([[1, 0, x], [0, 1, y]])
move = cv2.warpAffine(img, M, (width, height))
cv2.imshow("original", img)
cv2.imshow("move", move)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 旋转
img = cv2.imread("./material/cat.jfif")
height, width = img.shape[:2]
M = cv2.getRotationMatrix2D((width/2, height/2), 45, 0.6)
rotate = cv2.warpAffine(img, M, (width, height))
cv2.imshow("original", img)
cv2.imshow("rotation", rotate)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 更复杂的仿射变换
img = cv2.imread("./material/cat.jfif")
rows, cols, ch = img.shape
p1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
p2 = np.float32([[0, rows*0.33], [cols*0.85, rows*0.25], [cols*0.15, rows*0.7]])
M = cv2.getAffineTransform(p1, p2)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow("original", img)
cv2.imshow("result", dst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 透视
img = cv2.imread("./material/cat.jfif")
rows, cols = img.shape[:2]
p1 = np.float32([[150, 50], [400, 50], [60, 450], [310, 450]])
p2 = np.float32([[50, 50], [rows-50, 50], [50, cols-50], [rows-50, cols-50]])
M = cv2.getPerspectiveTransform(p1, p2)
dst = cv2.warpPerspective(img, M, (cols, rows))
cv2.imshow("original", img)
cv2.imshow("result", dst)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# 图像复制
img = cv2.imread("./material/cat.jfif")
rows, cols = img.shape[:2]
mapx = np.zeros(img.shape[:2], np.float32)
mapy = np.zeros(img.shape[:2], np.float32)
for i in range(rows):
    for j in range(cols):
        mapx.itemset((i, j), j)
        mapy.itemset((i, j), i)
rst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
cv2.imshow("original", img)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()
'''

# 翻转
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:, 0:100, 0] = 255
img[:, 100:200, 1] = 255
img[:, 200:300, 2] = 255
img[100:150, 100:150, :] = 255
rows, cols = img.shape[:2]
mapx1 = np.zeros(img.shape[:2], np.float32)
mapy1 = np.zeros(img.shape[:2], np.float32)
mapx2 = np.zeros(img.shape[:2], np.float32)
mapy2 = np.zeros(img.shape[:2], np.float32)
mapx3 = np.zeros(img.shape[:2], np.float32)
mapy3 = np.zeros(img.shape[:2], np.float32)
for i in range(rows):
    for j in range(cols):
        mapx1.itemset((i, j), j)
        mapy1.itemset((i, j), rows-1-i)
        mapx2.itemset((i, j), cols-1-j)
        mapy2.itemset((i, j), i)
        mapx3.itemset((i, j), cols-1-j)
        mapy3.itemset((i, j), rows-1-i)
rstx = cv2.remap(img, mapx1, mapy1, cv2.INTER_LINEAR)
rsty = cv2.remap(img, mapx2, mapy2, cv2.INTER_LINEAR)
rstxy = cv2.remap(img, mapx3, mapy3, cv2.INTER_LINEAR)
cv2.imshow("img", img)
cv2.imshow("resultx", rstx)
cv2.imshow("resulty", rsty)
cv2.imshow("resultxy", rstxy)
cv2.waitKey()
cv2.destroyAllWindows()






