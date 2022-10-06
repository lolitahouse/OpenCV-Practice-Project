# -*- coding = utf-8 -*-
# @Time :  19:22
# @Author : lolita
# @File : numpy_demo.py
# @Software: PyCharm

import cv2
import numpy as np

'''
# 用zeros生成二维数组
img = np.zeros((8, 8), dtype=np.uint8)
print("img = \n", img)
cv2.imshow("one", img)
print("读取像素点img[0, 3]=", img[0, 3])
img[0, 3] = 255
print("修改后 img = \n", img)
print("修改后 img[0,3]=", img[0, 3])
cv2.imshow("two", img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
# 读取并修改图像像素
img = cv2.imread("./material/cat.jfif", 0)
cv2.imshow("before", img)
for i in range(10, 100):
    for j in range(80, 100):
        img[i, j] = 0

cv2.imshow("after", img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
# zeros生成三维数组，观察通道
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:, 0:100, 0] = 255
img[:, 100:200, 1] = 255
img[:, 200:300, 2] = 255
print("img=\n", img)
cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
'''

'''
# random生成数组
img = np.random.randint(10, 99, size=[5, 5], dtype=np.uint8)
print("img=\n", img)
print(img.item(3, 2))
img.itemset((3, 2), 255)
print(img.item(3, 2))
'''

'''
# split拆分通道
img = np.random.randint(0, 256, size=(400, 400, 3), dtype=np.uint8)
b, g, r = cv2.split(img)
cv2.imshow("b", b)
# merge 合并通道
rgb = cv2.merge([r, g, b])
cv2.imshow("rgb", rgb)
cv2.waitKey()
cv2.destroyAllWindows()
'''

# 显示图像属性
gray = cv2.imread("./material/cat.jfif")
print(gray.shape)       # 返回行，列，通道
print(gray.size)        # 返回像素数
print(gray.dtype)       # 返回类型




