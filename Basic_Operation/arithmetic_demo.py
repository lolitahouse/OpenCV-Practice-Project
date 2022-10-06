# -*- coding = utf-8 -*-
# @Time :  20:45
# @Author : lolita
# @File : arithmetic_demo.py
# @Software: PyCharm

import cv2
import numpy as np

'''
# 加法 '+' 和 'add()' 和 'addWeighted()'
img1 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
img2 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
print(img1)
print(img2)
# print(img1 + img2)           # '+'大于255，取余
# print(cv2.add(img1, img2))     # 'add()'大于255，值为255
print(cv2.addWeighted(img1, 0.6, img2, 0.2, 5))  # 加权和，gamma为调节量
'''

# 位平面提取并处理显示
cat = cv2.imread("./material/cat.jfif", 0)
cv2.imshow("cat", cat)
r, c = cat.shape
x = np.zeros((r, c, 8), dtype=np.uint8)
for i in range(8):
    x[:, :, i] = 2 ** i
r = np.zeros((r, c, 8), dtype=np.uint8)
for i in range(8):
    r[:, :, i] = cv2.bitwise_and(cat, x[:, :, i])
    mask = r[:, :, i] > 0
    r[mask] = 255
    cv2.imshow(str(i), r[:, :, i])
cv2.waitKey()
cv2.destroyAllWindows()



