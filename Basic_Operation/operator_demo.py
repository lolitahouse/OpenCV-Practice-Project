# -*- coding = utf-8 -*-
# @Time :  8:35
# @Author : lolita
# @File : operator_demo.py
# @Software: PyCharm

import cv2

'''
# Sobel算子
img = cv2.imread("./material/cat.jfif", 0)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv2.imshow("sobelxy", sobelxy)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# Sobel算子
img = cv2.imread("./material/cat.jfif", 0)
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
cv2.imshow("sobelxy", scharrxy)
cv2.waitKey()
cv2.destroyAllWindows()
'''
'''
# laplacian算子
img = cv2.imread("./material/cat.jfif", 0)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
cv2.imshow("laplacian", laplacian)
cv2.waitKey()
cv2.destroyAllWindows()
'''

# Canny边缘检测
img = cv2.imread("./material/cat.jfif", 0)
rst = cv2.Canny(img, 10, 55)
cv2.imshow("result", rst)
cv2.waitKey()
cv2.destroyAllWindows()
