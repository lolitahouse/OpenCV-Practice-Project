# -*- coding = utf-8 -*-
# @Time :  23:39
# @Author : lolita
# @File : basic_test.py
# @Software: PyCharm

import cv2

# 读取图像
cat = cv2.imread("./material/cat.jfif")
print(cat)
# 显示图像
cv2.imshow("cat", cat)
# 等待按下按键
cv2.waitKey()
# 销毁窗口
cv2.destroyAllWindows()
# 保存图像
cv2.imwrite("cat.png", cat)











