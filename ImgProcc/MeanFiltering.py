import cv2
import numpy as np
from tool import noise
#原图
img = cv2.imread("../photo/5.jpg",1)
cv2.imshow("../photo/5.jpg", img)
#加入噪声
img_noise = noise.noise(img,0.001)
cv2.imshow("../photo/5_1.jpg", img_noise)
# #均值滤波
# h = img.shape[0]
# w = img.shape[1]
# dst = np.zeros_like(img,dtype="uint8")
# #使用的6x6 = 36的模板来计算
# for i in range(3,h-3):
#     for j in range(3,w-3):
#         sum_b = int(0)
#         sum_g = int(0)
#         sum_r = int(0)
#         for m in range(-3,3):
#             for n in range(-3,3):
#                 (b,g,r) = img[i + m,j + n]
#                 sum_b = sum_b + int(b)
#                 sum_g = sum_g + int(g)
#                 sum_r = sum_r + int(r)
#         b = np.uint8(sum_b / 36)
#         g = np.uint8(sum_g / 36)
#         r = np.uint8(sum_r / 36)
#         dst[i,j] = (b,g,r)
#
# cv2.imshow("../photo/5_2.jpg", dst)

#如何使用函数来进行均值滤波
dst_2 = cv2.blur(img,(3,3))
cv2.imshow("../photo/5_3.jpg", dst_2)
cv2.waitKey(0)