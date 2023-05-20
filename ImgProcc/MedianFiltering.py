import cv2
import numpy as np
from tool import noise
#原图
img = cv2.imread("../photo/5.jpg",1)
cv2.imshow("../photo/5.png", img)
h = img.shape[0]
w = img.shape[1]

img_noise = noise.noise(img,0.3)
cv2.imshow("../photo/5_1.png", img_noise)

# #实现中值滤波 3x3 的模板
# dst = np.zeros_like(img_noise,dtype="uint8")
# mask = []
# for i in range(1,h-1):
#     for j in range(1,w-1):
#         k = 0
#         for m in range(-1,2):
#             for n in range(-1,2):
#                 mask.append(img_noise[i+m,j+n])
#                 k = k + 1
#         b = np.median(np.array(mask)[:,0])
#         g = np.median(np.array(mask)[:,1])
#         r = np.median(np.array(mask)[:,2])
#
#         dst[i, j] = (b, g, r)
#         print(k)
dst = cv2.medianBlur(img_noise,9)
cv2.imshow("../photo/5_2.png", dst)
cv2.waitKey(0)