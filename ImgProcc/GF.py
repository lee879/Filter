import cv2
from tool import noise
# 添加椒盐噪声

img = cv2.imread("./photo/5.jpg",1)
cv2.imshow("../photo/5.jpg", img)

#加入噪声
img_noise = noise.noise(img,0.001)
cv2.imshow("./photo/5_1.jpg", img_noise)

#高斯滤波
dst = cv2.GaussianBlur(img_noise,(5,5),3)
cv2.imshow("./photo/5_2.jpg", dst)
cv2.waitKey(0)