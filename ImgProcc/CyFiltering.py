import cv2
from PIL import Image
import numpy as np


img_1 = cv2.imread(r"../photo/1.jpg",0)
img_2 = cv2.GaussianBlur(img_1,(3,3),0) #高斯去噪
img_3 = cv2.Canny(img_1,50,50) # 图片经过卷积后的点大于后面的门限就认为是边缘检测，否则就不是是边缘检
img_4 = cv2.Canny(img_2,50,50)

# cv2.imshow("1",img_1)
# cv2.imshow("2",img_2)
cv2.imshow("3",img_3)
cv2.imshow("4",img_4)
cv2.waitKey(0)