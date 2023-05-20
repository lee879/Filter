# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Description：jpg格式转化为png格式的图片
-------------------------------------------------
"""
import os
import cv2
def transform(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for name in files:
            file = os.path.join(root, name)
            print('transform:' + name)
            im = cv2.imread(file)
            if output_path:
                cv2.imwrite(os.path.join(output_path, name.replace('jpg', 'png')), im)
            else:
                cv2.imwrite(file.replace('jpg', 'png'), im)

if __name__ == '__main__':
    input_path = r".\p"

    output_path = r".\D"
    if not os.path.exists(input_path):
        print("文件夹不存在!")
    else:
        print("Start to transform!")
        transform(input_path, output_path)
        print("Transform end!")
