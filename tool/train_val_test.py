"""
-------------------------------------------------
Description：划分数据集
-------------------------------------------------
"""
import os
import random
# train val 80%  test 20%
trainval_percent = 0.8
# train 75% val 5% test 20%
train_percent = 0.75
xmlfilepath = './datasets/Annotations'
txtsavepath = '/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('./datasets/ImageSets/Main/trainval.txt', 'w')
ftest = open('./datasets/ImageSets/Main/test.txt', 'w')
ftrain = open('./datasets/ImageSets/Main/train.txt', 'w')
fval = open('./datasets/ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()