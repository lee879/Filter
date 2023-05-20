import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras import layers

class Basic_Block(layers.Layer):
    def __init__(self,filter_num,s=1):
        super(Basic_Block, self).__init__()
        self.conv1 = layers.Conv2D(filter_num,kernel_size=3,strides=s,padding="same")
        self.bn1 = layers.BatchNormalization(axis=-1)
        self.relu1 = layers.Activation("relu")

        self.conv2 = layers.Conv2D(filter_num,kernel_size=3,strides=1,padding="same")
        self.bn2 = layers.BatchNormalization(axis=-1)
        self.relu2 = layers.Activation("relu")
        if s !=1 :
            self.downsample = keras.Sequential()
            self.downsample.add(layers.Conv2D(filter_num,kernel_size=1,strides=s,padding="same"))
        else:
            self.downsample = lambda x:x
        self.stride = s

    def call(self, inputs, **kwargs):
        residual = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        add = layers.add([bn2,residual])
        out = self.relu2(add)
        return out







