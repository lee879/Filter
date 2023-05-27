import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import numpy as np
class LowPassFilterLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LowPassFilterLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        _, height, width, _ = input_shape
        # 计算图像中心点坐标
        center_row, center_col = height // 2, width // 2
        # 创建截止频率参数
        self.cutoff_frequency = self.add_weight(name='cutoff_frequency', shape=(),
                                    initializer=tf.keras.initializers.Constant(value=8.0),
                                    trainable=True)
        # 创建掩膜
        mask = np.zeros((height, width), np.float32)
        cutoff_frequency_value = tf.cast(self.cutoff_frequency, tf.int32).numpy()
        mask[center_row - cutoff_frequency_value: center_row + cutoff_frequency_value,
             center_col - cutoff_frequency_value: center_col + cutoff_frequency_value] = 1
        # 扩展掩膜到匹配图像的通道数
        mask = np.expand_dims(mask, axis=-1)
        mask = np.tile(mask, (1, 1, input_shape[-1]))
        # 创建掩膜的Tensor并转换为复数类型
        self.mask = tf.constant(mask, dtype=tf.complex64)
    def call(self, inputs):
        # 进行傅里叶变换
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        fft_shifted = tf.signal.fftshift(fft)
        # 将掩膜转换为复数类型
        mask_complex = tf.cast(self.mask, tf.complex64)
        # 应用掩膜
        filtered_shifted = fft_shifted * mask_complex
        # 进行逆向傅里叶变换
        filtered = tf.signal.ifftshift(filtered_shifted)
        filtered_image_complex = tf.signal.ifft2d(filtered)
        # 提取实部并取绝对值得到滤波后的图像张量
        filtered_image = tf.math.abs(tf.cast(filtered_image_complex, tf.float32))
        return filtered_image