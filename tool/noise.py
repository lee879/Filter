import numpy as np
def noise(img, density=0.2):
    assert img.ndim == 3,\
        "这是个给彩色图片添加噪声的"
    noise = np.zeros(shape=(img.shape[0], img.shape[1]), dtype="int32")
    noise = np.random.uniform(0, 1, noise.shape)
    threshold = 1 - density / 2
    noise[noise > threshold] = 255
    noise[noise < density] = -255
    k = noise
    k1 = k == 255
    k1 = np.int32(k1)
    K_1 = np.stack((k1,k1,k1),axis=-1) * 255
    img[(img + K_1) >= 255] = 255
    k2 = (k == -255)
    k2 = np.int32(k2)
    K_2 = np.stack((k2,k2,k2),axis=-1) * -255
    img[(img + K_2) <= 0] = 0
    return img
