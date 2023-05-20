import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# 生成带有高斯噪声的正弦函数
def generate_noisy_sine_wave(num_samples, amplitude, frequency, noise_std):
    time = (np.arange(num_samples) / num_samples) * 100
    clean_signal = amplitude * np.sin(2 * np.pi * frequency * time)
    noise = np.random.normal(0, noise_std, num_samples)
    noisy_signal = (clean_signal + noise)
    return noisy_signal

# 离散维纳滤波器模型
class WienerFilter(nn.Module):
    def __init__(self, filter_order):
        super(WienerFilter, self).__init__()
        self.filter_order = filter_order
        self.weights = nn.Parameter(torch.randn(filter_order + 1))

    def forward(self, input_signal):
        padded_input = torch.cat([torch.zeros(self.filter_order), input_signal])
        filtered_signal = torch.conv1d(padded_input[None, None, :], self.weights[None, None, :])
        return filtered_signal.squeeze()

# 设置参数
amplitude = 1.0
frequency = 0.1
phase = 0.0
noise_std = 1
num_samples = 1000
filter_order = 50

# 生成带有高斯噪声的正弦函数
noisy_signal = torch.from_numpy(generate_noisy_sine_wave(num_samples, amplitude, frequency, noise_std)).type(torch.float32)
expect_signal = torch.from_numpy(generate_noisy_sine_wave(num_samples, amplitude, frequency, noise_std = 0)).type(torch.float32)

# 创建离散维纳滤波器模型
wiener_filter = WienerFilter(filter_order)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(wiener_filter.parameters(), lr=0.001)

# 训练模型
num_epochs = 100000
for epoch in range(num_epochs):
    filtered_signal = wiener_filter(noisy_signal)
    loss = criterion(filtered_signal, expect_signal)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 绘制结果
plt.figure(figsize=(10, 4))
plt.plot(noisy_signal, label='Noisy Signal')
plt.plot(np.arange(len(expect_signal)),expect_signal, label='expect_signal')
plt.plot(filtered_signal.detach().numpy(), label='Filtered Signal')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Discrete Wiener Filter')
plt.show()