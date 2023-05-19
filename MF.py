import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 定义匹配滤波器
def matched_filter(signal, template):
    # 将信号和模板张量的形状调整为二维
    signal = signal.unsqueeze(0).unsqueeze(0)
    template = template.unsqueeze(0).unsqueeze(0)
    # 信号和模板进行互相关运算
    correlation = F.conv1d(signal, template)
    # 返回相关性结果
    return correlation.squeeze(0).squeeze(0)

# 生成正弦信号和模板
t = torch.linspace(0, 1, 1000)  # 时间轴
signal = torch.sin(2 * np.pi * 10 * t)
signal = signal + np.random.normal(loc=0,scale=1,size=(len(signal))).astype(np.float32) # 加上一个噪声
template = signal[:150]

# 使用匹配滤波器进行信号处理
correlation_result = matched_filter(signal, template)

# 裁剪相关性结果与时间轴的长度一致
correlation_result = correlation_result[:len(signal)-len(template)+1]

# 绘制信号和相关性结果
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(t, signal.numpy(), label='Signal')
plt.plot(t[:150], template.numpy(), label='Template')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()

t_corr = torch.linspace(0, 1, len(correlation_result))  # 更新时间轴
plt.subplot(2, 1, 2)
plt.plot(t_corr, correlation_result.numpy())
plt.xlabel('Time')
plt.ylabel('Correlation')
plt.tight_layout()
plt.show()