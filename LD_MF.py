import numpy as np
import torch
import torch.nn as nn


class MatchedFilter(nn.Module):
    def __init__(self, filter_size):
        super(MatchedFilter, self).__init__()
        self.filter_size = filter_size
        self.filter = nn.Parameter(torch.randn(filter_size))

    def forward(self, x):
        x = torch.conv1d(x.unsqueeze(0).unsqueeze(0), self.filter.unsqueeze(0).unsqueeze(0))
        return x.squeeze(0).squeeze(0)

# 定义滤波器尺寸和输入信号
filter_size = 2
input_signal = np.array([1, 0, 0, 1, 0, 0, 0]) + np.random.normal(size=(7)).astype(np.float32)

# 将输入信号转换为PyTorch的Tensor
input_tensor = torch.Tensor(input_signal)

# 创建匹配滤波器模型
matched_filter = MatchedFilter(filter_size)

# 对输入信号进行去噪
output_signal = matched_filter(input_tensor)

# 打印去噪后的输出信号
print(output_signal.detach().numpy())
