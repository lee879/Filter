import torch
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用第一个可用的 GPU
    print("use gpu")
else:
    device = torch.device("cpu")

def generate_noisy_sine_wave(num_samples, amplitude, frequency, noise_std):
    time = np.arange(num_samples) / float(num_samples)
    clean_signal = amplitude * np.sin(2 * np.pi * frequency * time)
    noise = np.random.normal(0, noise_std, num_samples)
    noisy_signal = (clean_signal + noise)
    return noisy_signal



def atc_diffusion_lms(input_signal, desired_signal, tap_length, num_users, num_iterations, step_size):
    # 初始化每个用户的滤波器权重
    filter_weights = torch.zeros((num_users, tap_length), dtype=torch.float32)


    for _ in range(num_iterations):
        # 每个用户的信道估计
        channel_estimates = torch.mul(input_signal , filter_weights)

        # 每个用户的误差计算
        errors = (torch.unsqueeze(desired_signal,0).tile((4,1)) - channel_estimates)

        # 更新每个用户的滤波器权重
        filter_weights += torch.mul(step_size , ( torch.mul(torch.unsqueeze(input_signal,0).tile((4,1)) , errors)))

       # print("error:",torch.mean(errors).numpy())

    return filter_weights, errors , channel_estimates

# 示例数据生成
num_samples = 1000  # 信号采样点数
amplitude = 1.0  # 正弦信号幅度
frequency = 10.0  # 正弦信号频率
noise_std = 0.5  # 噪声标准差

input_signal = torch.from_numpy(generate_noisy_sine_wave(num_samples, amplitude, frequency, noise_std)).type(torch.float32)
out_signal = torch.from_numpy(generate_noisy_sine_wave(num_samples, amplitude, frequency, noise_std =.0)).type(torch.float32)

# 其他代码和ATC Diffusion LMS算法保持一致
tap_length = 1000
num_users = 4
num_iterations = 800000 # 迭代的估计
step_size = 0.1

filter_weights, error, channel_estimates = atc_diffusion_lms(input_signal, out_signal, tap_length, num_users, num_iterations, step_size)

print(channel_estimates.shape)

# plt.plot(np.arange(len(input_signal)), input_signal.numpy(), label='inp')
plt.plot(np.arange(len(input_signal)), out_signal.numpy(), label='expect')
plt.plot(np.arange(len(input_signal)), channel_estimates[0].numpy(), label='user_0')
# plt.plot(np.arange(len(input_signal)), channel_estimates[1].numpy(), label='user_1')
# plt.plot(np.arange(len(input_signal)), channel_estimates[2].numpy(), label='user_2')
# plt.plot(np.arange(len(input_signal)), channel_estimates[3].numpy(), label='user_2')

plt.legend()
plt.show()

