import numpy as np
# 定义网络拓扑
G = {
    '1': ['2', '4'],
    '2': ['1', '3', '4'],
    '3': ['2', '4', '5'],
    '4': ['1', '2', '3', '5'],
    '5': ['3', '4']
}
# 定义传感器测量值
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# 定义迭代次数
T = 10
# 初始化中继节点估计值为0
y = np.zeros(len(G))
bias = 1.01
# 迭代更新过程
for t in range(T):
    for i in range(len(G)):
        # 计算传输给相邻中继节点的信息
        msg = x[i] - y[i]
        # 随机选择一个相邻中继节点作为代表
        j = int(np.random.choice(G[str(i+1)]))
        # 将信息发送给代表节点
        y[j-1] += msg / (len(G[str(j)]) + bias)
    # 所有节点达到一致的估计值
    x_est = y + x.mean()
print("估计结果：", x_est)