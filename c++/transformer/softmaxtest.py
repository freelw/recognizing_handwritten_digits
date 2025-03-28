import torch
import torch.nn as nn




# 构造输入特征 x
x = [
    [0, 1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5, 6],
]

# 将输入特征 x 转换为张量

x = torch.tensor(x, dtype=torch.float32)

x.requires_grad = True


softmax_size = 6

# 初始化一个 Softmax 层

softmax = nn.Softmax(dim=1)

# 将输入特征 x 传入 Softmax 层

y = softmax(x)

# 打印 Softmax 层的输出

print("Softmax 层的输出：", y)