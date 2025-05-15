import torch
import torch.nn.functional as F

x = torch.tensor([[[0, 0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6, 0.7],
  [0.8, 0.9, 1, 1.1]]])

# 在最后两个维度上进行softmax
result = F.softmax(x, dim=-1)
print(result)
