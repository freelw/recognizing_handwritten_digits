import torch
import torch.nn.functional as F

x = torch.tensor([[[1,2,1,1]]], dtype=torch.float32)
x.requires_grad = True

g = torch.tensor([[[1,1,1,1]]], dtype=torch.float32)

# 在最后两个维度上进行softmax
result = F.softmax(x, dim=-1)
print(result)

result.backward(g)
print("x grad : ", x.grad)
