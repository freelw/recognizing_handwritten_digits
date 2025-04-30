import torch

# 创建一个示例张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 对所有元素求和
sum_all = torch.sum(x)
print("所有元素求和结果:", sum_all)

# 在第 0 维上求和
sum_dim0 = torch.sum(x, dim=0)
print("在第 0 维上求和结果:", sum_dim0)

# 在第 1 维上求和，并保留维度
sum_dim1_keepdim = torch.sum(x, dim=1, keepdim=True)
print("在第 1 维上求和并保留维度结果:", sum_dim1_keepdim)