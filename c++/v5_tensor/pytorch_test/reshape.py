import torch

# 创建一个示例张量 3 * 2

x = torch.tensor([[1, 2], [3, 4], [5, 6]])

y = x.transpose(0, 1)

print ("x:", x)
print ("y:", y)

print ("x contigous:", x.is_contiguous())
print ("y contigous:", y.is_contiguous())

z = y.reshape(3, 2)

print ("z:", z)
print ("z contigous:", z.is_contiguous())