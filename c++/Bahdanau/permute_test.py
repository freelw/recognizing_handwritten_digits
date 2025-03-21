import torch
inputs = torch.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print("inputs:", inputs, inputs.shape)

p = inputs.permute(2, 0, 1)
print("p:", p, p.shape)