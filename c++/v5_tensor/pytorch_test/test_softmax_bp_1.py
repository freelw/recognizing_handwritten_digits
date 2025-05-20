import torch
import torch.nn as nn

input = torch.tensor(
    [[[1, 1, 1],
    [1, 1, 1]]],
    dtype=torch.float32
)
input.requires_grad_(True) 

res = input.reshape(2,3) + nn.functional.softmax(input.reshape(2,3), dim=-1)
res.retain_grad()

res_grad = torch.tensor(
    [[0, 1, 2],
    [3, 4, 5]],
    dtype=torch.float32
)

res.backward(gradient=res_grad)

print("res: ", res)
print("res grad: ", res.grad)
print("input: ", input)
print("input grad: ", input.grad)
