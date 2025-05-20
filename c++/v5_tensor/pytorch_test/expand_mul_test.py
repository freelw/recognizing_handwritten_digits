import torch
import torch.nn as nn

gamma = torch.tensor([0, 0.01, 0.02, 0.03, 0.04])
input1 = torch.tensor(
    [[0, 0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8, 0.9]]
)
input2 = torch.tensor(
    [[0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]],
    dtype=torch.float32
)

gamma.requires_grad_(True) 
input1.requires_grad_(True)
input2.requires_grad_(True)

res = input1 * gamma + input2 * gamma
res.retain_grad()

res_grad = torch.tensor(
    [[0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]],
    dtype=torch.float32
)

res.backward(gradient=res_grad)

print("res: ", res)
print("res grad: ", res.grad)
print("gamma: ", gamma)
print("input1: ", input1)
print("input2: ", input2)
print("gamma grad: ", gamma.grad)
print("input1 grad: ", input1.grad)
print("input2 grad: ", input2.grad)
print("res grad: ", res.grad)