import torch
import torch.nn as nn

input = torch.tensor(
    [[0, 0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6, 0.7],
    [0.8, 0.9, 1, 1.1]]
)
w1 = torch.tensor(
    [[0, 0.1, 0.2],
    [0.3, 0.4, 0.5]]
)
w2 = torch.tensor(
    [[0, 0.1, 0.2],
    [0.3, 0.4, 0.5]]
)

input.requires_grad_(True) 
w1.requires_grad_(True)
w2.requires_grad_(True)

res = w1 @ input + w2 @ input
res.retain_grad()

res_grad = torch.tensor(
    [[0, 1, 2, 3],
    [4, 5, 6, 7]],
    dtype=torch.float32
)

res.backward(gradient=res_grad)

print("res: ", res)
print("res grad: ", res.grad)
print("input: ", input)
print("w1: ", w1)
print("w2: ", w2)
print("input grad: ", input.grad)
print("w1 grad: ", w1.grad)
print("w2 grad: ", w2.grad)
