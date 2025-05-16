import torch
import torch.nn as nn

input = torch.tensor(
[[0, 0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6, 0.7],
  [0.8, 0.9, 1, 1.1]],
)
input.requires_grad_(True)  # Enable gradient calculation
w = torch.tensor(
    [[0, 0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1, 1.1],
  [1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
  [1.8, 1.9, 2, 2.1, 2.2, 2.3]]
)
w.requires_grad_(True)  # Enable gradient calculation
print("input: ", input)
print("w :", w)

labels = torch.tensor([0, 1, 2])
loss_fn = nn.CrossEntropyLoss()
res = input @ w / 10
print("res: ", res)
loss = loss_fn(res, labels)
loss.backward()
print("loss: ", loss)
print("input grad: ", input.grad)
print("w grad: ", w.grad)