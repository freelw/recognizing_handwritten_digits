import torch
import torch.nn as nn

input = torch.tensor(
[[[0, 0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6, 0.7],
  [0.8, 0.9, 1, 1.1]],

 [[1.2, 1.3, 1.4, 1.5],
  [1.6, 1.7, 1.8, 1.9],
  [2, 2.1, 2.2, 2.3]]]
)
input.requires_grad_(True)  # Enable gradient calculation
w = torch.tensor(
    [[[0, 0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1, 1.1],
  [1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
  [1.8, 1.9, 2, 2.1, 2.2, 2.3]],

 [[2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
  [3, 3.1, 3.2, 3.3, 3.4, 3.5],
  [3.6, 3.7, 3.8, 3.9, 4, 4.1],
  [4.2, 4.3, 4.4, 4.5, 4.6, 4.7]]]
)
w.requires_grad_(True)  # Enable gradient calculation
print("input: ", input)
print("w :", w)

labels = torch.tensor([0, 1, 2, 3, 4, 5])
loss_fn = nn.CrossEntropyLoss()
res_bmm = torch.bmm(input, w)
print("res_bmm: ", res_bmm)
loss = loss_fn(res_bmm.reshape(-1, 6), labels)
loss.backward()
print("loss: ", loss)
print("input grad: ", input.grad)
print("w grad: ", w.grad)