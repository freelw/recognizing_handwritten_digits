import torch
import torch.nn as nn
from torch.nn import functional as F

input = torch.tensor(
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    dtype=torch.float32
)
input.requires_grad_(True) 

labels = torch.tensor(
    [2, 3],
    dtype=torch.long
)


loss = F.cross_entropy(input, labels, reduction="none")
print("loss: ", loss)
loss_avg = loss.sum() / labels.shape[0]
loss_avg.backward()
print("input grad : ", input.grad)
