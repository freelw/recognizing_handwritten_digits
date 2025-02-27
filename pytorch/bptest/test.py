import torch
import torch.nn as nn


if __name__ == '__main__':

    x = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.05],
         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.05]]
    # construct a tensor with shape (1, 10) from x
    x = torch.tensor(x).view(2, 10)
    x.requires_grad = True
    print (x)
    print(x.shape)

    loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(x, torch.tensor([8, 8]))
    loss = loss_fn(x, torch.tensor([6, 8]))

    print(loss.item())

    loss.backward()

    print(x.grad)