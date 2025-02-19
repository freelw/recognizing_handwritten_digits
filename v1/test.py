import torch
import torch.nn as nn
import cmath

if __name__ == '__main__':

    x = [[1]]
    # construct a tensor with shape (1, 10) from x
    x = torch.tensor(x, dtype=torch.float32).view(1, 1)

    y = [[2]]
    y = torch.tensor(y, dtype=torch.float32).view(1, 1)

    x.requires_grad = True
    y.requires_grad = True

    print('x :', x)
    print('y :', y)

    z = nn.ReLU()(x * y + x)
    print(z)
    # log z
    z = torch.log(z)
    print(z)
    z = torch.exp(z)

    print(z)
    z.backward()

    print(x.grad)
    print(y.grad)