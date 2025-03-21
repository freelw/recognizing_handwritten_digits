import torch

if '__main__' == __name__:

    w = torch.randn(256, 256) * 0.01
    x = torch.randn(256, 256) * 0.01

    for i in range(10000):
        x = torch.mm(w, x)