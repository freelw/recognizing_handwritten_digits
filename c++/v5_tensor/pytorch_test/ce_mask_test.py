import torch
import torch.nn as nn
import cmath
from torch.nn import functional as F

def test1():
    
    res = torch.tensor([[10, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3]], dtype=torch.float32)
    res.requires_grad_(True)
    y = torch.tensor([1, 1, 2], dtype=torch.long)
    loss = F.cross_entropy(res, y, reduction="none")
    print("loss1: ", loss)
    loss = loss.sum()/y.shape[0]
    print("loss1 avg: ", loss)
    loss.backward()

def test2():
    
    res = torch.tensor([[10, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3]], dtype=torch.float32)
    res.requires_grad_(True)
    y = torch.tensor([1, 2, 1], dtype=torch.long)
    mask = (y.reshape(-1) != 2).type(torch.float32)
    loss = F.cross_entropy(res, y, reduction="none")
    print("loss2: ", loss)
    print ("mask: ", mask)
    print ("loss * mask: ", loss * mask)
    loss_mask = (loss * mask).sum() / mask.sum()
    print("loss_mask: ", loss_mask)
    loss_mask.backward()
    print("res grad : ", res.grad)

if __name__ == "__main__":
    test1()
    print("--------")
    test2()