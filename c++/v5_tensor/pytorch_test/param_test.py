import torch
import torch.nn as nn
import math
from torch.nn.init import constant_
from torch.nn import functional as F

class TransformerEncoderBlock(nn.Module):
    """A Transformer encoder block."""
    def __init__(self,):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )

    def forward(self, X):
        pass

class TransformerEncoder(nn.Module):
    """The Transformer encoder."""
    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear(1, 1)
        self.blks = nn.Sequential()
        for i in range(2):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock())
    def forward(self, X):

        return X

# main
if __name__ == "__main__":
    encoder = TransformerEncoder()
    # print all parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    