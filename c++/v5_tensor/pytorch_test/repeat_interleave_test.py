import torch
import torch.nn as nn
import math
from torch.nn.init import constant_

num_steps = 3
num_heads = 4
batch_size = 2

a = torch.arange(
                1, num_steps + 1)
valid_lens = a.repeat(batch_size, 1)

valid_lens_1 = torch.repeat_interleave(
                valid_lens, repeats=num_heads, dim=0)

print("a : ", a)
print("valid_lens:", valid_lens)
print("valid_lens_1:", valid_lens_1)
