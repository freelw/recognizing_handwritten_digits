import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        masked_X = X.clone()  # Create a copy of X to avoid in-place operations
        masked_X[~mask] = value
        return masked_X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

queries = torch.tensor(
    [[[10.6, 10.6]],

    [[10.6, 10.6]]]
    , dtype=torch.float32
)


keys = torch.tensor(
    [[[55.5, 55.5],
  [55.5, 55.5],
  [10, 0.1],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5]],

 [[55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5],
  [55.5, 55.5]]]
  , dtype=torch.float32
)
values = torch.tensor(
  [[[0, 0.1, 0.2, 0.3],
  [0.4, 0.5, 0.6, 0.7],
  [0.8, 0.9, 1, 1.1],
  [1.2, 1.3, 1.4, 1.5],
  [1.6, 1.7, 1.8, 1.9],
  [2, 2.1, 2.2, 2.3],
  [2.4, 2.5, 2.6, 2.7],
  [2.8, 2.9, 3, 3.1],
  [3.2, 3.3, 3.4, 3.5],
  [3.6, 3.7, 3.8, 3.9]],

 [[4, 4.1, 4.2, 4.3],
  [4.4, 4.5, 4.6, 4.7],
  [4.8, 4.9, 5, 5.1],
  [5.2, 5.3, 5.4, 5.5],
  [5.6, 5.7, 5.8, 5.9],
  [6, 6.1, 6.2, 6.3],
  [6.4, 6.5, 6.6, 6.7],
  [6.8, 6.9, 7, 7.1],
  [7.2, 7.3, 7.4, 7.5],
  [7.6, 7.7, 7.8, 7.9]]]
  , dtype=torch.float32
)
valid_lens = torch.tensor([2, 6])
queries.requires_grad_(True)  # Enable gradient calculation
keys.requires_grad_(True)  # Enable gradient calculation
values.requires_grad_(True)  # Enable gradient calculation

attention = DotProductAttention(dropout=0)
attention_res = attention(queries, keys, values, valid_lens)
softmax_res = F.softmax(attention_res, dim=-1)
labels = torch.tensor([0, 1])
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(softmax_res.reshape(-1, 4), labels)
loss.backward()
print ("loss: ", loss.item())
print ("softmax_res: ", softmax_res)
print ("q grad: ", queries.grad)
print ("k grad: ", keys.grad)
print ("v grad: ", values.grad)