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

queries = torch.tensor(
[[[0, 0.001]],

 [[0.002, 0.003]]]
)
queries.requires_grad_(True)  # Enable gradient calculation
keys = torch.tensor(
 [[[0, 0.001],
  [0.002, 0.003],
  [0.004, 0.005],
  [0.006, 0.007],
  [0.008, 0.009],
  [0.01, 0.011],
  [0.012, 0.013],
  [0.014, 0.015],
  [0.016, 0.017],
  [0.018, 0.019]],

 [[0.02, 0.021],
  [0.022, 0.023],
  [0.024, 0.025],
  [0.026, 0.027],
  [0.028, 0.029],
  [0.03, 0.031],
  [0.032, 0.033],
  [0.034, 0.035],
  [0.036, 0.037],
  [0.038, 0.039]]]
)
keys.requires_grad_(True)  # Enable gradient calculation


labels = torch.tensor([0, 1])
loss_fn = nn.CrossEntropyLoss()
res_bmm = torch.bmm(queries, keys.transpose(1, 2))
res_bmm.retain_grad()  # Retain gradient for res_bmm
print("res_bmm: ", res_bmm)
softmax_res = masked_softmax(res_bmm/math.sqrt(2), torch.tensor([2, 6]))
softmax_res.retain_grad()  # Retain gradient for softmax_res
loss = loss_fn(softmax_res.reshape(-1, 10), labels)
loss.backward()
print("softmax_res: ", softmax_res)
print("loss: ", loss)
print("queries grad: ", queries.grad)
print("keys grad: ", keys.grad)
print("res_bmm grad: ", res_bmm.grad)
print("softmax_res grad: ", softmax_res.grad)