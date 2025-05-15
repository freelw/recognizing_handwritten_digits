import torch
import torch.nn as nn
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

x = torch.tensor(
    [[[0, 0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6, 0.7]],

    [[0.8, 0.9, 1, 1.1],
    [1.2, 1.3, 1.4, 1.5]]]
)
x.requires_grad_(True)  # Enable gradient calculation

print("x :", x)
masked_softmax_res = masked_softmax(x, torch.tensor([[1, 3],[2, 4]]))

loss_fn = nn.CrossEntropyLoss()
labels = torch.tensor([0, 1, 2, 3])
loss = loss_fn(masked_softmax_res.reshape(-1, 4), labels)
print("masked_softmax_res: ", masked_softmax_res)
print("loss: ", loss)
loss.backward()
print("x.grad: ", x.grad)