import torch
import torch.nn as nn
import math
from torch.nn.init import constant_

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

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


# 定义初始化权重的钩子函数
def init_weights(module, input):
    # print("init_weights 0")
    # print(module)
    if isinstance(module, nn.Linear):
        constant_(module.weight, 1)
        module.weight.data[0, 0] = 0.1
        # eye_(module.weight)
        #print("init_weights")
        # 移除钩子，保证只执行一次
        module._forward_pre_hooks.pop(list(module._forward_pre_hooks.keys())[0])

class MultiHeadAttention:
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_q.register_forward_pre_hook(init_weights)
        self.W_k.register_forward_pre_hook(init_weights)
        self.W_v.register_forward_pre_hook(init_weights)
        self.W_o.register_forward_pre_hook(init_weights)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

class AddNorm():  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
        # 设定 gamma（weight）的初始化值
        # 例如，将 gamma 初始化为全 1
        nn.init.ones_(self.ln.weight)
        # 设定 beta（bias）的初始化值
        # 例如，将 beta 初始化为全 0
        nn.init.zeros_(self.ln.bias)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

def init_weights_ffn(module, input):
    if isinstance(module, nn.Linear):
        constant_(module.weight, 1)
        constant_(module.bias, 0)
        module.weight.data[0, 0] = 0.1
        # eye_(module.weight)
        #print("init_weights")
        # 移除钩子，保证只执行一次
        module._forward_pre_hooks.pop(list(module._forward_pre_hooks.keys())[0])

class PositionWiseFFN():  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens, bias=True)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs, bias=True)

        self.dense1.register_forward_pre_hook(init_weights_ffn)
        self.dense2.register_forward_pre_hook(init_weights_ffn)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        attention_res = self.attention.forward(X, X, X, valid_lens)
        #print("attention_res:", attention_res)
        Y = self.addnorm1.forward(X, attention_res)
        #print("addnorm1 res:", Y)
        ffn_res = self.ffn.forward(Y)
        #print("ffn_res:", ffn_res)
        res = self.addnorm2.forward(Y, ffn_res)
        #print("addnorm2 res:", res)
        return res

class PositionalEncoding:  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
def build_my_embedding(vocab_size, num_hiddens):
    matrix = torch.zeros((vocab_size, num_hiddens))
    matrix.fill_(1)
    for i in range(vocab_size):
        matrix[i, 0] = 0.1 * i
    return matrix

class TransformerEncoder():
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        self.num_hiddens = num_hiddens
        #self.embedding = nn.Embedding(vocab_size, num_hiddens)

        self.embedding = build_my_embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        embs = X @ self.embedding
        embs.requires_grad = True
        #print("embs:", embs)
        X = self.pos_encoding.forward(embs * math.sqrt(self.num_hiddens))
        #print("pos_encoding res:", X)
        cnt = 0
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            #print("blk", cnt, "res:", X)
            cnt += 1
        return X, embs

def test():
    x = torch.tensor(
        [[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
        ],
        [[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
        ], dtype=torch.float)

    # uint num_hiddens = 16;
    # uint num_blks = 2;
    # float dropout = 0;
    # uint ffn_num_hiddens = 4;
    # uint num_heads = 4;
    # uint vocab_size = 4;

    num_hiddens = 16
    num_blks = 2
    dropout = 0
    ffn_num_hiddens = 4
    num_heads = 4
    vocab_size = 4

    encoder = TransformerEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout)

    res, embs = encoder.forward(x, None)

    # print("res:", res)
    # print("res shape:", res.shape)

    loss = nn.CrossEntropyLoss()

    res = res.reshape(-1, res.shape[-1])

    res.retain_grad()

    print("res:", res)

    labels = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)

    loss_value = loss(res, labels)

    #print("loss_value:", loss_value)

    loss_value.backward()

    #print("embs:", embs)
    #print("embs.grad:", embs.grad)
    

if '__main__' == __name__:
    test()