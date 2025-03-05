import torch

def teststack():
    x = [[0.5, 1, ],
          [1, 1, ],
          [0.3, 1, ]]

    x1 = [[0.6, 2, ],
          [2, 2, ],
          [0.2, 2, ]]

    y = torch.tensor(x, dtype=torch.float32).view(3, 2)
    y1 = torch.tensor(x1, dtype=torch.float32).view(3, 2)
    arr = [y, y1]
    print (arr)
    print (torch.stack(arr, 1))

class Rnn:
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        self.W_xh = torch.empty(num_hiddens, num_inputs).fill_(0.1)
        
        self.W_hh = torch.empty(num_hiddens, num_hiddens).fill_(0.1)
        self.b_h = torch.empty(num_hiddens, 1).fill_(0.1)

        self.W_hh.requires_grad_()
        self.W_xh.requires_grad_()
        self.b_h.requires_grad_()

    def forward(self, X, h):
        assert len(X) > 0 
        assert X[0].shape[0] == self.num_inputs
        if h is None:
            h = torch.zeros(self.num_hiddens, 1)
        output = []
        for x in X:
            h = torch.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            output.append(h)
        return output

def clip_gradients(grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
class RnnLM:
    def __init__(self, rnn, vocab_size):
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.W = torch.empty(vocab_size, rnn.num_hiddens)
        tensor_1d = torch.arange(0.1, 0.1 * (vocab_size * rnn.num_hiddens+1), 0.1)
        self.W = tensor_1d.reshape(vocab_size, rnn.num_hiddens)
        print("W : ", self.W)
        #torch.empty(vocab_size, rnn.num_hiddens).fill_(0.1)
        self.B = torch.empty(vocab_size, 1).fill_(0.1)
        self.W.requires_grad_()
        self.B.requires_grad_()
    
    def forward(self, inputs, state):
        h = self.rnn.forward(inputs, state)
        hh = torch.stack(h, 1).squeeze(2)
        output = self.W @ hh + self.B
        return output

    def parameters(self):
        return [self.rnn.W_xh, self.rnn.W_hh, self.rnn.b_h, self.W, self.B]

def testgrad():
    vocab_size = 3
    rnn = Rnn(vocab_size, 4)

    # inputs = [torch.tensor([[1], [0], [0]], dtype=torch.float32),
    #             torch.tensor([[0], [1], [0]], dtype=torch.float32),
    #             torch.tensor([[0], [0], [1]], dtype=torch.float32),
    #             torch.tensor([[1], [0], [0]], dtype=torch.float32)
    #             ]
    # labels = torch.tensor([2, 1, 2, 0], dtype=torch.long)


    inputs = [torch.tensor([[1], [0], [0]], dtype=torch.float32),
                torch.tensor([[0], [1], [0]], dtype=torch.float32)
                ]
    labels = torch.tensor([2, 1], dtype=torch.long)

    # inputs = [torch.tensor([[1], 
    #                         [0], 
    #                         [0]], dtype=torch.float32)]
    # labels = torch.tensor([2], dtype=torch.long)

    rnnlm = RnnLM(rnn, vocab_size)
    output = rnnlm.forward(inputs, None)
    print("output : ", output)

    # loss
    loss = torch.nn.CrossEntropyLoss()
    l = loss(output.T, labels)
    print("loss : ", l)
    l.backward()
    clip_gradients(1, rnnlm)
    for param in rnnlm.parameters():
        print(param.grad)

if __name__ == '__main__':
    #teststack()
    testgrad()
