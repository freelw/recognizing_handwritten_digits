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

        print ("W_xh : ", self.W_xh)
        print ("W_hh : ", self.W_hh)
        print ("b_h : ", self.b_h)


    def forward(self, X, h):
        # assert x is array 
        assert len(X) > 0 
        assert X[0].shape[0] == self.num_inputs
        if h is None:
            h = torch.zeros(self.num_hiddens, 1)
        output = []
        for x in X:
            #print ("x : ", x)
            # print ("W_xh : ", self.W_xh)
            # print ("W_hh : ", self.W_hh)
            #print ("self.W_xh @ x : ", self.W_xh @ x)
            h = torch.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
            #print ("1 h : ", h)
            output.append(h)
        return output

class RnnLM:
    def __init__(self, rnn, vocab_size):
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.W = torch.empty(rnn.num_hiddens, vocab_size).fill_(0.1)
        self.B = torch.empty(vocab_size, 1).fill_(0.1)
    
    def forward(self, inputs, state):
        h = self.rnn.forward(inputs, state)
        print ("h : ", h)
        hh = torch.stack(h, 1).squeeze()
        print("hh : ", hh)
        print("W : ", self.W)
        print("B : ", self.B)
        print("W @ hh : ", self.W @ hh)
        output = self.W @ hh + self.B
        return output

def testgrad():
    rnn = Rnn(3, 4)

    inputs = [torch.tensor([[1], [0], [0]], dtype=torch.float32),
                torch.tensor([[0], [1], [0]], dtype=torch.float32),
                torch.tensor([[0], [0], [1]], dtype=torch.float32)]

    labels = torch.tensor([[2], [1], [2]], dtype=torch.float32)

    rnnlm = RnnLM(rnn, 4)
    output = rnnlm.forward(inputs, None)
    print(output)



if __name__ == '__main__':
    #teststack()

    testgrad()
