import torch
import collections
import torch.nn.functional as F

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
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, rand = True):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        if rand:
            self.W_xh = torch.randn(num_hiddens, num_inputs) * sigma
            self.W_hh = torch.randn(num_hiddens, num_hiddens) * sigma
            self.b_h = torch.randn(num_hiddens, 1) * sigma
        else :
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
            print ("norm : ", norm)
            param.grad[:] *= grad_clip_val / norm
class RnnLM:
    def __init__(self, rnn, vocab_size, rand = True):
        self.rnn = rnn
        self.vocab_size = vocab_size
        if rand:
            self.W = torch.randn(vocab_size, rnn.num_hiddens)
            self.B = torch.randn(vocab_size, 1)
        else:
            tensor_1d = torch.arange(0.1, 0.1 * (vocab_size * rnn.num_hiddens+1), 0.1)
            self.W = tensor_1d.reshape(vocab_size, rnn.num_hiddens)
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
    rnn = Rnn(vocab_size, 4, 0.01, False)

    inputs = [torch.tensor([[1], [0], [0]], dtype=torch.float32),
                torch.tensor([[0], [1], [0]], dtype=torch.float32),
                torch.tensor([[0], [0], [1]], dtype=torch.float32),
                torch.tensor([[1], [0], [0]], dtype=torch.float32)
                ]
    labels = torch.tensor([2, 1, 2, 0], dtype=torch.long)


    # inputs = [torch.tensor([[1], [0], [0]], dtype=torch.float32),
    #             torch.tensor([[0], [1], [0]], dtype=torch.float32)
    #             ]
    # print("inputs : ", inputs)
    # labels = torch.tensor([2, 1], dtype=torch.long)

    # inputs = [torch.tensor([[1], 
    #                         [0], 
    #                         [0]], dtype=torch.float32)]
    # labels = torch.tensor([2], dtype=torch.long)
    rnnlm = RnnLM(rnn, vocab_size, False)
    optimizer = torch.optim.Adam(rnnlm.parameters(), lr=0.001)
    for e in range(3):
        output = rnnlm.forward(inputs, None)
        # print("output : ", output)
        # loss
        loss = torch.nn.CrossEntropyLoss()
        l = loss(output.T, labels)
        print("loss : ", l)
        optimizer.zero_grad()
        l.backward()
        clip_gradients(1, rnnlm)
        optimizer.step()
        clip_gradients(1, rnnlm)
        for param in rnnlm.parameters():
            print(param.grad)

class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

def get_timemachine():
    with open("../../resources/timemachine_preprocessed.txt") as f:
        return f.read()

def tokenize(text):
    return list(text)[:1000] # fix me

def one_hot(x, vocab_size):
    ret = []
    for i in range(vocab_size):
        ret.append([0])
    ret[x][0] = 1
    return ret

def load_data(num_steps=32):
    text = get_timemachine()
    tokens = tokenize(text)
    # print(','.join(tokens[:30]))
    vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    X = []
    Y = []
    for i in range(len(corpus) - num_steps):
        x = []
        y = []
        for j in range(num_steps):
            x.append(corpus[j])
            y.append(corpus[j+1])
        X.append(x)
        Y.append(y)
    return X, Y, vocab

def train_llm():
    num_steps = 32
    num_hiddens = 32

    X, Y, vocab = load_data(num_steps)
    rnn = Rnn(len(vocab), num_hiddens)
    rnnlm = RnnLM(rnn, len(vocab))
    optimizer = torch.optim.Adam(rnnlm.parameters(), lr=0.001)  # Change learning rate to 0.001
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        loss_sum = 0
        print("epoch ", epoch, " started.")
        length = len(X)
        for i in range(length):
            if i % 10000 == 0:
                print("[", i, "/", length, "]")
            x, y = X[i], Y[i]
            inputs = []
            for item in x:
                inputs.append(torch.tensor(one_hot(item, len(vocab)), dtype=torch.float32))
            labels = torch.tensor(y, dtype=torch.long)
            output = rnnlm.forward(inputs, None)
            loss = loss_fn(output.T, labels)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(1, rnnlm)
            optimizer.step()
        print("epoch : ", epoch, " loss : ", loss_sum / length)

if __name__ == '__main__':
    #teststack()
    testgrad()
    #train_llm()
