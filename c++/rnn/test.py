import torch
import collections
import torch.nn.functional as F
global g_vocab_size
g_vocab_size = 28
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

emit_clip = 0

def clip_gradients(grad_clip_val, model):
    global emit_clip
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        emit_clip += 1
        for param in params:
            # print ("norm : ", norm)
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
    
    def output_layer(self, h):
        return self.W @ h + self.B
    
    def forward(self, inputs, state):
        h = self.rnn.forward(inputs, state)
        hh = torch.stack(h, 1).squeeze(2)
        return self.output_layer(hh)

    def parameters(self):
        return [self.rnn.W_xh, self.rnn.W_hh, self.rnn.b_h, self.W, self.B]

    def predict(self, prefix, num_preds):
        preprocess_prefix = []
        for item in list(prefix):
            #print ("item : ", item)
            preprocess_prefix.append(torch.tensor(one_hot(to_index(item), g_vocab_size), dtype=torch.float32))

        state = self.rnn.forward(preprocess_prefix, None)
        output_states = [state[-1]]
        outputs = []
        for i in range(num_preds):
            last_state = output_states[-1]
            #print("last_state : ", last_state)
            output = self.output_layer(last_state)
            # print("output : ", output)
            predict_index = torch.argmax(output).item()
            #print("predict_index : ", predict_index, " char : ", to_char(predict_index))
            outputs.append(to_char(predict_index))
            input = torch.tensor(one_hot(predict_index, g_vocab_size), dtype=torch.float32)
            state = self.rnn.forward([input], last_state)
            output_states.append(state[-1])
        print("prefix : ", prefix)
        print("predict : ", "".join(outputs))

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
    for e in range(30):
        output = rnnlm.forward(inputs, None)
        # print("output : ", output)
        # loss
        loss = torch.nn.CrossEntropyLoss()
        l = loss(output.T, labels)
        print("e : ", e, "loss : ", l)
        optimizer.zero_grad()
        l.backward()
        clip_gradients(1, rnnlm)
        optimizer.step()
    # for param in rnnlm.parameters():
    #     print(param)
    #     print(param.grad)

def get_timemachine():
    with open("../../resources/timemachine_preprocessed.txt") as f:
    #with open("../../resources/timemachine_middle.txt") as f:
        return f.read()

def tokenize(text):
    return list(text)

def one_hot(x, vocab_size):
    ret = []
    for i in range(vocab_size):
        ret.append([0])
    ret[x][0] = 1
    return ret

def to_index(x):
    if (x == ' '):
        return 26
    if (x >= 'a' and x <= 'z'):
        return ord(x) - ord('a')
    return 27

def to_char(index):
    if (index == 26):
        return ' '
    if (index < 26 and index >= 0):
        return chr(ord('a') + index)
    return '?'

def load_data(num_steps=32):
    text = get_timemachine()
    tokens = tokenize(text)
    X = []
    Y = []
    for i in range(len(tokens) - num_steps):
        x = []
        y = []
        for j in range(num_steps):
            pos = i + j
            index_x = to_index(tokens[pos])
            index_y = to_index(tokens[pos + 1])
            x.append(index_x)
            y.append(index_y)
        X.append(x)
        Y.append(y)
    return X, Y

def train_llm():
    num_steps = 32
    num_hiddens = 32
    vocab_size = g_vocab_size

    rand = True
    X, Y = load_data(num_steps)
    print("data size : ", len(X))
    rnn = Rnn(vocab_size, num_hiddens, 0.01, rand)
    rnnlm = RnnLM(rnn, vocab_size, rand)
    optimizer = torch.optim.Adam(rnnlm.parameters(), lr=0.001)  # Change learning rate to 0.001
    loss_fn = torch.nn.CrossEntropyLoss()
    
    global emit_clip
    for epoch in range(100):
        loss_sum = 0
        print("epoch ", epoch, " started.")
        length = len(X)
        emit_clip = 0
        for i in range(length):
            x, y = X[i], Y[i]
            inputs = []
            for item in x:
                inputs.append(torch.tensor(one_hot(item, vocab_size), dtype=torch.float32))
            labels = torch.tensor(y, dtype=torch.long)
            output = rnnlm.forward(inputs, None)
            loss = loss_fn(output.T, labels)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(1, rnnlm)
            optimizer.step()
            print("\r[", i, "/", length, "]", end="", flush=True)
        print("epoch : ", epoch, " loss : ", loss_sum / length, " emit_clip : ", emit_clip)
    
        prefixs = [
            "time traveller",
            "the time machine",
            "expounding a recondite",
            " traveller for so",
            "it has",
            "so most people",
            "is simply ",
            " we cannot move about",
            "and the still",
        ]

        for prefix in prefixs:
            rnnlm.predict(prefix, 20)

def testcrossentropy():
    x = [556.225, 1919.83, 2769.87, 2810.71, 2811.33, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34, 2811.34]
    x = torch.tensor(x, dtype=torch.float32).view(1, 32)
    x.requires_grad_()
    labels = [0]
    labels = torch.tensor(labels, dtype=torch.long).view(1)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(x, labels)
    print(loss.item())
    loss.backward()
    print(x.grad)

if __name__ == '__main__':
    # teststack()
    # testgrad()
    train_llm()

    # testcrossentropy()
