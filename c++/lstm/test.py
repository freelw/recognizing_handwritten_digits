import torch
import torch.nn as nn
global g_vocab_size
g_vocab_size = 28

class LSTM:
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(num_hiddens, num_inputs),
                          init_weight(num_hiddens, num_hiddens),
                          init_weight(num_hiddens, 1))
        self.W_xi, self.W_hi, self.b_i = triple()
        self.W_xf, self.W_hf, self.b_f = triple()
        self.W_xo, self.W_ho, self.b_o = triple()
        self.W_xc, self.W_hc, self.b_c = triple()

    def forward(self, X, c, h):
        assert len(X) > 0 
        assert X[0].shape[0] == self.num_inputs
        if h is None:
            h = torch.zeros(self.num_hiddens, 1)
        if c is None:
            c = torch.zeros(self.num_hiddens, 1)
        output = []
        for x in X:
            i = torch.sigmoid(self.W_xi @ x + self.W_hi @ h + self.b_i)
            f = torch.sigmoid(self.W_xf @ x + self.W_hf @ h + self.b_f)
            o = torch.sigmoid(self.W_xo @ x + self.W_ho @ h + self.b_o)
            c_tilda = torch.tanh(self.W_xc @ x + self.W_hc @ h + self.b_c)
            c = f * c + i * c_tilda
            h = o * torch.tanh(c)
            output.append((h, c))
        return output

class RnnLM:
    def __init__(self, rnn, vocab_size):
        self.rnn = rnn
        self.vocab_size = vocab_size
        tensor_1d = torch.arange(0.1, 0.1 * (vocab_size * rnn.num_hiddens+1), 0.1)
        self.W = tensor_1d.reshape(vocab_size, rnn.num_hiddens)
        self.B = torch.empty(vocab_size, 1).fill_(0.1)
        self.W.requires_grad_()
        self.B.requires_grad_()
    
    def output_layer(self, h):
        return self.W @ h + self.B
    
    def forward(self, inputs, state_c, state_h):
        h = [x for (x, _) in self.rnn.forward(inputs, state_c, state_h)]
        # print("h : ", h)
        hh = torch.stack(h, 1).squeeze(2)
        return self.output_layer(hh)

    def parameters(self):
        return [
            self.rnn.W_xi, self.rnn.W_hi, self.rnn.b_i,
            self.rnn.W_xf, self.rnn.W_hf, self.rnn.b_f,
            self.rnn.W_xo, self.rnn.W_ho, self.rnn.b_o,
            self.rnn.W_xc, self.rnn.W_hc, self.rnn.b_c,
            self.W, self.B
        ]

    def predict(self, prefix, num_preds):
        preprocess_prefix = []
        for item in list(prefix):
            preprocess_prefix.append(torch.tensor(one_hot(to_index(item), g_vocab_size), dtype=torch.float32))

        state = self.rnn.forward(preprocess_prefix, None, None)
        output_states = [state[-1]]
        outputs = []
        for i in range(num_preds):
            last_state_c, last_state_h = output_states[-1]
            output = self.output_layer(last_state_h)
            predict_index = torch.argmax(output).item()
            outputs.append(to_char(predict_index))
            input = torch.tensor(one_hot(predict_index, g_vocab_size), dtype=torch.float32)
            state = self.rnn.forward([input], last_state_h, last_state_c)
            output_states.append(state[-1])
        print("prefix : ", prefix)
        print("predict : ", prefix, "".join(outputs))

def get_timemachine():
    with open("../../resources/timemachine_preprocessed.txt") as f:
        return f.read()

def tokenize(text):
    return list(text)[:2000] # fix me
    #return list(text)

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

    X, Y = load_data(num_steps)
    rnn = LSTM(vocab_size, num_hiddens, 0.01)
    rnnlm = RnnLM(rnn, vocab_size)
    optimizer = torch.optim.Adam(rnnlm.parameters(), lr=0.001)  # Change learning rate to 0.001
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(30):
        loss_sum = 0
        print("epoch ", epoch, " started.")
        length = len(X)
        for i in range(length):
            x, y = X[i], Y[i]
            inputs = []
            for item in x:
                inputs.append(torch.tensor(one_hot(item, vocab_size), dtype=torch.float32))
            labels = torch.tensor(y, dtype=torch.long)
            output = rnnlm.forward(inputs, None, None)
            loss = loss_fn(output.T, labels)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch : ", epoch, " loss : ", loss_sum / length)
    rnnlm.predict("time traveller", 20)
    rnnlm.predict("the time machine", 20)
    rnnlm.predict("expounding a recondite", 20)
    rnnlm.predict(" traveller for so", 20)

if __name__ == '__main__':
    train_llm()
