import torch
import torch.nn as nn
import cmath

def test0():
    x = [[1]]
    # construct a tensor with shape (1, 10) from x
    x = torch.tensor(x, dtype=torch.float32).view(1, 1)

    y = [[2]]
    y = torch.tensor(y, dtype=torch.float32).view(1, 1)

    x.requires_grad = True
    y.requires_grad = True

    print('x :', x)
    print('y :', y)

    z = nn.ReLU()(x * y + x)
    print(z)
    # log z
    z = torch.log(z)
    print(z)
    z = torch.exp(z)

    print(z)
    z.backward()

    print(x.grad)
    print(y.grad)


def test1():
    # 3 layer neural network
    # input layer: 10 neurons
    # hidden layer: 20 neurons
    # output layer: 11 neurons 
    # activation function: ReLU
    # loss function: CrossEntropyLoss

    # input from 0 to 9
    x = [[i for i in range(4)]]
    x = torch.tensor(x, dtype=torch.float32).view(1, 4)
    # target is 1
    y = [1]
    y = torch.tensor(y, dtype=torch.long).view(1)
    # model
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )
    # initialize the weights to 0.1
    # initialize the bias to 0
    model[0].weight.data.fill_(0.1)
    model[0].bias.data.fill_(0.1)
    model[2].weight.data.fill_(0.1)
    model[2].bias.data.fill_(0.1)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    res = model(x)
    res.retain_grad()
    print("forward result: ", res)
    print("result grad: ", res.grad)
    loss = loss_fn(res, y)
    print("loss: ", loss)
    loss.backward()

    #show the gradients
    print(model[0].weight.grad)
    print(model[0].bias.grad)
    print(model[2].weight.grad)
    print(model[2].bias.grad)

def testce():

    # input from 0 to 9
    x = [[i for i in range(4)]]
    x = torch.tensor(x, dtype=torch.float32).view(1, 4)
    # target is 1
    y = [1]
    y = torch.tensor(y, dtype=torch.long).view(1)
    # model
   
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(x, y)

    print(loss.item())

    

if __name__ == '__main__':

    test1()
    testce()