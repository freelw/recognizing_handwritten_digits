import torch
import torch.nn as nn
import cmath

def test1():
    # 3 layer neural network
    # input layer: 10 neurons
    # hidden layer: 20 neurons
    # output layer: 11 neurons 
    # activation function: ReLU
    # loss function: CrossEntropyLoss

    # input from 0 to 9
    x = [[i+10 for i in range(2)]]
    x = torch.tensor(x, dtype=torch.float32).view(1, 2)
    print("x: ", x)
    print("x shape:", x.shape)
    # target is 1
    y = [1]
    y = torch.tensor(y, dtype=torch.long).view(1)
    # model
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 3)
    )
    # initialize the weights to 0.1
    # initialize the bias to 0
    model[0].weight.data.fill_(0.1)
    model[0].bias.data.fill_(0.1)
    model[2].weight.data.fill_(0.1)
    model[2].bias.data.fill_(0.1)

    model[0].weight.data[0, 0] = 0.9
    model[0].weight.data[1, 0] = -0.9

    print("model[0].weight : ", model[0].weight)

    model[2].weight.data[0, 0] = 0.9
    model[2].weight.data[1, 0] = -0.9


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

    # print(model[0].weight)
    # print(model[0].weight.grad)
    # print(model[0].bias)
    # print(model[0].bias.grad)
    # print(model[2].weight)
    # print(model[2].weight.grad)
    # print(model[2].bias)
    # print(model[2].bias.grad)

if __name__ == "__main__":
    test1()