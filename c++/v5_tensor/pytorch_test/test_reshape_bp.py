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
    # x = [[i * 1e-5 for i in range(20)]]
    ox = torch.arange(20, dtype=torch.float32).reshape(5, 4) * 1e-5
    print ("ox: ", ox)
    ox.requires_grad_(True)  # 启用梯度计算

    x = ox.transpose(0, 1)
    x = x.reshape(-1, 2)
    print("x: ", x)
    print("x shape:", x.shape)
    # target is 1
    y = [1 for i in range(10)]
    y = torch.tensor(y, dtype=torch.long).view(10)
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

    model[2].weight.data[0, 0] = 0.9
    model[2].weight.data[1, 0] = -0.9

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    res = model(x)
    res.retain_grad()
    loss = loss_fn(res, y)
    print("loss: ", loss)
    loss.backward()

    print("ox grad: ", ox.grad)

def test2():
    ox = torch.arange(20, dtype=torch.float32).reshape(5, 4) * 1e-5
    print ("ox: ", ox)
    ox.requires_grad_(True)  # 启用梯度计算
    x = ox.reshape(-1, 2)
    print("x: ", x)
    print("x shape:", x.shape)
    # target is 1
    y = [1 for i in range(10)]
    y = torch.tensor(y, dtype=torch.long).view(10)
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
    model[2].weight.data[0, 0] = 0.9
    model[2].weight.data[1, 0] = -0.9
    # loss function
    loss_fn = nn.CrossEntropyLoss()

    res = model(x)
    res.retain_grad()
    loss = loss_fn(res, y)
    print("loss: ", loss)
    loss.backward()


    print("ox grad: ", ox.grad)


if __name__ == "__main__":
    test1()
    test2()