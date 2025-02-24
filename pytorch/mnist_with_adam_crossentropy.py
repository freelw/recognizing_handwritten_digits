import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mnist_loader

def run():
    images = mnist_loader.load_mnist_images("../resources/train-images-idx3-ubyte")
    labels = mnist_loader.load_mnist_labels("../resources/train-labels-idx1-ubyte")

    print(len(images))
    print(len(labels))

    boundary = 50000

    train_images = images[:boundary]
    train_labels = labels[:boundary]

    test_images = images[boundary:]
    test_labels = labels[boundary:]

    # use gpu if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use cpu single thread
    device = torch.device("cpu")
    # use 1 thread
    torch.set_num_threads(1)

    # print device info
    print(device)
    # copy all images and labels to the device
    train_images = torch.tensor(train_images, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)  # Change dtype to long
    test_images = torch.tensor(test_images, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)  # Change dtype to long

    # create DataLoader
    train_dataset = TensorDataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Change batch size to 128

    test_dataset = TensorDataset(test_images, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    # custom weight initialization function
    def initialize_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.normal_(m.bias, mean=0.0, std=0.02)

    # use gpu device
    model = nn.Sequential(
        nn.Linear(784, 30),
        nn.ReLU(),
        nn.Linear(30, 10)
    ).to(device)  # Move model to the device
    model.apply(initialize_weights)  # Apply the custom weight initialization
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Change learning rate to 0.001
    print("start training")

    for epoch in range(100):
        for batch_images, batch_labels in train_dataloader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_images)
            loss = loss_fn(output, batch_labels)
            print(loss.item())
            loss.backward()
            optimizer.step()
        print("epoch: ", epoch)
        # for each 10 epoch, calculate the accuracy
        if epoch % 1 == 0:
            correct = 0
            for batch_images, batch_labels in test_dataloader:
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                output = model(batch_images)
                correct += (torch.argmax(output, dim=1) == batch_labels).sum().item()  # Change accuracy calculation
            print("correct: ", correct, "/" , len(images) - boundary)

    # Save the model and parameters to disk
    torch.save(model.state_dict(), "model_parameters.pth")
    torch.save(model, "model.pth")

if __name__ == '__main__':
    run()