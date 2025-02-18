import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mnist_loader

if __name__ == "__main__":
    images = mnist_loader.load_mnist_images("../resources/train-images-idx3-ubyte")
    labels = mnist_loader.load_mnist_labels("../resources/train-labels-idx1-ubyte")

    # use sgd optimizer and cross entropy loss
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print device info
    print(device)

    # copy all images and labels to the device
    images = torch.tensor(images, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)  # Change dtype to long

    # create DataLoader
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)  # Change batch size to 128

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
        # shuffle the dataloader
        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_images)
            # print output shape and batch_labels shape
            # print(output.shape, batch_labels.shape)
            loss = loss_fn(output, batch_labels)
            #print (loss.shape)
            loss.backward()
            optimizer.step()
        
        # for each epoch, print log info
        print("epoch: ", epoch)
        # for each 10 epoch, calculate the accuracy
        if epoch % 10 == 9:
            correct = 0
            for batch_images, batch_labels in dataloader:
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                output = model(batch_images)
                correct += (torch.argmax(output, dim=1) == batch_labels).sum().item()  # Change accuracy calculation
            print("accuracy: ", correct / len(images))