import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mnist_loader

def run():
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print device info
    print(device)

    # load model from file model.pth
    model = torch.load("model.pth", weights_only=False)  # Set weights_only to False
    # load model parameters from file model_parameters.pth
    model.load_state_dict(torch.load("model_parameters.pth", weights_only=True))  # Ensure weights_only is True
    images = mnist_loader.load_mnist_images("../resources/train-images-idx3-ubyte")
    labels = mnist_loader.load_mnist_labels("../resources/train-labels-idx1-ubyte")
    images = torch.tensor(images, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)  # Change dtype to long

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=128)  # Change batch size to 128
    print("start testing")
    correct = 0
    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        output = model(batch_images)
        correct += (torch.argmax(output, dim=1) == batch_labels).sum().item()  # Change accuracy calculation
    print("accuracy: ", correct / len(images))

if __name__ == "__main__":
    run()