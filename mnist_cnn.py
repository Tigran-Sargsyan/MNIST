import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.nn.functional as F

""" Loading MNIST dataset from torchvision and normalizing the data """
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

""" Splitting the dataset into training and validation sets """
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

fig = plt.figure(figsize=(2, 2))
image, label = train_data[0]
plt.imshow(image.squeeze(), cmap="gray")
print("Label:", label)
print("shape:", image.shape)

""" Implementing LeNets architecture and resizing our 28x28 images -> 32x32 to feed to our CNN """
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Adding padding of size 2 to the input image
        x = F.pad(x, (2, 2, 2, 2))
        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 16 * 5 * 5)  # Flattening the tensor
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x


""" Hyperparameters and training, test loops """
model = LeNet()
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
batch_size = 2 ** 6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.92, 0.999), eps=1e-08)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Computing prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return correct


""" Loading the data into dataloaders """
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)  # No need to shuffle the test data

epochs = 25

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)

# 98.6 accuracy after 25 epochs!
