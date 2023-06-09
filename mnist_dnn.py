import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

"""Loading MNIST dataset from torchvision."""

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Splitting the dataset into training and validation sets
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

test_loader = DataLoader(test_data) # We don't need to shuffle our test data

def init_weights(m):
    """ A function for manually initializing weights """
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
        m.bias.data.fill_(0)

class NeuralNetwork(nn.Module):
    """
    Neural Network class implementing 4-layer Dense Neural Network with batch normalization layers,
    RELU activation function was used in hidden layers, and softmax in the output layer
     (Actually this NN returns logits and softmax is computed automatically).
    """
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.momentum = 0.75
        #self.apply(init_weights) # Default Kaiming initialization is better for ReLU
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            torch.nn.BatchNorm1d(num_features=512, eps=1e-05, momentum=self.momentum, affine=True, track_running_stats=True,
                                 device=None, dtype=None),
            nn.ReLU(),
            nn.Linear(512, 256),
            torch.nn.BatchNorm1d(num_features=256, eps=1e-05, momentum=self.momentum, affine=True, track_running_stats=True,
                                 device=None, dtype=None),
            nn.ReLU(),
            nn.Linear(256, 128),
            torch.nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=self.momentum, affine=True, track_running_stats=True,
                                 device=None, dtype=None),
            nn.ReLU(),
            nn.Linear(128, 10),
            #nn.Softmax(),          Softmax is done later, automatically
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.92, 0.999), eps=1e-08)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)

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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

# For keeping validation scores of a model trained with certain hyperparameters
val_scores = []

# For grid-searching best hyperparameters
param_grid = {
    "learning_rates": [0.01, 0.1],
    "epochs": [5, 10],
    "batch_sizes": [32, 128]
}

# Dictionary for holding best hyperparameter values found during the training
best_params = {
    "learning_rate": 0.01,
    "epochs": 5,
    "batch_size": 32
}

# For keeping maximum validation score and updating when we get a higher score
max_val_score = 0

# Grid searching best parameters in 'param_grid'
for i, learning_rate in enumerate(param_grid["learning_rates"]):
    for j, epochs in enumerate(param_grid["epochs"]):
        for k, batch_size in enumerate(param_grid["batch_sizes"]):
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            for t in range(epochs):
                print(f"Epoch {t + 1}\n-------------------------------")
                train_loop(train_loader, model, loss_fn, optimizer)
                val_scores.append(test_loop(val_loader, model, loss_fn))
            mean_val_score = np.mean(val_scores)
            if mean_val_score > max_val_score:
                max_val_score = mean_val_score
                best_params["learning_rate"] = learning_rate
                best_params["epochs"] = epochs
                best_params["batch_size"] = batch_size
            print("\n---------batch_sizes---------")
        print('\n---------epochs---------')
    print('\n---------learning_rates---------')
 
print("Done!")

print(f"maximum validation score = {max_val_score}")
print(f"best hyperparameters found are: {best_params}")

loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.01
epochs = 15
batch_size = 2**7

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)

torch.save(model.state_dict(), "my_model_weights.pth")

test_loop(test_loader, model, loss_fn)
