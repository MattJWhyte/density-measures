
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

DATASET_DIR = "datasets/val_dataset/"


class ClassificationDataset(Dataset):

    def __init__(self):
        self.labels = np.load(DATASET_DIR + "annotation.npy")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.load(DATASET_DIR + "sample-{}.npy".format(idx))).float(),\
               torch.from_numpy(np.array([self.labels[idx]])).float()


class ValNet(nn.Module):

    def __init__(self):
        super(ValNet, self).__init__()
        # No more than 1024
        self.fc1 = nn.Linear(11206, 360)
        self.fc2 = nn.Linear(360, 1)

        self.fc1_bn = nn.BatchNorm1d(360)

    def forward(self, x):
        x = F.relu(self.fc1_bn(self.fc1(x)))
        return F.relu(self.fc2(x))

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


def epoch(dataloader, model, loss_fn, optimizer=None):
    size = len(dataloader.dataset)
    correct = 0.0
    epoch_loss = 0.0
    ct = 0.0

    istrain = optimizer is not None
    torch.autograd.set_grad_enabled(istrain)
    model = model.train() if istrain else model.eval()

    for batch, (X, y) in enumerate(dataloader):
        ct += 1
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        if istrain:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred = pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        correct += np.count_nonzero(np.abs(pred - y) < 0.1)

        epoch_loss += loss.item()

        if istrain and batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    epoch_loss /= ct
    correct /= float(len(dataloader.dataset))
    if istrain:
        print("Train loss: {}".format(epoch_loss))
        print("Train accuracy: {}".format(correct))
        train_loss_ls.append(epoch_loss)
        train_acc_ls.append(correct)
    else:
        print("Test loss: {}".format(epoch_loss))
        print("Test accuracy: {}".format(correct))
        test_loss_ls.append(epoch_loss)
        test_acc_ls.append(correct)


dataset = ClassificationDataset()
train_set, val_set = torch.utils.data.random_split(dataset, [30000, 16656])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_ls = []
train_acc_ls = []
test_loss_ls = []
test_acc_ls = []

train_dataloader = DataLoader(train_set, batch_size=512, shuffle=True)
test_dataloader = DataLoader(val_set, batch_size=512)

model = ValNet()
model.to(device)
name = type(model).__name__.lower()

loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch(train_dataloader, model, loss_fn, optimizer)
    epoch(test_dataloader, model, loss_fn)
    model.save("models/net-"+name+".pth")
    plt.plot([i for i in range(1,t+2)], train_acc_ls, 'r-', label="Train acc.")
    plt.plot([i for i in range(1, t + 2)], train_loss_ls, 'r--', label="Train loss")
    plt.plot([i for i in range(1, t + 2)], test_acc_ls, 'b-', label="Test acc.")
    plt.plot([i for i in range(1, t + 2)], test_loss_ls, 'b--', label="Test loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("training-"+name+".png")
    plt.clf()
