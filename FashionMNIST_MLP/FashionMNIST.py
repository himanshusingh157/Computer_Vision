#Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt


#Download Dataset if needed
data_folder = "~\data\FMNIST"
fmnist = datasets.FashionMNIST(data_folder, download = True, train = True)
images = fmnist.data
targets = fmnist.targets


#Plot Random images for all classe
R = len(targets.unique())
C = 10
fig, ax = plt.subplots(R,C, figsize = (10,10))
for class_, plt_row in enumerate(ax):
    label_x = np.where(targets == class_)[0]
    for plt_square in plt_row:
        plt_square.grid(False)
        plt_square.axis('off')
        index = np.random.choice(label_x)
        img = images[index]
        plt_square.imshow(img, cmap = 'gray')
plt.tight_layout()
plt.savefig('Data.png')
    
#Defining Datatset    
class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        x = x.float()/255
        self.X = x.view(-1,28*28)
        self.Y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return self.X[index], self.Y[index]

#Defining Model
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,200)
        self.layer2 = nn.Linear(200,10)
        self.actiavtionfn = nn.Sigmoid()
        
    def forward(self,x):
        y = self.actiavtionfn(self.layer1(x))
        y = self.layer2(y)
        return y

#Check Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device Available : {device}")

#Model, Data, Loss, Optimizer
traindata = MyDataset(images[:50000],targets[:50000])
trainloader = DataLoader(traindata, batch_size = 128, shuffle = True)
testdata = MyDataset(images[50000:],targets[50000:])
testloader = DataLoader(testdata, batch_size = 128, shuffle = True)
net = MyNet()
torch.save(net.state_dict(), "model.pth")
net = net.to(device)
lossfn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr = 0.001)
epochs = 100


#Training with SGD
sgd_train_losses = []
sgd_train_accuracies = []
sgd_test_losses = []
sgd_test_accuracies = []
for epoch in range(epochs):
    net.train()
    train_epoch_losses = []
    train_epoch_accuracies = []
    for x,y in trainloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        train_epoch_accuracies.append(correct.sum().item())
        optim.zero_grad()
        loss = lossfn(yhat,y)
        loss.backward()
        optim.step()
        train_epoch_losses.append(loss.item())
    sgd_train_losses.append(np.array(train_epoch_losses).mean())
    sgd_train_accuracies.append(np.array(train_epoch_accuracies).sum())
    net.eval()
    test_epoch_losses = []
    test_epoch_accuracies = []
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        loss = lossfn(yhat,y)
        test_epoch_losses.append(loss.item())
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        test_epoch_accuracies.append(correct.sum().item())
    sgd_test_accuracies.append(np.array(test_epoch_accuracies).sum())
    sgd_test_losses.append(np.array(test_epoch_losses).mean())
    print(f"Epoch({epoch}) : Train loss = {sgd_train_losses[-1]:.2f} || \
    Train Accuracy = {sgd_train_accuracies[-1]/50000:.2f} || \
    Test loss = {sgd_test_losses[-1]:.2f} || \
    Test Accuracy = {sgd_test_accuracies[-1]/10000:.2f} ")
torch.save(net.state_dict(), "final_SGD.pth")

#Training with Adam
net.load_state_dict(torch.load("model.pth"))
optim = torch.optim.Adam(net.parameters(), lr = 0.001)
adam_train_losses = []
adam_train_accuracies = []
adam_test_losses = []
adam_test_accuracies = []
for epoch in range(epochs):
    net.train()
    train_epoch_losses = []
    train_epoch_accuracies = []
    for x,y in trainloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        train_epoch_accuracies.append(correct.sum().item())
        optim.zero_grad()
        loss = lossfn(yhat,y)
        loss.backward()
        optim.step()
        train_epoch_losses.append(loss.item())
    adam_train_losses.append(np.array(train_epoch_losses).mean())
    adam_train_accuracies.append(np.array(train_epoch_accuracies).sum())
    net.eval()
    test_epoch_losses = []
    test_epoch_accuracies = []
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        loss = lossfn(yhat,y)
        test_epoch_losses.append(loss.item())
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        test_epoch_accuracies.append(correct.sum().item())
    adam_test_accuracies.append(np.array(test_epoch_accuracies).sum())
    adam_test_losses.append(np.array(test_epoch_losses).mean())
    print(f"Epoch({epoch}) : Train loss = {adam_train_losses[-1]:.2f} || \
    Train Accuracy = {adam_train_accuracies[-1]/50000:.2f} || \
    Test loss = {adam_test_losses[-1]:.2f} || \
    Test Accuracy = {adam_test_accuracies[-1]/10000:.2f} ")
torch.save(net.state_dict(), "final_adam.pth")


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(np.arange(epochs)+1, sgd_train_losses, label = "Training")
axs[0, 0].plot(np.arange(epochs)+1, sgd_test_losses, label = "Validation")
axs[0, 0].set_title('SGD Loss')
axs[0, 0].legend()

axs[0, 1].plot(np.arange(epochs)+1, np.array(sgd_train_accuracies)/500, label = "Training")
axs[0, 1].plot(np.arange(epochs)+1, np.array(sgd_test_accuracies)/100, label = "Validation")
axs[0, 1].set_title('SGD Accuracy')
axs[0, 1].legend()

axs[1, 0].plot(np.arange(epochs)+1, adam_train_losses, label = "Training")
axs[1, 0].plot(np.arange(epochs)+1, adam_test_losses, label = "Validation")
axs[1, 0].set_title('Adam Loss')
axs[1, 0].legend()

axs[1, 1].plot(np.arange(epochs)+1, np.array(adam_train_accuracies)/500, label = "Training")
axs[1, 1].plot(np.arange(epochs)+1, np.array(adam_test_accuracies)/100, label = "Validation")
axs[1, 1].set_title('Adam Accuracy')
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('Epochs.png')



